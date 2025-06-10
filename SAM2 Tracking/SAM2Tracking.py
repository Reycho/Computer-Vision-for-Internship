# sam2_video_tracker_parallel.py
# Final version with parallelized output generation for maximum speed.

import cv2
import torch
import os
import numpy as np
import argparse
import json
import uuid
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from sam2.build_sam import build_sam2_video_predictor
from sam2.utils import misc as sam2_misc

# --- Helper Functions (Proven Correct) ---
BBOX_COLORS_RGB = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
    (255, 0, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0)
]
def get_sorted_image_files(image_dir_path):
    try:
        image_filenames = sorted(os.listdir(image_dir_path), key=lambda f: int(os.path.splitext(f)[0]))
    except ValueError:
        print("Warning: Could not sort filenames numerically. Falling back to alphabetical sort.")
        image_filenames = sorted(os.listdir(image_dir_path))
    return [os.path.join(image_dir_path, f) for f in image_filenames if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

def draw_visualizations(image_np, masks_to_draw, alpha=0.5):
    output_image = image_np.copy()
    for mask_info in masks_to_draw:
        mask_bool, color, text = mask_info["mask"], mask_info["color"], mask_info["text"]
        overlay = np.zeros_like(output_image, dtype=np.uint8)
        overlay[mask_bool] = color
        output_image = cv2.addWeighted(output_image, 1.0, overlay, alpha, 0)
        rows, cols = np.where(mask_bool)
        if len(rows) > 0:
            xmin, xmax = cols.min(), cols.max()
            ymin, ymax = rows.min(), rows.max()
            cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), color, 2)
            if text:
                cv2.putText(output_image, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return output_image

def create_video_from_frames(image_folder, video_path, fps):
    print(f"\nCreating video from annotated frames...")
    images = get_sorted_image_files(image_folder)
    if not images: return
    frame = cv2.imread(images[0])
    if frame is None: return
    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path, fourcc, float(fps), (width, height))
    for image_file in images:
        img = cv2.imread(image_file)
        if img is not None: video.write(img)
    video.release()
    print(f"--> Video saved successfully to {video_path}")
# --- End of Helper Functions ---


def initialize_tracks(predictor, inference_state, objects_initial, mask_threshold):
    print("\n--- Initializing objects on Frame 0 with Bounding Box Prompts ---")
    script_id_map, masks_for_frame_0, frame_0_prelabels = {}, [], []
    for i, obj_data in enumerate(objects_initial):
        # ... This function's internal logic is identical ...
        predictor_obj_id = str(i + 1)
        bndbox = obj_data["bndbox"]
        xmin, ymin, xmax, ymax = int(bndbox["xmin"]), int(bndbox["ymin"]), int(bndbox["xmax"]), int(bndbox["ymax"])
        box_prompt = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
        _, returned_obj_ids, frame_masks = predictor.add_new_points_or_box(
            inference_state=inference_state, frame_idx=0, obj_id=predictor_obj_id,
            box=box_prompt, normalize_coords=True
        )
        if returned_obj_ids and predictor_obj_id in returned_obj_ids:
            obj_idx = returned_obj_ids.index(predictor_obj_id)
            mask_bool = (frame_masks[obj_idx, 0].cpu().numpy() > mask_threshold)
            if np.any(mask_bool):
                print(f"  --> SUCCESS: Initialized object {i+1}.")
                persistent_uid = str(uuid.uuid4())
                script_id_map[predictor_obj_id] = {"name": obj_data['name'], "script_id_int": i + 1, "uid": persistent_uid}
                color = BBOX_COLORS_RGB[i % len(BBOX_COLORS_RGB)]
                masks_for_frame_0.append({"mask": mask_bool, "color": color, "text": f"ID:{i+1}"})
                bbox = sam2_misc.mask_to_box(torch.from_numpy(mask_bool).unsqueeze(0).unsqueeze(0))[0, 0].numpy().tolist()
                frame_0_prelabels.append({"name": obj_data['name'], "uid": persistent_uid, "type": "rect", "points": [{"x": float(p[0]), "y": float(p[1])} for p in [[bbox[0],bbox[1]],[bbox[2],bbox[1]],[bbox[2],bbox[3]],[bbox[0],bbox[3]]]]})
    return script_id_map, masks_for_frame_0, frame_0_prelabels


def propagate_and_collect_results(predictor, inference_state, image_files, script_id_map, args):
    """PHASE 1: GPU-INTENSIVE - Propagates tracks and collects all results in memory."""
    print(f"\nInitialized {len(script_id_map)} objects. Starting propagation (inference only)...")
    all_frame_results = {}
    # The predictor's internal loop will print a clean tqdm progress bar for the GPU work.
    for frame_idx, pred_obj_ids, frame_masks_all in predictor.propagate_in_video(inference_state, start_frame_idx=1):
        frame_results = {"visuals": [], "prelabels": []}
        if pred_obj_ids:
            for i, pred_id in enumerate(pred_obj_ids):
                if pred_id in script_id_map:
                    obj_info = script_id_map[pred_id]
                    mask_bool_np = (frame_masks_all[i, 0].cpu().numpy() > args.mask_threshold)
                    if np.any(mask_bool_np):
                        color = BBOX_COLORS_RGB[(obj_info["script_id_int"] - 1) % len(BBOX_COLORS_RGB)]
                        frame_results["visuals"].append({"mask": mask_bool_np, "color": color, "text": f"ID:{obj_info['script_id_int']}"})
                        bbox = sam2_misc.mask_to_box(torch.from_numpy(mask_bool_np).unsqueeze(0).unsqueeze(0))[0, 0].numpy().tolist()
                        frame_results["prelabels"].append({
                            "name": obj_info["name"], "uid": obj_info["uid"], "type": "rect",
                            "points": [{"x": float(p[0]), "y": float(p[1])} for p in [[bbox[0],bbox[1]],[bbox[2],bbox[1]],[bbox[2],bbox[3]],[bbox[0],bbox[3]]]]
                        })
        all_frame_results[frame_idx] = frame_results
    print("\nInference complete.")
    return all_frame_results

# --- NEW WORKER FUNCTION ---
# This function must be at the top level of the script so it can be "pickled" and sent to other processes.
def save_output_worker(args_tuple):
    """A single worker's job: save the output for one frame."""
    frame_idx, results, image_files, args = args_tuple
    
    current_img_path = image_files[frame_idx]
    annotated_dir = os.path.join(args.output_dir, "annotated_frames")
    jsonl_path = os.path.join(args.output_dir, "tracked_detections.jsonl")

    # Save visualization
    original_frame_np = np.array(Image.open(current_img_path).convert("RGB"))
    annotated_frame = draw_visualizations(original_frame_np, results["visuals"])
    cv2.imwrite(os.path.join(annotated_dir, os.path.basename(current_img_path)), cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
    
    # Return the data to be written to the main JSONL file
    if results["prelabels"]:
        return json.dumps({"fileName": os.path.basename(current_img_path), "prelabels": results["prelabels"]})
    return None

def save_all_outputs_parallel(all_frame_results, image_files, args):
    """PHASE 2: CPU/IO-INTENSIVE (Parallelized) - Saves all collected results to disk using multiple processes."""
    print("\n--- Saving all outputs in parallel across CPU cores ---")
    
    # Prepare the list of jobs for the worker pool
    tasks = [(frame_idx, results, image_files, args) for frame_idx, results in all_frame_results.items()]
    
    jsonl_lines = []
    # Use ProcessPoolExecutor to manage a pool of worker processes
    with ProcessPoolExecutor() as executor:
        # executor.map distributes the tasks to the workers. tqdm tracks the progress.
        # The result from each worker (the JSONL string) is collected as it completes.
        for result in tqdm(executor.map(save_output_worker, tasks), total=len(tasks), desc="Saving outputs"):
            if result:
                jsonl_lines.append(result + '\n')

    # Write all the JSONL lines to the file at once
    jsonl_path = os.path.join(args.output_dir, "tracked_detections.jsonl")
    with open(jsonl_path, 'w') as f_out: # 'w' to overwrite with all data
        f_out.writelines(jsonl_lines)

    print("All outputs saved.")


def main():
    parser = argparse.ArgumentParser(description="SAM2 Video Tracker (Parallelized Output Version)")
    # ... (args parsing is identical) ...
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--json_annotation", type=str, required=True)
    parser.add_argument("--model_config_yaml", type=str, required=True)
    parser.add_argument("--local_checkpoint_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="sam2_parallel_output")
    parser.add_argument("--mask_threshold", type=float, default=0.0)
    parser.add_argument("--make_video", action='store_true')
    parser.add_argument("--video_fps", type=float, default=10.0)
    args = parser.parse_args()

    # ... (setup is identical) ...
    annotated_dir = os.path.join(args.output_dir, "annotated_frames")
    os.makedirs(annotated_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    autocast_context = torch.autocast("cuda", dtype=torch.bfloat16)
    if torch.cuda.get_device_properties(0).major >= 8:
        print("Enabling TF32 for Ampere GPUs.")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print("Initializing SAM2VideoPredictor...")
    predictor = build_sam2_video_predictor(args.model_config_yaml, args.local_checkpoint_path, device=device)
    predictor.eval()

    # --- Phase 1: Inference ---
    all_tracked_results = {}
    with autocast_context:
        image_files = get_sorted_image_files(args.image_dir)
        print("Initializing inference state...")
        inference_state = predictor.init_state(video_path=args.image_dir, async_loading_frames=True)
        with open(args.json_annotation, 'r') as f:
            objects_initial = json.load(f).get("annotation", {}).get("object", [])

        script_id_map, masks_for_frame_0, frame_0_prelabels = initialize_tracks(
            predictor, inference_state, objects_initial, args.mask_threshold
        )
        all_tracked_results[0] = {"visuals": masks_for_frame_0, "prelabels": frame_0_prelabels}
        
        if script_id_map:
            other_frames_results = propagate_and_collect_results(
                predictor, inference_state, image_files, script_id_map, args
            )
            all_tracked_results.update(other_frames_results)
        else:
            print("\nNo objects initialized. Exiting.")
            return

    # --- Phase 2: Parallel Output ---
    save_all_outputs_parallel(all_tracked_results, image_files, args)

    # --- Phase 3: Video Creation ---
    if args.make_video:
        create_video_from_frames(annotated_dir, os.path.join(args.output_dir, "tracking_video.mp4"), args.video_fps)

    print("\nProcessing complete.")

if __name__ == "__main__":
    main()
