import cv2
import torch
import os
import numpy as np
import argparse
import json
import uuid
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from sam2.build_sam import build_sam2_video_predictor
from sam2.utils import amg

# --- Helper Functions (unchanged) ---
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
        mask_bool, color, text, box = mask_info["mask"], mask_info["color"], mask_info["text"], mask_info["box"]
        overlay = np.zeros_like(output_image, dtype=np.uint8)
        overlay[mask_bool] = color
        output_image = cv2.addWeighted(output_image, 1.0, overlay, alpha, 0)
        xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
        cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), color, 2)
        if text:
            cv2.putText(output_image, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return output_image

def create_video_from_frames(image_folder, video_path, fps):
    print(f"\nCreating video from annotated frames...")
    images = get_sorted_image_files(image_folder)
    if not images:
        print("--> No images found to create a video.")
        return
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
    # This function remains mostly the same, but now it returns the full result dict for frame 0
    script_id_map = {}
    frame_0_results = {"visuals": [], "prelabels": []}
    
    for i, obj_data in enumerate(objects_initial):
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
            mask_tensor_gpu = frame_masks[obj_idx, 0]
            mask_bool_gpu = mask_tensor_gpu > mask_threshold

            if torch.any(mask_bool_gpu):
                persistent_uid = str(uuid.uuid4())
                script_id_map[predictor_obj_id] = {"name": obj_data['name'], "script_id_int": i + 1, "uid": persistent_uid}
                color = BBOX_COLORS_RGB[i % len(BBOX_COLORS_RGB)]

                bbox_gpu = amg.batched_mask_to_box(mask_bool_gpu.unsqueeze(0))
                bbox_cpu = bbox_gpu[0].cpu().numpy().astype(int)

                frame_0_results["visuals"].append({
                    "mask": mask_bool_gpu.cpu().numpy(), "color": color, "text": f"ID:{i+1}", "box": bbox_cpu
                })
                frame_0_results["prelabels"].append({
                    "name": obj_data['name'], "uid": persistent_uid, "type": "rect",
                    "points": [{"x": float(p[0]), "y": float(p[1])} for p in [[bbox_cpu[0],bbox_cpu[1]],[bbox_cpu[2],bbox_cpu[1]],[bbox_cpu[2],bbox_cpu[3]],[bbox_cpu[0],bbox_cpu[3]]]]
                })
    return script_id_map, frame_0_results

def save_output_worker(args_tuple):
    frame_idx, worker_payload, image_files, args = args_tuple
    if "visuals" in worker_payload:
        visuals_data = worker_payload["visuals"]
        current_img_path = image_files[frame_idx]
        annotated_dir = os.path.join(args.output_dir, "annotated_frames")
        original_frame_np = np.array(Image.open(current_img_path).convert("RGB"))
        annotated_frame = draw_visualizations(original_frame_np, visuals_data)
        jpeg_quality = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        output_path = os.path.join(annotated_dir, os.path.basename(current_img_path))
        cv2.imwrite(output_path, cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR), jpeg_quality)
    if worker_payload["prelabels"]:
        return json.dumps({"fileName": os.path.basename(image_files[frame_idx]), "prelabels": worker_payload["prelabels"]})
    return None

def propagate_and_save_streamed(predictor, inference_state, image_files, script_id_map, frame_0_results, args):
    """
    This is the core of the new streaming architecture.
    It overlaps inference with saving to keep memory usage low and eliminate pauses.
    """
    jsonl_path = os.path.join(args.output_dir, "tracked_detections.jsonl")
    if os.path.exists(jsonl_path):
        os.remove(jsonl_path)

    with ProcessPoolExecutor() as executor, open(jsonl_path, 'w') as f_out:
        futures = []
        
        # --- Handle Frame 0 ---
        # Submit the first frame's results for saving immediately.
        if frame_0_results["prelabels"]:
            worker_payload = {"prelabels": frame_0_results["prelabels"]}
            if not args.no_save_annotated_frames:
                worker_payload["visuals"] = frame_0_results["visuals"]
            task_args = (0, worker_payload, image_files, args)
            futures.append(executor.submit(save_output_worker, task_args))
        
        print(f"\nInitialized {len(script_id_map)} objects. Starting propagation and saving stream...")
        
        # --- Main Inference and Saving Loop (Frames 1+) ---
        # The main process will focus on inference, while worker processes handle saving in the background.
        for frame_idx, pred_obj_ids, frame_masks_all in predictor.propagate_in_video(inference_state, start_frame_idx=1):
            valid_masks_gpu, valid_obj_info = [], []
            if pred_obj_ids:
                for i, pred_id in enumerate(pred_obj_ids):
                    if pred_id in script_id_map:
                        mask_bool_gpu = frame_masks_all[i, 0] > args.mask_threshold
                        if torch.any(mask_bool_gpu):
                            valid_masks_gpu.append(mask_bool_gpu.unsqueeze(0))
                            valid_obj_info.append(script_id_map[pred_id])

            if not valid_masks_gpu:
                continue
            
            # Perform GPU-side calcs for the current frame
            batched_masks_gpu = torch.cat(valid_masks_gpu, dim=0)
            batched_boxes_gpu = amg.batched_mask_to_box(batched_masks_gpu)

            # Transfer only this frame's data to CPU, create payload, and submit to worker.
            # The main process can then drop this data from memory.
            worker_payload = {"prelabels": []}
            if not args.no_save_annotated_frames:
                worker_payload["visuals"] = []

            batched_boxes_cpu = batched_boxes_gpu.cpu().numpy().astype(int)
            # Transfer masks to CPU only if they need to be saved as images
            batched_masks_cpu = batched_masks_gpu.cpu().numpy() if not args.no_save_annotated_frames else None
            
            for i, obj_info in enumerate(valid_obj_info):
                box_coords = batched_boxes_cpu[i]
                if not args.no_save_annotated_frames:
                    color = BBOX_COLORS_RGB[(obj_info["script_id_int"] - 1) % len(BBOX_COLORS_RGB)]
                    worker_payload["visuals"].append({
                        "mask": batched_masks_cpu[i], "color": color, "text": f"ID:{obj_info['script_id_int']}", "box": box_coords
                    })
                worker_payload["prelabels"].append({
                    "name": obj_info["name"], "uid": obj_info["uid"], "type": "rect",
                    "points": [{"x": float(p[0]), "y": float(p[1])} for p in [[box_coords[0],box_coords[1]],[box_coords[2],box_coords[1]],[box_coords[2],box_coords[3]],[box_coords[0],box_coords[3]]]]
                })
            
            task_args = (frame_idx, worker_payload, image_files, args)
            futures.append(executor.submit(save_output_worker, task_args))
        
        print("\nInference complete. Waiting for final save operations to finish...")
        
        # --- Collect Results ---
        desc_text = "Writing JSONL"
        if not args.no_save_annotated_frames:
            desc_text = "Writing JSONL and image files"
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc_text):
            json_line = future.result()
            if json_line:
                f_out.write(json_line + '\n')


def main():
    parser = argparse.ArgumentParser(description="SAM2 Video Tracker")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to the directory of input image frames.")
    parser.add_argument("--json_annotation", type=str, required=True, help="Path to the JSON file with initial bounding box annotations for the first frame.")
    parser.add_argument("--model_config_yaml", type=str, required=True, help="Path to the SAM2 model configuration YAML file.")
    parser.add_argument("--local_checkpoint_path", type=str, required=True, help="Path to the local SAM2 model checkpoint (.pt file).")
    parser.add_argument("--output_dir", type=str, default="sam2_output", help="Directory to save the output files.")
    parser.add_argument("--mask_threshold", type=float, default=0.0, help="Threshold for converting mask probabilities to binary masks.")
    parser.add_argument("--make_video", action='store_true', help="If set, create an MP4 video from the annotated frames.")
    parser.add_argument("--video_fps", type=float, default=10.0, help="Frames per second for the output video.")
    parser.add_argument("--no_save_annotated_frames", action='store_true', help="If set, do not save annotated frame images to disk. This significantly speeds up processing if only JSONL output is needed.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    annotated_dir = None
    if not args.no_save_annotated_frames:
        annotated_dir = os.path.join(args.output_dir, "annotated_frames")
        os.makedirs(annotated_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    use_bfloat16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    autocast_context = torch.autocast("cuda", dtype=torch.bfloat16) if use_bfloat16 else torch.autocast("cuda")

    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        print("Enabling TF32 for Ampere and newer GPUs.")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print("Initializing SAM2VideoPredictor...")
    predictor = build_sam2_video_predictor(args.model_config_yaml, args.local_checkpoint_path, device=device, apply_postprocessing=True)
    predictor.eval()

    inference_state = None

    try:
        with autocast_context:
            image_files = get_sorted_image_files(args.image_dir)
            print("Initializing inference state with async loader...")
            inference_state = predictor.init_state(video_path=args.image_dir, async_loading_frames=True)
            
            with open(args.json_annotation, 'r') as f:
                objects_initial = json.load(f).get("annotation", {}).get("object", [])

            script_id_map, frame_0_results = initialize_tracks(
                predictor, inference_state, objects_initial, args.mask_threshold
            )
            
            if not script_id_map:
                print("\nNo objects initialized. Exiting.")
                return

            propagate_and_save_streamed(predictor, inference_state, image_files, script_id_map, frame_0_results, args)

    finally:
        if inference_state and inference_state.get('video_reader'):
            print("\nShutting down the asynchronous frame loader...")
            inference_state['video_reader'].close()
            
    if args.make_video:
        if not args.no_save_annotated_frames:
            create_video_from_frames(annotated_dir, os.path.join(args.output_dir, "tracking_video.mp4"), args.video_fps)
        else:
            print("\nWarning: --make_video was specified, but --no_save_annotated_frames is active. Cannot create video.")

    print("\nProcessing complete.")

if __name__ == "__main__":
    main()
