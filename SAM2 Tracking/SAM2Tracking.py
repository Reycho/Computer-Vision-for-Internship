import cv2
import torch
import os
import numpy as np
import argparse
import json
import uuid
from PIL import Image

from sam2.build_sam import build_sam2_video_predictor
from sam2.utils import misc as sam2_misc

BBOX_COLORS_RGB = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
    (255, 0, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0)
]
SUPPORTED_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')

def format_bbox_for_output_json(bbox_coords_xyxy):
    if not (isinstance(bbox_coords_xyxy, list) and len(bbox_coords_xyxy) == 4 and bbox_coords_xyxy[0] < bbox_coords_xyxy[2] and bbox_coords_xyxy[1] < bbox_coords_xyxy[3]):
        return []
    xmin, ymin, xmax, ymax = bbox_coords_xyxy
    return [
        {"x": float(xmin), "y": float(ymin)}, {"x": float(xmax), "y": float(ymin)},
        {"x": float(xmax), "y": float(ymax)}, {"x": float(xmin), "y": float(ymax)}
    ]

def get_sorted_image_files(image_dir_path):
    try:
        image_filenames = sorted(os.listdir(image_dir_path), key=lambda f: int(os.path.splitext(f)[0]))
    except ValueError:
        print("Warning: Could not sort filenames numerically. Falling back to alphabetical sort.")
        image_filenames = sorted(os.listdir(image_dir_path))
    image_files = [os.path.join(image_dir_path, f) for f in image_filenames if f.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS)]
    if not image_files: raise FileNotFoundError(f"No images in {image_dir_path}")
    return image_files

def draw_final_masks(image_np, masks_to_draw, alpha=0.5):
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
    if not images:
        print("No annotated frames found to create video.")
        return
    frame = cv2.imread(images[0])
    if frame is None:
        print(f"Error: Could not read first frame: {images[0]}")
        return
    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path, fourcc, float(fps), (width, height))
    for image_file in images:
        img = cv2.imread(image_file)
        if img is not None:
            video.write(img)
    video.release()
    print(f"--> Video saved successfully to {video_path}")

def main():
    parser = argparse.ArgumentParser(description="Final, OPTIMIZED SAM2 tracker using box prompts and creating a video output.")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--json_annotation", type=str, required=True)
    parser.add_argument("--model_config_yaml", type=str, required=True)
    parser.add_argument("--local_checkpoint_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="sam2_final_optimized_output")
    parser.add_argument("--mask_threshold", type=float, default=0.0)
    parser.add_argument("--make_video", action='store_true')
    parser.add_argument("--video_fps", type=float, default=10.0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    annotated_dir = os.path.join(args.output_dir, "annotated_frames")
    os.makedirs(annotated_dir, exist_ok=True)
    jsonl_path = os.path.join(args.output_dir, "tracked_detections.jsonl")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    autocast_context = torch.autocast("cuda", dtype=torch.bfloat16)
    if torch.cuda.get_device_properties(0).major >= 8:
        print("Enabling TF32 for Ampere GPUs.")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print("Initializing SAM2VideoPredictor...")
    predictor = build_sam2_video_predictor(
        args.model_config_yaml,
        args.local_checkpoint_path,
        device=device,
        vos_optimized=True
    )
    predictor.eval()

    with autocast_context:
        image_files = get_sorted_image_files(args.image_dir)
        print("Initializing inference state (first run may be slow due to compilation)...")
        inference_state = predictor.init_state(video_path=args.image_dir)

        with open(args.json_annotation, 'r') as f:
            objects_initial = json.load(f).get("annotation", {}).get("object", [])

        script_id_map = {}
        
        print("\n--- Initializing objects on Frame 0 with Bounding Box Prompts ---")
        original_first_frame_np = np.array(Image.open(image_files[0]).convert("RGB"))
        frame_0_prelabels, masks_for_frame_0 = [], []
        
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
                mask_bool = (frame_masks[obj_idx, 0].cpu().numpy() > args.mask_threshold)
                if np.any(mask_bool):
                    print(f"  --> SUCCESS: Initialized object {i+1}.")
                    persistent_uid = str(uuid.uuid4())
                    script_id_map[predictor_obj_id] = {
                        "name": obj_data['name'], 
                        "script_id_int": i + 1,
                        "uid": persistent_uid  # Store the persistent ID
                    }
                    color = BBOX_COLORS_RGB[i % len(BBOX_COLORS_RGB)]
                    masks_for_frame_0.append({"mask": mask_bool, "color": color, "text": f"ID:{i+1}"})
                    bbox = sam2_misc.mask_to_box(torch.from_numpy(mask_bool).unsqueeze(0).unsqueeze(0))[0, 0].numpy().tolist()
                    # Use the stored persistent_uid for the output JSON
                    frame_0_prelabels.append({"name": obj_data['name'], "uid": persistent_uid, "type": "rect", "points": format_bbox_for_output_json(bbox)})
                else:
                    print(f"  --> WARNING: Empty mask for object {i+1}. Skipping.")

        annotated_frame_0 = draw_final_masks(original_first_frame_np, masks_for_frame_0)
        cv2.imwrite(os.path.join(annotated_dir, os.path.basename(image_files[0])), cv2.cvtColor(annotated_frame_0, cv2.COLOR_RGB2BGR))
        with open(jsonl_path, 'w') as f_out:
            f_out.write(json.dumps({"fileName": os.path.basename(image_files[0]), "prelabels": frame_0_prelabels}) + '\n')

        if not script_id_map:
            print("\nNo objects were initialized successfully. Exiting.")
            return

        print(f"\nInitialized {len(script_id_map)} objects. Starting propagation (first run may be slow due to compilation)...")
        for frame_idx, pred_obj_ids, frame_masks_all in predictor.propagate_in_video(inference_state, start_frame_idx=1):
            print(f"  Processing frame: {os.path.basename(image_files[frame_idx])}", end='\r')
            original_frame_np = np.array(Image.open(image_files[frame_idx]).convert("RGB"))
            frame_prelabels, masks_for_current_frame = [], []

            if pred_obj_ids:
                for i, pred_id in enumerate(pred_obj_ids):
                    if pred_id in script_id_map:
                        obj_info = script_id_map[pred_id]
                        mask_bool_np = (frame_masks_all[i, 0].cpu().numpy() > args.mask_threshold)
                        if np.any(mask_bool_np):
                            color = BBOX_COLORS_RGB[(obj_info["script_id_int"] - 1) % len(BBOX_COLORS_RGB)]
                            masks_for_current_frame.append({"mask": mask_bool_np, "color": color, "text": f"ID:{obj_info['script_id_int']}"})
                            bbox = sam2_misc.mask_to_box(torch.from_numpy(mask_bool_np).unsqueeze(0).unsqueeze(0))[0, 0].numpy().tolist()
                            # Retrieve the persistent UUID from the map ---
                            frame_prelabels.append({
                                "name": obj_info["name"], 
                                "uid": obj_info["uid"], # Use the stored ID instead of generating a new one
                                "type": "rect", 
                                "points": format_bbox_for_output_json(bbox)
                            })
            
            annotated_frame = draw_final_masks(original_frame_np, masks_for_current_frame)
            cv2.imwrite(os.path.join(annotated_dir, os.path.basename(image_files[frame_idx])), cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
            if frame_prelabels:
                with open(jsonl_path, 'a') as f_out:
                    f_out.write(json.dumps({"fileName": os.path.basename(image_files[frame_idx]), "prelabels": frame_prelabels}) + '\n')
        print("\nTracking complete.                           ")

    if args.make_video:
        create_video_from_frames(annotated_dir, os.path.join(args.output_dir, "tracking_video.mp4"), args.video_fps)

if __name__ == "__main__":
    main()

# Example:
# python SAM2Tracking.py     --image_dir ./inputs/FrontCam02     --json_annotation ./inputsjson/FrontCam02.json     --model_config_yaml "configs/sam2.1/sam2.1_hiera_l.yaml"     --local_checkpoint_path ./checkpoints/sam2.1_hiera_large.pt     --make_video