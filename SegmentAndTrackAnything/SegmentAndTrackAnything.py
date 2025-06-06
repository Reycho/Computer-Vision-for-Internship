import cv2
import torch
import os
import numpy as np
import argparse
import json 
import gc
import time
import uuid 

# Project specific imports
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_args import segtracker_args, sam_args, aot_args
from SegTracker import SegTracker
from tool.transfer_tools import mask2bbox

BBOX_COLORS_RGB = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), 
    (255, 0, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (0, 128, 128), (128, 0, 128), (255, 165, 0), (255, 192, 203), (75, 0, 130) 
]

def parse_arguments():
    parser = argparse.ArgumentParser(description="Track multiple objects from a JSON annotation and output tracked images and a JSON data file.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing input image frames.")
    parser.add_argument("--json_annotation", type=str, required=True, help="Path to the JSON file (single object) annotating objects in the first frame.")
    parser.add_argument("--base_output_dir", type=str, default="tracking_results", 
                        help="Base directory where a new folder '[input_dir_name]_tracked_output' will be created for results.")
    parser.add_argument("--model_variant", type=str, default="vit_b", choices=['vit_b', 'vit_l', 'vit_h'], 
                        help="SAM model variant.")
    parser.add_argument("--aot_model_variant", type=str, default="r50_deaotl", choices=['deaotb', 'deaotl', 'r50_deaotl'],
                        help="AOT model variant.")
    parser.add_argument("--bbox_thickness", type=int, default=2, help="Thickness of the bounding box.")
    parser.add_argument("--object_name_filter", type=str, default=None, help="Optional: Track only objects with this name from the JSON. If None, tracks all objects.")
    return parser.parse_args()

def format_bbox_for_output_json(bbox_coords): 
    xmin, ymin, xmax, ymax = bbox_coords[0][0], bbox_coords[0][1], bbox_coords[1][0], bbox_coords[1][1]
    return [
        {"x": float(xmin), "y": float(ymin)}, {"x": float(xmax), "y": float(ymin)},
        {"x": float(xmax), "y": float(ymax)}, {"x": float(xmin), "y": float(ymax)}
    ]

def main():
    args = parse_arguments()
    overall_start_time = time.time()

    # --- Determine Output Paths ---
    input_folder_name = os.path.basename(os.path.normpath(args.image_dir))
    main_output_folder_name = f"{input_folder_name}_tracked_output"
    
    if not os.path.exists(args.base_output_dir):
        os.makedirs(args.base_output_dir, exist_ok=True)
        
    final_output_dir = os.path.join(args.base_output_dir, main_output_folder_name)
    output_annotated_image_dir = os.path.join(final_output_dir, "annotated_images")
    output_tracking_data_path = os.path.join(final_output_dir, f"{input_folder_name}_tracked_detections.txt")

    if not os.path.exists(final_output_dir): os.makedirs(final_output_dir, exist_ok=True)
    if not os.path.exists(output_annotated_image_dir): os.makedirs(output_annotated_image_dir, exist_ok=True)
    
    print(f"Input image directory: {args.image_dir}")
    print(f"Input JSON annotation: {args.json_annotation}")
    print(f"All outputs will be saved under: {final_output_dir}")
    print(f"  Annotated images will be in: {output_annotated_image_dir}")
    print(f"  Tracking data TXT file will be: {output_tracking_data_path}") # Updated print

    # --- Model and Device Setup ---
    # ... (SAM and AOT checkpoint setup as before) ...
    sam_checkpoint_map = { 
        "vit_h": "ckpt/sam_vit_h_4b8939.pth", "vit_l": "ckpt/sam_vit_l_0b3195.pth", "vit_b": "ckpt/sam_vit_b_01ec64.pth",
    }
    sam_args['model_type'] = args.model_variant
    sam_args['sam_checkpoint'] = sam_checkpoint_map[args.model_variant]
    if not os.path.exists(sam_args['sam_checkpoint']): print(f"Error: SAM ckpt missing at {sam_args['sam_checkpoint']}"); return

    aot_model2ckpt = { 
        "deaotb": "ckpt/DeAOTB_PRE_YTB_DAV.pth", "deaotl": "ckpt/DeAOTL_PRE_YTB_DAV.pth", "r50_deaotl": "ckpt/R50_DeAOTL_PRE_YTB_DAV.pth",
    }
    aot_args['model'] = args.aot_model_variant
    aot_args['model_path'] = aot_model2ckpt[args.aot_model_variant]
    if not os.path.exists(aot_args['model_path']): print(f"Error: AOT ckpt missing at {aot_args['model_path']}"); return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    sam_args['gpu_id'] = 0 ; aot_args['gpu_id'] = 0
    if device == "cpu": print("Warning: Running on CPU. Slow.")

    model_init_start_time = time.time()
    print("Initializing SegTracker...")
    try:
        seg_tracker = SegTracker(segtracker_args, sam_args, aot_args)
        seg_tracker.restart_tracker() 
    except Exception as e: print(f"Error initializing SegTracker: {e}"); return
    model_init_time = time.time() - model_init_start_time
    print(f"SegTracker initialized in {model_init_time:.2f} seconds.")

    # --- Image and Annotation Loading ---
    image_files = sorted([os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not image_files: print(f"Error: No images found in {args.image_dir}"); return
    num_total_frames = len(image_files)
    print(f"Found {num_total_frames} images for tracking.")
    
    print(f"Loading JSON annotation from {args.json_annotation}...")
    try:
        with open(args.json_annotation, 'r') as f:
            first_frame_annotation_data = json.load(f) 
    except Exception as e: print(f"Error reading/parsing JSON file {args.json_annotation}: {e}"); return

    annotation_details = first_frame_annotation_data.get("annotation", {})
    first_frame_filename_from_json = annotation_details.get("filename")
    objects_to_track_initial = annotation_details.get("object", [])
    
    first_image_filename_from_dir = os.path.basename(image_files[0])
    if first_frame_filename_from_json != first_image_filename_from_dir:
        print(f"Warning: JSON filename ('{first_frame_filename_from_json}') != first image ('{first_image_filename_from_dir}').")
    
    if args.object_name_filter:
        objects_to_track_initial = [obj for obj in objects_to_track_initial if obj.get("name") == args.object_name_filter]
        print(f"Filtered to track {len(objects_to_track_initial)} of '{args.object_name_filter}'.")
    if not objects_to_track_initial: print("Error: No objects in JSON to track."); return

    # --- First Frame Processing ---
    first_frame_proc_start_time = time.time()
    print("Processing the first frame...")
    first_frame_path = image_files[0]
    origin_frame_bgr = cv2.imread(first_frame_path)
    if origin_frame_bgr is None: print(f"Error reading first frame {first_frame_path}"); return
    origin_frame_rgb = cv2.cvtColor(origin_frame_bgr, cv2.COLOR_BGR2RGB)
    
    h, w, _ = origin_frame_rgb.shape
    combined_first_frame_mask_with_ids = np.zeros((h, w), dtype=np.uint8)
    assigned_object_id_counter = 0
    script_id_to_original_name = {} 
    first_frame_output_prelabels = []
    script_id_to_first_frame_derived_bbox_points = {}

    seg_tracker.sam.interactive_predictor.reset_image() 
    seg_tracker.sam.set_image(origin_frame_rgb)         

    with torch.cuda.amp.autocast(enabled=(device == "cuda")):
        for obj_data in objects_to_track_initial:
            assigned_object_id_counter += 1 
            bndbox_dict = obj_data.get("bndbox") 
            if not bndbox_dict: print(f"Warn: No 'bndbox' for {obj_data.get('name', 'Unnamed')}. Skip."); assigned_object_id_counter -=1 ; continue
            try:
                xmin, ymin, xmax, ymax = int(bndbox_dict["xmin"]), int(bndbox_dict["ymin"]), int(bndbox_dict["xmax"]), int(bndbox_dict["ymax"])
            except Exception as e: print(f"Warn: Bad 'bndbox' for {obj_data.get('name', 'Unnamed')}: {e}. Skip."); assigned_object_id_counter -= 1; continue
            
            current_json_bbox_coords = np.array([[xmin, ymin], [xmax, ymax]], dtype=np.int64)
            individual_object_mask_binary_list = seg_tracker.sam.segment_with_box(
                origin_frame_rgb, current_json_bbox_coords, reset_image=False
            )
            if not individual_object_mask_binary_list: print(f"Warn: SAM mask empty for {current_json_bbox_coords}. Skip."); assigned_object_id_counter -=1; continue
            
            individual_object_mask_binary = individual_object_mask_binary_list[0]
            combined_first_frame_mask_with_ids[individual_object_mask_binary > 0] = assigned_object_id_counter
            original_name = obj_data.get('name', f'object_{assigned_object_id_counter}')
            script_id_to_original_name[assigned_object_id_counter] = original_name
            derived_bbox = mask2bbox(individual_object_mask_binary) 
            if derived_bbox[0][0] < derived_bbox[1][0] and derived_bbox[0][1] < derived_bbox[1][1]:
                 formatted_points = format_bbox_for_output_json(derived_bbox)
                 first_frame_output_prelabels.append({
                     "name": original_name, "uid": str(uuid.uuid4()), 
                     "type": "rect", "select": {}, "points": formatted_points
                 })
                 script_id_to_first_frame_derived_bbox_points[assigned_object_id_counter] = formatted_points

    if assigned_object_id_counter == 0: print("Error: No objects segmented for first frame."); return

    seg_tracker.first_frame_mask = combined_first_frame_mask_with_ids
    if device == "cuda":
        with torch.cuda.amp.autocast(enabled=False): 
            seg_tracker.add_reference(origin_frame_rgb, seg_tracker.first_frame_mask, frame_step=0)
    else: 
        seg_tracker.add_reference(origin_frame_rgb, seg_tracker.first_frame_mask, frame_step=0)
    
    first_frame_proc_time = time.time() - first_frame_proc_start_time
    print(f"Initialized {seg_tracker.get_obj_num()} objects (first frame) in {first_frame_proc_time:.2f}s.")

    with open(output_tracking_data_path, 'w') as output_file:
        first_frame_json_output = {"fileName": first_image_filename_from_dir, "prelabels": first_frame_output_prelabels}
        output_file.write(json.dumps(first_frame_json_output) + '\n')

        first_frame_display_rgb = origin_frame_rgb.copy()
        for prelabel_info in first_frame_output_prelabels:
            script_id_for_color = None
            for sid, name_map in script_id_to_original_name.items():
                if name_map == prelabel_info['name'] and \
                   script_id_to_first_frame_derived_bbox_points.get(sid) == prelabel_info['points']:
                    script_id_for_color = sid
                    break
            if script_id_for_color is None:
                print(f"Warning: Could not map prelabel {prelabel_info['name']} back to a script_id for consistent coloring on first frame.")
                color = (128,128,128) 
                display_id_text = prelabel_info['name']
            else:
                color = BBOX_COLORS_RGB[(script_id_for_color - 1) % len(BBOX_COLORS_RGB)]
                display_id_text = f"ID:{script_id_for_color}" + (f" ({prelabel_info['name']})" if prelabel_info['name'] else "")

            all_x = [p['x'] for p in prelabel_info['points']]
            all_y = [p['y'] for p in prelabel_info['points']]
            xmin, ymin, xmax, ymax = int(min(all_x)), int(min(all_y)), int(max(all_x)), int(max(all_y))
            cv2.rectangle(first_frame_display_rgb, (xmin, ymin), (xmax, ymax), color, args.bbox_thickness)
            cv2.putText(first_frame_display_rgb, display_id_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, args.bbox_thickness // 2 + 1)
        
        annotated_first_frame_path = os.path.join(output_annotated_image_dir, first_image_filename_from_dir)
        cv2.imwrite(annotated_first_frame_path, cv2.cvtColor(first_frame_display_rgb, cv2.COLOR_RGB2BGR))

        # --- Subsequent Frame Tracking ---
        tracking_start_time = time.time()
        num_tracked_frames = 0
        print("Starting tracking for subsequent frames...")
        for frame_idx_loop, frame_path in enumerate(image_files[1:], start=1):
            current_image_filename = os.path.basename(frame_path)
            current_frame_bgr = cv2.imread(frame_path)
            
            if current_frame_bgr is None:
                print(f"Warning: Could not read frame {frame_path}. Skipping JSON output for this frame.")
                frame_json_output = {"fileName": current_image_filename, "prelabels": []}
                output_file.write(json.dumps(frame_json_output) + '\n')
                continue
            
            current_frame_rgb = cv2.cvtColor(current_frame_bgr, cv2.COLOR_BGR2RGB)
            frame_to_draw_on_rgb = current_frame_rgb.copy()
            current_frame_output_prelabels = []

            if device == "cuda":
                with torch.cuda.amp.autocast(enabled=False): 
                    pred_mask_current_frame = seg_tracker.track(current_frame_rgb, update_memory=True)
            else: 
                pred_mask_current_frame = seg_tracker.track(current_frame_rgb, update_memory=True)
            
            num_tracked_frames += 1
            
            object_ids_in_mask_values = np.unique(pred_mask_current_frame)
            object_ids_in_mask_values = object_ids_in_mask_values[object_ids_in_mask_values != 0] 

            for script_obj_id in object_ids_in_mask_values:
                binary_mask_for_id = (pred_mask_current_frame == script_obj_id).astype(np.uint8)
                if np.any(binary_mask_for_id):
                    bbox = mask2bbox(binary_mask_for_id)
                    if bbox[0][0] < bbox[1][0] and bbox[0][1] < bbox[1][1]: 
                        obj_name = script_id_to_original_name.get(script_obj_id, f"object_{script_obj_id}")
                        current_frame_output_prelabels.append({
                             "name": obj_name, "uid": str(uuid.uuid4()), "type": "rect", 
                             "select": {}, "points": format_bbox_for_output_json(bbox)
                        })
                        color = BBOX_COLORS_RGB[(int(script_obj_id) - 1) % len(BBOX_COLORS_RGB)]
                        cv2.rectangle(frame_to_draw_on_rgb, (bbox[0][0], bbox[0][1]), (bbox[1][0], bbox[1][1]), color, args.bbox_thickness)
                        label_text = f"ID:{int(script_obj_id)}" + (f" ({obj_name})" if obj_name else "")
                        cv2.putText(frame_to_draw_on_rgb, label_text, (bbox[0][0], bbox[0][1] - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, args.bbox_thickness // 2 + 1)
            
            annotated_frame_path = os.path.join(output_annotated_image_dir, current_image_filename)
            cv2.imwrite(annotated_frame_path, cv2.cvtColor(frame_to_draw_on_rgb, cv2.COLOR_RGB2BGR))
            
            frame_json_output = {"fileName": current_image_filename, "prelabels": current_frame_output_prelabels}
            output_file.write(json.dumps(frame_json_output) + '\n')

            print(f"Processed and saved frame {frame_idx_loop + 1}/{num_total_frames} ({current_image_filename})", end='\r')
            
            if frame_idx_loop % 20 == 0 and device == "cuda": torch.cuda.empty_cache(); gc.collect()
        
        tracking_time = time.time() - tracking_start_time
    # output_file is closed by the 'with open' context manager
    print("\nTracking finished.")

    # --- Performance Metrics & Final Summary ---
    if num_tracked_frames > 0:
        avg_time_per_tracking_frame = tracking_time / num_tracked_frames
        avg_fps_tracking = 1.0 / avg_time_per_tracking_frame
        print(f"--- Performance ---")
        print(f"Model initialization time: {model_init_time:.2f} seconds")
        print(f"First frame processing time: {first_frame_proc_time:.2f} seconds")
        print(f"Number of subsequent frames tracked: {num_tracked_frames}")
        print(f"Total time for tracking subsequent frames: {tracking_time:.2f} seconds")
        print(f"Average time per tracking frame: {avg_time_per_tracking_frame:.3f} seconds")
        print(f"Average FPS for tracking: {avg_fps_tracking:.2f} FPS")
    else:
        print("No subsequent frames were tracked to calculate tracking performance.")

    if device == "cuda": torch.cuda.empty_cache(); gc.collect()
    overall_proc_time = time.time() - overall_start_time
    print(f"Total script execution time: {overall_proc_time:.2f} seconds")
    print(f"Annotated images saved to: {output_annotated_image_dir}")
    print(f"Output tracking data saved to: {output_tracking_data_path}") 
    print("Script finished.")

if __name__ == "__main__":
    main()