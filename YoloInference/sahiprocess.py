import os
import json
import uuid
import time
import torch
from sahi.predict import predict
import traceback

def format_data_for_jsonl(payload):
    """Formats a payload dictionary into a JSONL string."""
    prelabels_list = []
    for box_data in payload['boxes']:
        xmin, ymin, xmax, ymax = box_data['coords']
        prelabel_entry = {
            "name": box_data['name'], "uid": str(uuid.uuid4()), "type": "rect",
            "points": [{"x": xmin, "y": ymin}, {"x": xmax, "y": ymin}, {"x": xmax, "y": ymax}, {"x": xmin, "y": ymax}],
            "select": {}
        }
        prelabels_list.append(prelabel_entry)
    image_data_to_write = {"fileName": payload['filename'], "fileId": str(uuid.uuid4()), "prelabels": prelabels_list}
    return json.dumps(image_data_to_write) + '\n'

def process_source_with_sahi_pipeline(
        source_path,
        model_path,
        text_output_dir,
        annotated_output_base_dir,
        slice_height,
        slice_width,
        overlap_height_ratio,
        overlap_width_ratio,
        conf_thresh=0.25,
        class_remap_dict=None):

    print(f"--- Processing {source_path} using SAHI's high-level pipeline ---")
    start_time = time.time()

    output_base_name = os.path.basename(os.path.normpath(source_path))
    run_name = f"{output_base_name}_sahi_output"

    # For image directories, we want visuals. For videos, we do not, as SAHI
    # doesn't create annotated videos, only frames.
    is_video = source_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    
    # 1. LET SAHI DO ALL THE HARD WORK WITH THE CORRECT PARAMETERS
    prediction_results = predict(
        model_type="ultralytics",
        model_path=model_path,
        model_confidence_threshold=conf_thresh,
        model_device="cuda:0" if torch.cuda.is_available() else "cpu",
        source=source_path,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
        project=annotated_output_base_dir,
        name=run_name,
        novisual=is_video,      # THIS IS THE CORRECT PARAMETER. True for video, False for images.
        return_dict=True,       # This is the key to getting results back for our JSONL.
        verbose=1               # Controls console output level. 1 is reasonable.
    )
    
    results_list = prediction_results.get('results')
    if not results_list:
        print("SAHI pipeline returned no predictions for this source.")
        return

    print(f"\nInference complete. Found predictions for {len(results_list)} images. Formatting to JSONL...")
    jsonl_lines = []
    
    # 2. PROCESS THE RESULTS TO CREATE OUR JSONL
    for result in results_list:
        output_filename = os.path.basename(result.image_path)
        payload_boxes = []
        for pred in result.object_prediction_list:
            if class_remap_dict and pred.category.id in class_remap_dict:
                name = class_remap_dict[pred.category.id]
            else:
                name = pred.category.name
            
            coords = pred.bbox.to_xyxy()
            payload_boxes.append({"coords": coords, "name": name})
            
        payload = {"filename": output_filename, "boxes": payload_boxes}
        jsonl_lines.append(format_data_for_jsonl(payload))

    # 3. SAVE OUR CUSTOM JSONL FILE
    if text_output_dir and jsonl_lines:
        os.makedirs(text_output_dir, exist_ok=True)
        final_output_path = os.path.join(text_output_dir, f"{output_base_name}.txt")
        
        with open(final_output_path, 'w') as outfile:
            outfile.writelines(jsonl_lines)
            
        print(f"✓ Successfully saved custom JSONL to: {final_output_path}")

    duration = time.time() - start_time
    print(f"✓ Finished processing source in {duration:.2f} seconds.")


# ==============================================================================
# SCRIPT CONTROL PANEL
# ==============================================================================
if __name__ == '__main__':
    INPUT_SOURCES = [
        "/home/ryan/yolov12/dynamic2dtest-50-0617",
        "/home/ryan/yolov12/dynamic2dtest-100-0617-p1",
        "/home/ryan/yolov12/dynamic2dtest-100-0617-p2"
    ]
    MODEL_FILE = "/home/ryan/yolov12/staticobjectX.pt"
    
    TEXT_OUTPUT_DIRECTORY = "/home/ryan/Desktop/Tobecombined/static_sahi_jsonl" 
    ANNOTATED_OUTPUT_DIRECTORY = "/home/ryan/Desktop/Tobecombined/sahi_runs" # SAHI will save annotated images here
    
    CLASS_REMAPPING_DICT = None
    
    CONFIDENCE_THRESHOLD = 0.2
    SLICE_HEIGHT = 1280
    SLICE_WIDTH = 1280
    OVERLAP_HEIGHT_RATIO = 0.2
    OVERLAP_WIDTH_RATIO = 0.2

    # --- Main Execution Loop ---
    for source_path in INPUT_SOURCES:
        print(f"\n{'='*80}\nSTARTING PROCESSING FOR: {source_path}\n{'='*80}\n")
        try:
            process_source_with_sahi_pipeline(
                source_path=source_path,
                model_path=MODEL_FILE,
                text_output_dir=TEXT_OUTPUT_DIRECTORY,
                annotated_output_base_dir=ANNOTATED_OUTPUT_DIRECTORY,
                slice_height=SLICE_HEIGHT,
                slice_width=SLICE_WIDTH,
                overlap_height_ratio=OVERLAP_HEIGHT_RATIO,
                overlap_width_ratio=OVERLAP_WIDTH_RATIO,
                conf_thresh=CONFIDENCE_THRESHOLD,
                class_remap_dict=CLASS_REMAPPING_DICT,
            )
        except Exception as e:
            print(f"\nAn UNHANDLED error occurred while processing '{source_path}':")
            traceback.print_exc()

    print("\n\n--- All specified sources have been processed. ---")