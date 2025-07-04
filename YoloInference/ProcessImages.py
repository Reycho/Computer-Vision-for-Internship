# ==============================================================================
#                      Combined Inference and Annotation Script
# ==============================================================================
# This script performs a two-step process:
# 1.  YOLO Inference: It runs a YOLOv8 model on images in specified directories,
#     detecting objects and saving the results (bounding boxes, class names)
#     into a JSONL text file (one JSON object per image).
# 2.  Annotation Update: It then reads the newly generated JSONL file, and for
#     each detected object, it fills in a 'select' dictionary with predefined
#     default attributes based on the object's class name.
#
# The final output is a text file ready for import into an annotation tool,
# with default values pre-filled to speed up the labeling process.
# ==============================================================================

import json
import os
import uuid
import time
import ultralytics
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import cv2

# ==============================================================================
#                   PART 1: Annotation Update Logic
# ==============================================================================

# Default attributes extracted from the UI, keyed by object name.
# The values are taken from the options marked with a blue dot in the UI.
DEFAULT_ATTRIBUTES = {
    "car": {
        "door_opened": "0",
        "lane_height": "1",
        "model_car": "0"
    },
    "bus": {
        "door_opened": "0",
        "lane_height": "1",
        "model_car": "0"
    },
    "truck": {
        "door_opened": "0",
        "business_type": "other",
        "lane_height": "1"
    },
    "pedestrian": {
        "sub_category": "unknown",
        "lane_height": "1",
        "is_pushed": "unknown",
        "pedestrian_on_cart": "unknown",
        "scooter_status": "unknown",
        "is_simulated_vru": "0"
    },
    "two_wheeler": {
        "lane_height": "1",
        "is_pushed": "0",
        "scooter_status": "unknown",
        "is_simulated_vru": "0"
    }
    # 'static_object' is not included as no default values are shown for it in the UI.
}

def update_annotations(input_file_path, output_file_path):
    """
    Reads a file with JSON annotations line by line, fills in the default
    values for the 'select' field based on the object's category, and
    writes the updated data to a new file.

    Args:
        input_file_path (str): The path to the input annotation file.
        output_file_path (str): The path where the updated file will be saved.
    """
    if not os.path.exists(input_file_path):
        print(f"Error: Input file '{input_file_path}' not found for updating.")
        return

    updated_lines = []
    with open(input_file_path, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                if 'prelabels' in data and data['prelabels'] is not None:
                    for prelabel in data['prelabels']:
                        obj_name = prelabel.get('name')
                        if obj_name in DEFAULT_ATTRIBUTES:
                            # Assign the default attributes to the 'select' field
                            prelabel['select'] = DEFAULT_ATTRIBUTES[obj_name].copy()

                # Convert back to a compact JSON string, preserving any non-ASCII chars
                updated_line = json.dumps(data, separators=(',', ':'), ensure_ascii=False)
                updated_lines.append(updated_line)

            except json.JSONDecodeError:
                # If a line is not valid JSON, keep it as is
                updated_lines.append(line)

    # Write the updated content to the new file
    with open(output_file_path, 'w', encoding='utf-8') as f_out:
        f_out.write('\n'.join(updated_lines))
        f_out.write('\n')  # Add a trailing newline for consistency

    print(f"Annotation update complete. Final file saved as '{output_file_path}'")


# ==============================================================================
#                   PART 2: YOLOv8 Inference Logic
# ==============================================================================

def format_data_for_jsonl(payload):
    """
    Handles CPU-bound tasks. Formats data into a JSON string with a specific
    schema to be written as a line in a TXT file.
    """
    prelabels_list = []
    for box_data in payload['boxes']:
        xmin, ymin, xmax, ymax = box_data['coords']
        prelabel_entry = {
            "name": box_data['name'],
            "uid": str(uuid.uuid4()),
            "type": "rect",
            "points": [
                {"x": xmin, "y": ymin}, {"x": xmax, "y": ymin},
                {"x": xmax, "y": ymax}, {"x": xmin, "y": ymax}
            ],
            "select": {}  # Initially empty, to be filled by update_annotations
        }
        prelabels_list.append(prelabel_entry)

    image_data_to_write = {
        "fileName": payload['filename'],
        "fileId": str(uuid.uuid4()),
        "prelabels": prelabels_list
    }

    return json.dumps(image_data_to_write) + '\n'


def process_single_directory(
        input_path,
        model_path,
        output_dir,
        remap_classes=False,
        annotated_output_base_dir=None,
        line_thickness=None,
        font_size=None,
        conf_thresh=0.25,
        batch_size=8,
        img_size=1280,
        classes_to_detect=None):
    """
    Runs YOLOv8 inference on a directory and produces a raw JSONL output file.
    Returns the path to the generated file on success, or None on failure.
    """
    # --- Input Validation ---
    print(f"Step 1: Validating input directory: {input_path}")
    if not os.path.isdir(input_path):
        print(f"Error: Input path '{input_path}' is not a valid directory. Skipping.")
        return None
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Error: Model file not found at '{model_path}'.")
    print("Validation successful.")

    # --- Setup Model and Output Paths ---
    print("\nStep 2: Initializing model and preparing output paths...")
    try:
        model = ultralytics.YOLO(model_path)
        print(f"Successfully loaded model from '{model_path}'.")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return None

    # --- Conditional Class Name Setup ---
    if remap_classes:
        print("Class remapping is ENABLED.")
        names_to_use = {0: 'static_object', 1: 'static_object', 2: 'static_object', 3: 'static_object', 4: 'car', 5: 'truck', 6: 'pedestrian', 7: 'two_wheeler'}
    else:
        print("Class remapping is DISABLED. Using original model class names.")
        names_to_use = model.names

    # --- Automatic Filename and Directory Generation ---
    folder_name = os.path.basename(os.path.normpath(input_path))
    os.makedirs(output_dir, exist_ok=True)
    final_output_path = os.path.join(output_dir, f"{folder_name}.txt")
    print(f"Raw inference output will be saved to: {final_output_path}")

    specific_annotated_dir = None
    if annotated_output_base_dir:
        specific_annotated_dir = os.path.join(annotated_output_base_dir, folder_name)
        os.makedirs(specific_annotated_dir, exist_ok=True)
        print(f"Annotated images will be saved to: '{specific_annotated_dir}'")

    # --- Parallel Processing ---
    print("\nStep 3: Starting parallel image processing...")
    start_time = time.perf_counter()
    mp_context = mp.get_context("spawn")

    with ProcessPoolExecutor(mp_context=mp_context) as executor, open(final_output_path, 'w') as outfile:
        futures = []
        results_generator = model.predict(
            source=input_path, stream=True, conf=conf_thresh,
            batch=batch_size, imgsz=img_size, verbose=False, classes=classes_to_detect
        )
        results_list = list(tqdm(results_generator, desc=f"Inferring on {folder_name}"))
        total_image_count = len(results_list)

        for result in results_list:
            if specific_annotated_dir:
                plot_kwargs = {}
                if line_thickness is not None:
                    plot_kwargs['line_width'] = line_thickness
                if font_size is not None:
                    plot_kwargs['font_size'] = font_size
                annotated_image = result.plot(names=names_to_use, **plot_kwargs)
                image_filename = os.path.basename(result.path)
                output_image_path = os.path.join(specific_annotated_dir, image_filename)
                cv2.imwrite(output_image_path, annotated_image)

            payload_boxes = []
            if result.boxes:
                coords_list = result.boxes.xyxy.cpu().tolist()
                conf_list = result.boxes.conf.cpu().tolist()
                cls_list = result.boxes.cls.cpu().tolist()
                for i in range(len(coords_list)):
                    class_id = int(cls_list[i])
                    new_name = names_to_use.get(class_id, f"unknown_class_{class_id}")
                    payload_boxes.append({
                        "coords": coords_list[i],
                        "confidence": conf_list[i],
                        "name": new_name
                    })

            payload = {
                "filename": os.path.basename(result.path),
                "boxes": payload_boxes,
            }
            future = executor.submit(format_data_for_jsonl, payload)
            futures.append(future)

        print("\nInference complete. Formatting and writing to file...")
        for future in tqdm(as_completed(futures), total=len(futures), desc="Writing TXT file"):
            json_line = future.result()
            outfile.write(json_line)

    # --- Final Report for this Directory ---
    duration = time.perf_counter() - start_time
    print("\n--- Inference Processing Complete ---")
    if total_image_count > 0:
        fps = total_image_count / duration if duration > 0 else 0
        print(f"✓ Processed {total_image_count} images in {duration:.2f} seconds ({fps:.2f} FPS).")
        print(f"✓ Raw results saved to: {final_output_path}")
        if specific_annotated_dir:
            print(f"✓ Annotated images saved to: {specific_annotated_dir}")
    else:
        print("✗ No images were found or processed.")
    
    return final_output_path # Return the path for the next step


# ==============================================================================
#                             MAIN EXECUTION
# ==============================================================================

if __name__ == '__main__':
    # --- USER CONFIGURATION ---
    INPUT_FOLDERS = [
        "/home/user/data/project_batch_1",
        "/home/user/data/project_batch_2"
    ]
    MODEL_FILE = "yolov8l_custom.pt"

    # --- Set the main output directories ---
    # This is where the intermediate and final text files will be saved.
    TEXT_OUTPUT_DIRECTORY = "text_outputs"

    # To disable saving annotated images, set ANNOTATED_OUTPUT_DIRECTORY = None
    ANNOTATED_OUTPUT_DIRECTORY = "annotated_images"

    # --- Class Remapping Control ---
    # Set to True to enable custom class remapping defined in process_single_directory.
    # Set to False to use the model's original class names.
    REMAP_CLASSES = False  # Default is OFF

    # --- Annotation Appearance (for saved images) ---
    ANNOTATION_LINE_THICKNESS = 1
    ANNOTATION_FONT_SIZE = 0.5

    # --- Main Execution Loop ---
    for folder_path in INPUT_FOLDERS:
        print(f"\n============================================================")
        print(f"      STARTING FULL WORKFLOW FOR: {folder_path}      ")
        print(f"============================================================\n")
        try:
            # STEP A: Run YOLO inference to generate the raw annotation file
            print(f"--- [WORKFLOW STEP 1/2] Running YOLO Inference ---")
            raw_output_file = process_single_directory(
                input_path=folder_path,
                model_path=MODEL_FILE,
                output_dir=TEXT_OUTPUT_DIRECTORY,
                remap_classes=REMAP_CLASSES,
                annotated_output_base_dir=ANNOTATED_OUTPUT_DIRECTORY,
                line_thickness=ANNOTATION_LINE_THICKNESS,
                font_size=ANNOTATION_FONT_SIZE,
                conf_thresh=0.4,
                batch_size=10,
                img_size=1280,
                classes_to_detect=None  # Set to None to detect all classes
            )

            # STEP B: If inference was successful, update the generated file
            if raw_output_file:
                print(f"\n--- [WORKFLOW STEP 2/2] Filling Default Annotation Attributes ---")
                # Define the final output path for the updated file
                # Example: 'text_outputs/folder.txt' -> 'text_outputs/folder_filled.txt'
                final_filled_file = raw_output_file.replace('.txt', '_filled.txt')
                
                # Run the update function
                update_annotations(
                    input_file_path=raw_output_file,
                    output_file_path=final_filled_file
                )
            else:
                print(f"Skipping annotation update for '{folder_path}' due to an issue in the inference step.")

        except (FileNotFoundError, IsADirectoryError, TypeError, Exception) as e:
            print(f"\nAn unrecoverable error occurred while processing '{folder_path}': {e}")
            print("Moving to the next folder.")

    print("\n\n--- All specified directories have been processed. ---")
