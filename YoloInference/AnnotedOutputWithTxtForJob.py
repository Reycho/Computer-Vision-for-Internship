import os
import json
import uuid
import time
import ultralytics
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import cv2


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
            "select": {}  
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
        annotated_output_base_dir=None,
        line_thickness=None,  
        font_size=None,       
        conf_thresh=0.25,
        batch_size=8,
        img_size=1280,
        classes_to_detect=None):
    
    # --- Input Validation ---
    print(f"Step 1: Validating input directory: {input_path}")
    if not os.path.isdir(input_path):
        print(f"Error: Input path '{input_path}' is not a valid directory. Skipping.")
        return
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
        return

    # ---FIX: This is the custom mapping we will use AFTER prediction.---
    class_mapping = {0: 'static_object', 1: 'static_object', 2: 'static_object', 3: 'static_object', 4: 'car', 5: 'truck', 6: 'pedestrian', 7: 'two_wheeler'}
    # The original loop to modify model.names is removed as it's ineffective.

    # --- Automatic Filename and Directory Generation ---
    folder_name = os.path.basename(os.path.normpath(input_path))
    os.makedirs(output_dir, exist_ok=True)
    
    final_output_path = os.path.join(output_dir, f"{folder_name}.txt")
    print(f"Output TXT file will be saved to: {final_output_path}")

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
                
                # --- FIX: Pass the custom class_mapping to plot() to get correct labels on images ---
                annotated_image = result.plot(names=class_mapping, **plot_kwargs)
                
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
                    # --- FIX: Look up the new name from our mapping dictionary ---
                    # Use .get() for safety, falling back to original name if ID is not in our map.
                    new_name = class_mapping.get(class_id, model.names[class_id])
                    
                    payload_boxes.append({
                        "coords": coords_list[i],
                        "confidence": conf_list[i],
                        "name": new_name  # Use the newly mapped name
                    })

            payload = {
                "filename": os.path.basename(result.path),
                "img_h": result.orig_shape[0],
                "img_w": result.orig_shape[1],
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
    print("\n--- Directory Processing Complete ---")
    if total_image_count > 0:
        fps = total_image_count / duration if duration > 0 else 0
        print(f"✓ Processed {total_image_count} images in {duration:.2f} seconds ({fps:.2f} FPS).")
        print(f"✓ Results saved to: {final_output_path}")
        if specific_annotated_dir:
            print(f"✓ Annotated images saved to: {specific_annotated_dir}")
    else:
        print("✗ No images were found or processed.")


if __name__ == '__main__':
    # --- USER CONFIGURATION ---
    INPUT_FOLDERS = [
        "/home/ryan/yolov12/dynamic2dtest-100-0617-p1",
        "/home/ryan/yolov12/dynamic2dtest-100-0617-p2"
    ]
    MODEL_FILE = "bettershuffled.pt"
    
    # --- Set the main output directories ---
    TEXT_OUTPUT_DIRECTORY = "text_outputs"
    
    # To disable annotated output, set ANNOTATED_OUTPUT_DIRECTORY = None
    ANNOTATED_OUTPUT_DIRECTORY = None
    
    # --- Annotation Appearance ---
    ANNOTATION_LINE_THICKNESS = 1  
    ANNOTATION_FONT_SIZE = 0.5     

    # --- Main Execution Loop ---
    for folder_path in INPUT_FOLDERS:
        print(f"\n============================================================")
        print(f"          STARTING PROCESSING FOR: {folder_path}          ")
        print(f"============================================================\n")
        try:
            process_single_directory(
                input_path=folder_path,
                model_path=MODEL_FILE,
                output_dir=TEXT_OUTPUT_DIRECTORY,
                annotated_output_base_dir=ANNOTATED_OUTPUT_DIRECTORY,
                line_thickness=ANNOTATION_LINE_THICKNESS, 
                font_size=ANNOTATION_FONT_SIZE,           
                conf_thresh=0.4,
                batch_size=10, 
                img_size=1280,
                classes_to_detect=[0, 1, 2, 3, 5, 7]
            )
        except (FileNotFoundError, IsADirectoryError, TypeError, Exception) as e:
            print(f"\nAn error occurred while processing '{folder_path}': {e}")
            print("Moving to the next folder.")

    print("\n\n--- All specified directories have been processed. ---")