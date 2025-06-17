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
    Handles CPU-bound tasks. Receives a simple dictionary ('payload')
    containing only basic data types. (This function is unchanged).
    """
    img_h = payload['img_h']
    img_w = payload['img_w']
    
    prelabels_list = []
    for box_data in payload['boxes']:
        xmin, ymin, xmax, ymax = box_data['coords']
                
        prelabel_entry = {
            "name": box_data['name'],
            "confidence": box_data['confidence'],
            "detection_uid": str(uuid.uuid4()),
            "type": "rect",
            "points": [
                {"x": xmin, "y": ymin}, {"x": xmax, "y": ymin},
                {"x": xmax, "y": ymax}, {"x": xmin, "y": ymax}
            ]
        }
        prelabels_list.append(prelabel_entry)

    image_data_to_write = {
        "fileName": payload['filename'],
        "image_uid": str(uuid.uuid4()),
        "dimensions": {"width": img_w, "height": img_h},
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
        img_size=1280):
    
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

    # --- Automatic Filename and Directory Generation ---
    # Get the base name of the input folder (e.g., "dynamicpt2")
    folder_name = os.path.basename(os.path.normpath(input_path))
    
    # Create the full path for the output JSONL file
    os.makedirs(output_dir, exist_ok=True)
    final_output_path = os.path.join(output_dir, f"{folder_name}.jsonl")
    print(f"Output JSONL file will be saved to: {final_output_path}")

    # Create a specific subdirectory for annotated images for this run
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
            batch=batch_size, imgsz=img_size, verbose=False
        )
        
        # Use a temporary list to show a proper progress bar for inference
        results_list = list(tqdm(results_generator, desc=f"Inferring on {folder_name}"))
        total_image_count = len(results_list)

        for result in results_list:
            if specific_annotated_dir:
                plot_kwargs = {}
                if line_thickness is not None:
                    plot_kwargs['line_width'] = line_thickness
                if font_size is not None:
                    plot_kwargs['font_size'] = font_size
                annotated_image = result.plot(**plot_kwargs)
                image_filename = os.path.basename(result.path)
                output_image_path = os.path.join(specific_annotated_dir, image_filename)
                cv2.imwrite(output_image_path, annotated_image)
            
            payload_boxes = []
            if result.boxes:
                coords_list = result.boxes.xyxy.cpu().tolist()
                conf_list = result.boxes.conf.cpu().tolist()
                cls_list = result.boxes.cls.cpu().tolist()

                for i in range(len(coords_list)):
                    payload_boxes.append({
                        "coords": coords_list[i],
                        "confidence": conf_list[i],
                        "name": model.names[int(cls_list[i])]
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
        for future in tqdm(as_completed(futures), total=len(futures), desc="Writing JSONL"):
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
        "/home/ryan/Desktop/pt2",
        "/home/ryan/Desktop/part1",
        "/home/ryan/Desktop/dynamic2dtest-50-0617" # You can add more folders here
    ]
    MODEL_FILE = "bettershuffled.pt"
    
    # --- Set the main output directories ---
    # JSONL files will be created inside this folder
    JSONL_OUTPUT_DIRECTORY = "jsonl_outputs"
    # Annotated images will be saved in subfolders inside this one
    ANNOTATED_OUTPUT_DIRECTORY = "annotated_results"

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
                output_dir=JSONL_OUTPUT_DIRECTORY,
                annotated_output_base_dir=None,
                line_thickness=ANNOTATION_LINE_THICKNESS, 
                font_size=ANNOTATION_FONT_SIZE,           
                conf_thresh=0.4,
                batch_size=10, 
                img_size=1280
            )
        except (FileNotFoundError, IsADirectoryError, TypeError, Exception) as e:
            print(f"\nAn error occurred while processing '{folder_path}': {e}")
            print("Moving to the next folder.")

    print("\n\n--- All specified directories have been processed. ---")
