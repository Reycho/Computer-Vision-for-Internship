import ultralytics
import time
import json
import uuid
import os
import cv2

def run_yolo_on_images(
        input_path,
        model_path_or_name="yolo12x.pt",
        output_txt_path="runs/detections_output.txt",
        conf_thresh=0.25, iou_thresh=0.7, img_size=640, use_tta=False,
        max_det=300, use_half=False, classes_to_detect=None, device=None,
        batch_size=1, save_txt=False,
        save_conf=False, save_crop=False,
        custom_annotated_images_output_dir=None, # MODIFIED: Default to None
        custom_line_width=None,
        custom_font_size=None):

    model = ultralytics.YOLO(model_path_or_name)
    print(f"Loaded YOLO model: {model_path_or_name}")

    # Determine the output directory for annotated images
    final_annotated_images_output_dir: str
    if custom_annotated_images_output_dir is None:
        # Default behavior: create a directory based on input_path name
        # Normalize path to remove trailing slashes, then get base name
        base_name = os.path.basename(os.path.normpath(input_path))
        # Remove extension if present (e.g., for video files or single image processing)
        name_part, _ = os.path.splitext(base_name)
        derived_output_dir_name = f"{name_part}_processed"
        # Place it under runs/detect/ similar to default Ultralytics behavior for 'predict' runs
        final_annotated_images_output_dir = os.path.join("runs", "detect", derived_output_dir_name)
    else:
        # User provided a specific path
        final_annotated_images_output_dir = custom_annotated_images_output_dir

    if not os.path.exists(final_annotated_images_output_dir):
        os.makedirs(final_annotated_images_output_dir, exist_ok=True)

    print(f"Processing images from: {input_path} using {model_path_or_name}")
    print(f"Custom annotated images will be saved to: {final_annotated_images_output_dir}")

    predict_args = {
        "source": input_path, "stream": True, "verbose": False,
        "conf": conf_thresh, "iou": iou_thresh, "imgsz": img_size, "augment": use_tta,
        "max_det": max_det, "half": use_half, "classes": classes_to_detect,
        "device": device, "batch": batch_size,
        "save_txt": save_txt, "save_conf": save_conf, "save_crop": save_crop,
        "save": False # We are handling saving of annotated images manually
    }

    loop_start_time = time.perf_counter()

    # Ensure the directory for the output TXT file exists
    output_txt_dir = os.path.dirname(output_txt_path)
    if output_txt_dir and not os.path.exists(output_txt_dir): # Check if output_txt_dir is not empty string
        os.makedirs(output_txt_dir, exist_ok=True)

    with open(output_txt_path, 'w') as outfile: # Open the .txt file for writing

        results_generator = model.predict(**predict_args)
        processed_count = 0

        for result in results_generator:
            processed_count += 1

            if processed_count % 10 == 0:
                current_time = time.perf_counter()
                elapsed_time = current_time - loop_start_time
                avg_time_so_far = elapsed_time / processed_count if processed_count > 0 else 0
                print(f"Processed {processed_count} images... (Avg time per image so far: {avg_time_so_far:.3f}s)")

            current_image_path = result.path
            file_name = os.path.basename(current_image_path) if current_image_path else f"image_{processed_count}.jpg"
            file_id = str(uuid.uuid4()) # Unique ID for the image processing instance

            plot_kwargs = {}
            if custom_line_width is not None:
                plot_kwargs['line_width'] = custom_line_width
            if custom_font_size is not None:
                plot_kwargs['font_size'] = custom_font_size

            annotated_image_bgr = result.plot(**plot_kwargs)
            output_image_file_path = os.path.join(final_annotated_images_output_dir, file_name)
            cv2.imwrite(output_image_file_path, annotated_image_bgr)

            prelabels_list = []
            if result.boxes: # Check if any detections (boxes) exist for this image
                for box in result.boxes: # Iterate over each detected box
                    class_id = int(box.cls.item()) # Get class ID
                    class_name = model.names[class_id] # Get human-readable class name
                    detection_uid = str(uuid.uuid4()) # Unique ID for this specific detection

                    coords = box.xyxy[0].tolist()
                    xmin, ymin, xmax, ymax = float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])

                    points = [
                        {"x": xmin, "y": ymin}, {"x": xmax, "y": ymin},
                        {"x": xmax, "y": ymax}, {"x": xmin, "y": ymax}
                    ]

                    prelabel_entry = {
                        "name": class_name, "uid": detection_uid,
                        "type": "rect", "select": {}, "points": points
                    }
                    prelabels_list.append(prelabel_entry)

            image_data_to_write = {
                "fileName": file_name,
                "fileId": file_id,
                "prelabels": prelabels_list
            }
            outfile.write(json.dumps(image_data_to_write) + '\n')

    duration = time.perf_counter() - loop_start_time

    if processed_count > 0:
        avg_time = duration / processed_count
        fps = 1.0 / avg_time if avg_time > 0 else 0
        print(f"Finished. Processed {processed_count} images in {duration:.2f}s.")
        print(f"  Average YOLO processing time per image: {avg_time:.3f}s ({fps:.2f} FPS).")
        print(f"Custom detections TXT file saved to: {output_txt_path}")
    else:
        print(f"No images were processed from '{input_path}'.")

if __name__ == '__main__':
    overall_start_time = time.perf_counter()

    run_yolo_on_images(
        #Edit below
        model_path_or_name="yolo12x", # As per original, ensure this model exists or is downloadable
        input_path="20250604-2Ddynamic-test-160", # This will be used for <input_folder_name>_processed
        img_size=(2176,1440),
        custom_line_width=2,
        custom_font_size=1.5, # Ultralytics plot() font_size is a relative factor
        batch_size=3,
        conf_thresh=0.35,      #confidence threshold
        iou_thresh=0.7,        #Intersection over Union threshold
        use_tta=True, # Use Test-Time Augmentation
        max_det=300, # Maximum number of detections per image
        use_half=True, # Use FP16 precision (ensure GPU support)
        classes_to_detect=[0, 1, 2, 3, 5, 7], # Specify class IDs to detect, None for all
        device=None, # None for auto-detection (CPU or GPU)
        save_txt=False,
        save_conf=False,
        save_crop=False,
        # MODIFIED: custom_annotated_images_output_dir is removed from here to use the new default logic
        # custom_annotated_images_output_dir="runs/detect/annotated_images", # This line is effectively removed
    )

    overall_duration = time.perf_counter() - overall_start_time
    print(f"Total script execution completed in {overall_duration:.2f} seconds.")