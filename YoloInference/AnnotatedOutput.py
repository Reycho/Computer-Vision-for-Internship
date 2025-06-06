import ultralytics 
import time

def run_yolo_on_images( 
        input_path = "/home/ryan/GithubRoom/c7",
        model_path_or_name="yolo12x.pt",
        output_txt_path="runs/detections_output.txt",
        conf_thresh=0.25, iou_thresh=0.7, img_size=640, use_tta=False,
        max_det=300, use_half=False, classes_to_detect=None, device=None,
        batch_size=1, save_txt=False,
        save_conf=False, save_crop=False):

    model = ultralytics.YOLO(model_path_or_name) 
    print(f"Loaded YOLO model: {model_path_or_name}")
    
    predict_args = {
        "source": input_path, "stream": True, "save": True, "verbose": False,
        "conf": conf_thresh, "iou": iou_thresh, "imgsz": img_size, "augment": use_tta,
        "max_det": max_det, "half": use_half, "classes": classes_to_detect, 
        "device": device, "batch": batch_size,
        "save_txt": save_txt, "save_conf": save_conf, "save_crop": save_crop,
        "save" : True
    }

    print(f"Processing images from: {input_path} using {model_path_or_name}")

    loop_start_time = time.perf_counter()

    results_generator = model.predict(**predict_args)

    processed_count = 0

    for result in results_generator:
        processed_count += 1

        if processed_count % 100 == 0:
            print(f"Processed {processed_count} images...")
    
    duration = time.perf_counter() - loop_start_time

    if processed_count > 0:
        avg_time = duration / processed_count
        fps = 1.0 / avg_time if avg_time > 0 else 0
        print(f"Finished. Processed {processed_count} images in {duration:.2f}s.")
        print(f"  Average YOLO processing time per image: {avg_time:.3f}s ({fps:.2f} FPS).")
    else:
        print(f"No images were processed from '{input_path}'. Total loop time: {duration:.2f}s.")

if __name__ == '__main__':        
        overall_start_time = time.perf_counter()
        
        run_yolo_on_images(
            conf_thresh=0.15,       
            iou_thresh=0.7,         
            img_size=(1280, 736), # Resize images to this size (WxH) for inference   
            use_tta=True, # Use Test-Time Augmentation         
            max_det=500, # Maximum number of detections per image       
            use_half=False, # Use FP16 precision     
            classes_to_detect=[0, 2, 3, 5, 7], # Specify class IDs to detect, None for all 
            device=None,    # Auto-select (CPU/GPU), or '0' for GPU 0, 'cpu'
            batch_size=8,   # Batch size for inference (effectiveness varies with source type and stream=True)
        )
        
        overall_duration = time.perf_counter() - overall_start_time
        print(f"Total script execution completed in {overall_duration:.2f} seconds.")   