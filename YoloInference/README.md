# Advanced YOLO12 Inference Pipeline

This repository contains a powerful and highly customizable Python script for running object detection using the Ultralytics YOLOv8 framework. It is designed to process batches of images, save detailed detection results in a specific JSON-line format, and generate annotated images for visual inspection.

The script is built to be flexible, allowing for fine-tuning of nearly every aspect of the inference process, from model selection and image size to performance optimizations like half-precision and Test-Time Augmentation (TTA).

# Demo
![Example Output](20250401104225.046401_RearCam01.jpeg)

## Features

-   **Flexible Model Support:** Use any official Ultralytics YOLO model (e.g., `yolov12s.pt`, `yolov12x.pt`) or your own custom-trained models.
-   **Batch Processing:** Efficiently process entire directories of images.
-   **Custom Detection Output:** Generates a `.txt` file with one JSON object per image, containing detailed bounding box coordinates and class information.
-   **Visual Annotation:** Automatically saves copies of the input images with detection bounding boxes, class names, and confidence scores drawn on them.
-   **Highly Configurable Inference:**
    -   Custom input image size (`img_size`).
    -   Confidence (`conf_thresh`) and IoU (`iou_thresh`) thresholds.
    -   Device selection (`device`): CPU, CUDA GPU, or Apple MPS.
    -   Adjustable `batch_size` for optimal GPU memory usage.
    -   Enable/disable Test-Time Augmentation (`use_tta`) for an accuracy boost.
    -   Enable/disable half-precision (FP16) inference (`use_half`) for a significant speedup on compatible GPUs.
    -   Filter detections by specific class IDs (`classes_to_detect`).
-   **Customizable Annotations:** Control the line width and font size of the saved bounding box annotations.
-   **Performance Tracking:** Reports total processing time and average Frames Per Second (FPS).

## Requirements

To run this script, you will need Python 3.8+ and the following libraries.

-   `ultralytics`
-   `opencv-python`
-   `torch`
-   `torchvision`

For GPU acceleration, ensure you have a compatible NVIDIA GPU with CUDA and cuDNN installed. The `ultralytics` package will often handle the correct PyTorch installation.

You can install all required packages using pip:
```bash
pip install ultralytics opencv-python
```

## Usage

1.  Clone this repository or download the `run_yolo.py` script (assuming you name your file that).
2.  Place your images in a directory (e.g., `path/to/your/images`).
3.  Place your YOLO model file (e.g., `yolo12x.pt`) where the script can access it, or use an official model name which will be downloaded automatically.
4.  Configure the parameters directly within the `if __name__ == '__main__':` block at the bottom of the script.


## Output Format

The script produces two primary outputs:

### 1. Annotated Images

Annotated images are saved in the directory specified by `custom_annotated_images_output_dir`. Each image will have bounding boxes, class names, and confidence scores drawn on it.

### 2. Detections TXT File

A text file (specified by `output_txt_path`) is created, containing one JSON object per line for each processed image. This format is useful for easy parsing and integration with other tools.

**Example JSON Line:**
```json
{"fileName": "image_01.jpg", "fileId": "a1b2c3d4-e5f6-a1b2-c3d4-e5f6a1b2c3d4", "prelabels": [{"name": "car", "uid": "f1e2d3c4-b5a6-f1e2-d3c4-b5a6f1e2d3c4", "type": "rect", "select": {}, "points": [{"x": 747.0, "y": 471.0}, {"x": 1133.0, "y": 471.0}, {"x": 1133.0, "y": 709.0}, {"x": 747.0, "y": 709.0}]}, {"name": "person", "uid": "9a8b7c6d-5e4f-9a8b-7c6d-5e4f9a8b7c6d", "type": "rect", "select": {}, "points": [{"x": 415.0, "y": 520.0}, {"x": 510.0, "y": 520.0}, {"x": 510.0, "y": 815.0}, {"x": 415.0, "y": 815.0}]}]}
```
-   `fileName`: The original basename of the image file.
-   `fileId`: A unique UUID generated for this processing instance of the image.
-   `prelabels`: A list of all detections found in the image.
    -   `name`: The human-readable class name.
    -   `uid`: A unique UUID for this specific detection.
    -   `type`: The shape of the annotation (always `rect` in this script).
    -   `points`: A list of the four corner points of the bounding box, starting from the top-left and moving clockwise (based on image coordinates `[xmin, ymin, xmax, ymax]`).
