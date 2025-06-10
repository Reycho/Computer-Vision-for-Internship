# Yolo12 Based Detector

A simple script utilising ultralytics frameowrk to detect and export detected data.

The script is flexible with options ranging from model selection and image size to performance optimizations like half-precision and Test-Time Augmentation (TTA).

# Demo
![Example Output](20250401104225.046401_RearCam01.jpeg)

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
