# YOLO Batch Inference with Class Remapping

A high-performance Python script for running batch object detection using Ultralytics YOLO models. It is optimized for processing large image datasets and includes a key feature for remapping model output classes to a custom schema.

![Example Output](20250401104225.046401_RearCam01.jpeg)

## Core Features

-   **Batch Processing**: Efficiently processes multiple directories of images.
-   **Custom Class Remapping**: Maps original model class IDs to new, user-defined labels in both data and visual outputs.
-   **Parallelized I/O**: Uses `ProcessPoolExecutor` to format and write results without blocking the main inference loop.
-   **Structured Outputs**: Generates organized JSONL data files and optional annotated images.
-   **Performance Metrics**: Reports processing speed in Frames Per Second (FPS).

## Installation

```bash
pip install ultralytics opencv-python tqdm
```

## Configuration and Usage

1.  **Configure Parameters**: All primary settings are located in the `if __name__ == '__main__':` block. Adjust paths, model settings, and output directories as needed.

    ```python
    # --- USER CONFIGURATION ---
    INPUT_FOLDERS = ["/path/to/images_part1", "/path/to/images_part2"]
    MODEL_FILE = "yolov8l.pt"
    TEXT_OUTPUT_DIRECTORY = "text_outputs"
    ANNOTATED_OUTPUT_DIRECTORY = "annotated_images" # Set to None to disable

    # --- INFERENCE PARAMETERS ---
    ANNOTATION_LINE_THICKNESS = 1
    ANNOTATION_FONT_SIZE = 0.5
    CONFIDENCE_THRESHOLD = 0.4
    BATCH_SIZE = 10
    IMAGE_SIZE = 1280
    CLASSES_TO_DETECT = [0, 1, 2, 3, 5, 7] # Set to None for all classes
    ```

2.  **Define Class Mapping**: The class remapping logic is defined within the `process_single_directory` function. Modify the `class_mapping` dictionary to fit your schema.

    ```python
    # Keys are original class IDs, values are the new desired names.
    class_mapping = {
        0: 'static_object', 1: 'static_object', 2: 'static_object',
        3: 'static_object', 4: 'car', 5: 'truck', 6: 'pedestrian',
        7: 'two_wheeler'
    }
    ```

3.  **Execute**: Run the script from the terminal.

    ```bash
    python your_script_name.py
    ```

## Output Format

### 1. JSONL Data File (`.txt`)

A `.txt` file is generated for each input directory, containing one JSON object per line.

**Example JSON Line:**
```json
{"fileName": "image_01.jpg", "fileId": "a1b2c3d4-e5f6-a1b2-c3d4-e5f6a1b2c3d4", "prelabels": [{"name": "car", "uid": "f1e2d3c4-b5a6-f1e2-d3c4-b5a6f1e2d3c4", "type": "rect", "points": [{"x": 747.0, "y": 471.0}, {"x": 1133.0, "y": 471.0}, {"x": 1133.0, "y": 709.0}, {"x": 747.0, "y": 709.0}], "select": {}}]}
```
-   **`fileName`**: Original image filename.
-   **`fileId`**: Unique UUID for the processed image.
-   **`prelabels`**: List of detection objects.
    -   **`name`**: The remapped class name.
    -   **`uid`**: A unique UUID for the detection.
    -   **`type`**: Annotation shape (`rect`).
    -   **`points`**: Four corner points of the bounding box.

### 2. Annotated Images

If enabled, annotated images are saved to a corresponding subdirectory, preserving the input folder structure. Bounding boxes and labels reflect the custom `class_mapping`.
