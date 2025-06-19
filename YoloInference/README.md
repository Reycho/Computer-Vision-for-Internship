# YOLO Batch Inference with Class Remapping

A high-performance Python script for running batch object detection using Ultralytics YOLO models. It is optimized for processing large image datasets and includes a key feature for remapping model output classes to a custom schema.

![Example Output](20250401104225.046401_RearCam01.jpeg)

## Core Features

-   **Batch Processing**: Efficiently processes multiple directories of images.
-   **Optional Class Remapping**: Toggle a custom class name schema on or off. Defaults to using the model's original names.
-   **Parallelized I/O**: Uses `ProcessPoolExecutor` to format and write results without blocking the main inference loop.
-   **Structured Outputs**: Generates organized JSONL data files and optional annotated images.
-   **Performance Metrics**: Reports processing speed in Frames Per Second (FPS).

## Installation

```bash
pip install ultralytics opencv-python tqdm
```

## Configuration and Usage

1.  **Configure Parameters**: All primary settings are located in the `if __name__ == '__main__':` block. Adjust paths, model settings, and output directories. The new `REMAP_CLASSES` flag controls the name remapping feature.

    ```python
    # --- USER CONFIGURATION ---
    INPUT_FOLDERS = ["/path/to/images_part1", "/path/to/images_part2"]
    MODEL_FILE = "yolov8l.pt"
    TEXT_OUTPUT_DIRECTORY = "text_outputs"
    ANNOTATED_OUTPUT_DIRECTORY = "annotated_images" # Set to None to disable

    # --- CLASS REMAPPING CONTROL ---
    # Set to True to enable custom remapping, False to use original model names.
    REMAP_CLASSES = False # Default is OFF

    # --- INFERENCE PARAMETERS ---
    ANNOTATION_LINE_THICKNESS = 1
    ANNOTATION_FONT_SIZE = 0.5
    CONFIDENCE_THRESHOLD = 0.4
    BATCH_SIZE = 10
    IMAGE_SIZE = 1280
    CLASSES_TO_DETECT = [0, 1, 2, 3, 5, 7] # Set to None for all classes
    ```

2.  **Define Custom Mapping (Optional)**: If you set `REMAP_CLASSES = True`, modify the `class_mapping` dictionary within the `process_single_directory` function to define your custom schema. Otherwise, this step can be ignored.

    ```python
    # If remapping is enabled, define the new names here.
    # Keys are original class IDs, values are the new desired names.
    names_to_use = {
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
-   **`name`**: The class name (remapped or original, based on the `REMAP_CLASSES` setting).
-   **`fileId`**, **`fileName`**, **`uid`**, **`type`**, **`points`**: Standard fields as described previously.

### 2. Annotated Images

If enabled, annotated images are saved to a corresponding subdirectory. Bounding boxes and labels will reflect the chosen name schema (custom or original).
