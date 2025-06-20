# SAM2 Multi-Object Video Tracker

This project provides a command-line script to track multiple objects in an image sequence using the **Segment Anything 2 (SAM2)** model. It initializes objects on the first frame using bounding box prompts from a JSON file and leverages SAM2's built-in video tracking capabilities to propagate the masks through the entire sequence.

## Demo of Results

The tracker successfully identifies and follows multiple objects throughout the video sequence.

**Final Tracking Video Demo:**
![Tracking Demo GIF](output.gif)

## Installation

1.  **Follow installation steps according to the official sam2 repo.**
    https://github.com/facebookresearch/sam2

2.  **If you encounter an error related to CUDA compilation.**
    You can force-compile the SAM2 extensions as long as you have the right NVIDIA toolkit installed.
    More information here: https://github.com/facebookresearch/sam2/blob/main/INSTALL.md

3.  **Install remaining dependencies:**
    ```bash
    pip install opencv-python tqdm Pillow
    ```

## Usage

Place the final script (`SAM2Tracking.py`) in the root of the cloned `sam2` repository. This is necessary because the script imports directly from the `sam2` library. Ensure your data is also accessible from this location.

Run the script from your terminal:

```bash
python SAM2Tracking.py \
    --image_dir path/to/your/image_sequence \
    --json_annotation path/to/your/annotation.json \
    --model_config_yaml "configs/sam2.1/sam2.1_hiera_l.yaml" \
    --local_checkpoint_path path/to/your/sam2.1_hiera_large.pt \
    --output_dir "my_tracking_results" \
    --make_video
```

**For maximum speed (if you only need the JSONL data):**

```bash
python SAM2Tracking.py \
    --image_dir path/to/your/image_sequence \
    --json_annotation path/to/your/annotation.json \
    --model_config_yaml "configs/sam2.1/sam2.1_hiera_l.yaml" \
    --local_checkpoint_path path/to/your/sam2.1_hiera_large.pt \
    --no_save_annotated_frames
```

**Note**:
Inside sam2_video_predictor.py toggle to True if your input directory if you get CUDA out of memory errors.

```python     
    offload_video_to_cpu=True,
    offload_state_to_cpu=True,
```

**Reccomended Change**

You should install Pillow-SIMD for a substantial boost in image loading performance.
```
$ pip uninstall pillow
$ CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
```

### Command-Line Arguments

-   `--image_dir`: (Required) Path to the input image sequence folder. Frames should be named numerically (e.g., `1.jpg`, `2.jpg`, ... `10.jpg`).
-   `--json_annotation`: (Required) Path to the JSON annotation file for the first frame.
-   `--model_config_yaml`: (Required) Path to the model's YAML configuration file.
-   `--local_checkpoint_path`: (Required) Path to the local `.pt` model checkpoint file.
-   `--output_dir`: Directory to save all outputs. (Default: `sam2_output`)
-   `--mask_threshold`: Threshold for binarizing mask logits. (Default: `0.0`)
-   `--make_video`: A flag to enable final video creation from the annotated frames. **Note:** This has no effect if `--no_save_annotated_frames` is used.
-   `--video_fps`: Framerate for the output video. (Default: `10.0`)
-   `--no_save_annotated_frames`: A flag to disable saving annotated frame images. This significantly speeds up processing and reduces disk usage if you only need the `tracked_detections.jsonl` data file.

## Input & Output

-   **Input JSON:** A single JSON object describing objects in the first frame, similar to LabelMe or Pascal VOC format. It must contain an `annotation` key with an `object` list. Each object in the list needs a `name` and a `bndbox` dictionary (`xmin`, `ymin`, `xmax`, `ymax`).

-   **Output Data File (`tracked_detections.jsonl`):** A JSON Lines file created in the output directory. Each line is a JSON object containing the tracking results for one frame. Each object includes a `name`, a persistent unique `uid` for the track, and the rectangular `points` of its bounding box.

-   **Visual Output (Optional):**
    -   A sub-directory named `annotated_frames` containing the annotated frames with semi-transparent masks and bounding boxes drawn for each tracked object. This is **not** created if `--no_save_annotated_frames` is set.
    -   If `--make_video` is specified, a final `tracking_video.mp4` compiling these frames is saved in the output directory.

## Acknowledgements

This work is built upon the official [SAM 2](https://github.com/facebookresearch/sam2) repository and its powerful video segmentation model.
