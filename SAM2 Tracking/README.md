# SAM2 Multi-Object Video Tracker

This project provides a command-line script to track multiple objects in an image sequence using the **Segment Anything 2 (SAM2)** model. It initializes objects on the first frame using bounding box prompts from a JSON file and leverages SAM2's built-in video tracking capabilities to propagate the masks through the entire sequence.

The script outputs a directory of annotated images (with both masks and bounding boxes), a JSON Lines file with frame-by-frame tracking data, and a final video compiling the results.

## Demo of Results

The tracker successfully identifies and follows multiple objects throughout the video sequence.

**Final Tracking Video Demo:**
![Tracking Demo GIF](output.gif)


### Installation Steps

1.  **Setup your enviroment according to the sam2 repo installation steps.**
    https://github.com/facebookresearch/sam2
    ```bash
    git clone https://github.com/facebookresearch/sam2.git
    cd sam2
    ```

3. **Download checkpoints from the sam2 repo.**
   ```
    cd checkpoints && \
    ./download_ckpts.sh && \
    cd ..
   ```

4.  ** If you encounter an error related to CUDA compilation.** 

    You can force-compile the SAM2 extensions as long as you have the right nvidia toolkit installed.
    More information here: https://github.com/facebookresearch/sam2/blob/main/INSTALL.md
    ```bash
    pip uninstall -y sam2
    rm -f ./sam2/_C*.so
    SAM2_BUILD_ALLOW_ERRORS=0 pip install -v -e .
    ```

5.  **Install remaining dependencies:**
    ```bash
    pip install opencv-python matplotlib
    ```

## Usage

Place the final script (`SAM2Tracking.py`) in the root of the cloned `sam2` repository. Ensure your data is also accessible from this location.

Run the script from your terminal:

```bash
python SAM2Tracking.py \
    --image_dir path/to/your/image_sequence \
    --json_annotation path/to/your/annotation.json \
    --model_config_yaml "configs/sam2.1/sam2.1_hiera_l.yaml" \
    --local_checkpoint_path path/to/your/sam2.1_hiera_large.pt \
    --make_video
```

### Command-Line Arguments

-   `--image_dir`: (Required) Path to the input image sequence folder.
-   `--json_annotation`: (Required) Path to the JSON annotation file for the first frame.
-   `--model_config_yaml`: (Required) Path to the model's YAML configuration file.
-   `--local_checkpoint_path`: (Required) Path to the local `.pt` model checkpoint file.
-   `--output_dir`: Directory to save all outputs. (Default: `sam2_final_optimized_output`)
-   `--mask_threshold`: Threshold for binarizing mask logits. (Default: `0.0`)
-   `--make_video`: A flag to enable final video creation from annotated frames.
-   `--video_fps`: Framerate for the output video. (Default: `10.0`)

## Input & Output

-   **Input JSON:** A single JSON object describing objects in the first frame, similar to LabelMe or Pascal VOC format. It must contain an `object` list with `bndbox` dictionaries (`xmin`, `ymin`, `xmax`, `ymax`).
-   **Output Data File (`tracked_detections.jsonl`):** A JSON Lines file where each line is a JSON object containing the tracking results for one frame. Each object includes a `name`, a unique `uid`, and the rectangular `points` of its bounding box.
-   **Visual Output:**
    -   A folder of annotated frames with semi-transparent masks and bounding boxes drawn for each tracked object.
    -   A final `tracking_video.mp4` compiling these frames.

## Acknowledgements

This work is built upon the official [SAM 2](https://github.com/facebookresearch/sam2) repository and its powerful video segmentation model.
