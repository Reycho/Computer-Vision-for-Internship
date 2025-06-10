# Multi-Object Tracking from Annotations using SAM & AOT

This project provides a command-line script to track multiple objects in an image sequence using a single-frame JSON annotation file. It uses the Segment Anything Model (SAM) for initial segmentation and an Adaptive Object Tracker (AOT) for tracking.

The script outputs a directory of annotated images and a JSON Lines file with frame-by-frame tracking data.

# Demo
![Tracking Demo GIF](output.gif)


## Note
Please use SAM2 Tracking as it is simply more accurate.

## Setup

Follow original Repo install guide: [Segment-and-Track-Anything](https://github.com/z-x-yang/Segment-and-Track-Anything)

## Usage

Run the script from your terminal:

```bash
python tracking_script.py \
    --image_dir path/to/image/sequence \
    --json_annotation path/to/first_frame_annotation.json
```


### Command-Line Arguments

-   `--image_dir`: (Required) Path to the input image sequence folder.
-   `--json_annotation`: (Required) Path to the JSON annotation file for the first frame.
-   `--base_output_dir`: Base directory for all output folders. (Default: `tracking_results`)
-   `--model_variant`: SAM model type: `vit_b`, `vit_l`, `vit_h`. (Default: `vit_b`)
-   `--aot_model_variant`: Tracker model: `deaotb`, `deaotl`, `r50_deaotl`. (Default: `r50_deaotl`)
-   `--object_name_filter`: Track only objects with a specific name from the JSON (e.g., "car").
-   `--bbox_thickness`: Thickness of the drawn bounding boxes. (Default: `2`)

## Input & Output

-   **Input JSON:** A single JSON object describing objects in the first frame (e.g., from LabelMe).
-   **Output Data File:** A `.txt` file in JSON Lines format, with each line containing the tracking results for one frame (`fileName`, `prelabels` list with `name`, `uid`, and `points` for each object).

## Acknowledgements

This work is built upon the [Segment-and-Track-Anything](https://github.com/z-x-yang/Segment-and-Track-Anything) project and its core components (SAM, AOT, GroundingDINO).
