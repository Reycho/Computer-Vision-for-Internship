# Computer Vision Scripts

This repository is a collection of high-performance Python scripts developed during a computer vision internship. The projects focus on two primary areas: object tracking in video and end-to-end YOLO pipelines for object detection.

Please be aware that most scripts in this repository were developed for high-end systems (e.g., 64GB+ RAM, 24GB+ VRAM). You may need to adjust parameters like batch sizes to run them your hardware.

## Project Directory

This repository is organized into three distinct project folders. Each folder is self-contained with its own dependencies and detailed `README.md` file.

| Directory | Primary Focus | Key Technology |
| :--- | :--- | :--- |
| **[`YoloInference/`](./YoloInference/)** | **YOLOv8 Pipelines** | A suite of utilities for batch inference, dataset preparation (`convert.py`, `shuffle.py`), and video processing. |
| **[`SAM2 Tracking/`](./SAM2%20Tracking/)** | **Modern Video Tracking (Recommended)** | State-of-the-art multi-object tracking in video sequences using Meta AI's SAM2 model. |
| **[`SegmentAndTrackAnything/`](./SegmentAndTrackAnything/)** | **Legacy Video Tracking** | An alternative tracker using the original SAM and AOT. **Note:** `SAM2 Tracking` is the superior and recommended version. |

## Getting Started

1.  **Clone the Repository**
    ```bash
    git clone https://your-repository-url.git
    cd your-repository-name
    ```

2.  **Choose a Project**
    Navigate into the project folder you are interested in (e.g., `cd YoloInference/`).

3.  **Follow Local Instructions**
    Each folder contains a dedicated `README.md` with specific installation and usage instructions. These local READMEs are the primary source of truth for each project.

## Core Technologies

This repository leverages several powerful open-source computer vision frameworks:

-   **Ultralytics YOLO**: For object detection, training, and inference.
-   **Meta AI's SAM & SAM2**: For segmentation and video object tracking.
-   **SAHI (Slicing Aided Hyper Inference)**: For improving small object detection.
-   **PyAV**: For efficient, hardware-accelerated video I/O.
-   **PyTorch**: The backend deep learning framework for all models.
