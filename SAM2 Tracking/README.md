# SAM2 Multi-Object Video Tracker

This project uses the SAM2 model to perform multi-object tracking on a sequence of video frames. It initializes objects on the first frame using bounding box prompts from a JSON file and tracks them through the entire sequence. The JSON file format is specific to requirements from my job.

## Final Result

The final tracking output has been compiled into the video below.

**[Click here to watch the tracking video](ExampleTrackedVideo.mp4)**


---

## Technical Summary & Challenges

Successfully running this project required a highly specific and clean environment due to the model's sensitivity to library and toolkit versions.

**Key Requirements for Reproduction:**

*   **PyTorch & CUDA Alignment:** The environment must use a PyTorch build compiled with the exact CUDA toolkit version installed on the system (in this case, `torch==2.5.1` for `cu124`).
*   **Forced Extension Compilation:** The native SAM2 C++ CUDA extensions had to be force-compiled after installing the CUDA 12.4 toolkit. The default `pip install` resulted in a silent build failure, which caused numerical instability and tracking failure.
*   **Correct Model Logic:** The `SAM2VideoPredictor` requires that all model interactions (from `init_state` through propagation) occur within a `torch.autocast` context and that pixel coordinate prompts are used with `normalize_coords=True`.


