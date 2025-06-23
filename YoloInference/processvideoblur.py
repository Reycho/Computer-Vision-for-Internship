import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import shared_memory
import queue
import time
import os
import av
import signal
import psutil
from pathlib import Path
from typing import List

# =================================================================================
# --- Configuration & Hyperparameters ---
# =================================================================================
INPUT_VIDEO_PATH = Path("/home/ryan/yolov12/D2-P3-pm-2-uno-2.mp4")
OUTPUT_VIDEO_PATH = Path("video_blurred.mp4")
MODEL_PATH = Path("/home/ryan/yolov12/facedetection.pt")

# --- Video Codec Configuration ---
# Use NVIDIA's CUVID for hardware-accelerated decoding
HW_DECODER_CODEC = 'h264_cuvid'

# NEW: Configuration for the writer for smaller file sizes
# Use NVIDIA's NVENC for hardware-accelerated H.264 encoding
HW_ENCODER_CODEC = 'h264_nvenc'
# Best fallback: High-quality CPU-based H.264 encoder
CPU_ENCODER_CODEC = 'libx264'
# Encoder options for GPU. 'preset' affects speed/compression. 'cq' controls quality.
# Lower 'cq' = higher quality, larger file. Good range: 23-28.
ENCODER_OPTIONS = {
    'preset': 'slow',
    'cq': '24'
}
# Fallback options for CPU. 'crf' is the CPU equivalent of 'cq'.
CPU_ENCODER_OPTIONS = {
    'preset': 'slow',
    'crf': '23'
}

# --- Pipeline Tuning ---
MAX_BATCH_SIZE = 100
BATCH_COLLECTION_TIMEOUT = 0.01
RAW_SHM_POOL_SIZE = 600
PROCESSED_SHM_POOL_SIZE = 600
QUEUE_SIZE = max(RAW_SHM_POOL_SIZE, PROCESSED_SHM_POOL_SIZE)
CONF_THRESHOLD = 0.2
g_all_shm_names = []

# =================================================================================
# --- Process & Resource Management ---
# =================================================================================
def cleanup_shm(signum=None, frame=None):
    """Gracefully cleans up shared memory blocks on exit or interrupt."""
    print("\nCaught signal, cleaning up all shared memory...")
    cleaned_count = 0
    # Use a copy of the list to avoid issues if it's modified elsewhere
    for name in list(g_all_shm_names):
        try:
            shm = shared_memory.SharedMemory(name=name)
            shm.close()
            shm.unlink()
            cleaned_count += 1
        except FileNotFoundError:
            pass
    print(f"Cleaned up {cleaned_count} shared memory blocks.")
    if signum is not None:
        current_process = psutil.Process()
        for child in current_process.children(recursive=True):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        current_process.kill()
        exit(1)

def set_process_affinity_and_priority(pid: int, cores: List[int], priority: str = 'normal'):
    """Sets CPU affinity and process priority for better performance."""
    try:
        p = psutil.Process(pid)
        p.cpu_affinity(cores)
        if os.name == 'posix':
            priority_map = {'high': -10, 'normal': 0, 'low': 10}
            p.nice(priority_map.get(priority, 0))
        print(f"[{p.name()}/{pid}] Pinned to CPU(s) {cores}, Priority '{priority}' (nice={p.nice()})")
    except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError) as e:
        print(f"Warning: Could not set affinity/priority for PID {pid} on cores {cores}: {e}")

def reader_process_wrapper(cores, *args):
    set_process_affinity_and_priority(os.getpid(), cores, 'high')
    _pyav_reader_logic(*args)

def blur_worker_process_wrapper(cores, *args):
    set_process_affinity_and_priority(os.getpid(), cores, 'low')
    _blur_worker_logic(*args)

def writer_process_wrapper(cores, *args):
    set_process_affinity_and_priority(os.getpid(), cores, 'normal')
    _pyav_writer_logic(*args)

# =================================================================================
# --- Core Process Logic ---
# =================================================================================
def _pyav_reader_logic(hw_decoder_codec, video_path, frame_shape, frame_dtype, free_raw_shm_queue, ready_raw_shm_queue):
    """Reads frames, attempting HW acceleration with robust fallback."""
    container = None
    using_hw_accel = True
    try:
        print(f"[Reader] Attempting HW-accelerated decoding with '{hw_decoder_codec}'...")
        container = av.open(str(video_path), options={'video_codec': hw_decoder_codec})
        if hw_decoder_codec in container.streams.video[0].codec.name:
            print(f"[Reader] Success! Using GPU for video decoding.")
            using_hw_accel = True
        else:
            print(f"[Reader] HW codec '{hw_decoder_codec}' requested but not used. Re-opening with CPU.")
            container.close()
            container = av.open(str(video_path))
    except Exception as e:
        print(f"[Reader] HW-acceleration failed: {e}. Falling back to CPU decoding.")
        if container: container.close()
        container = av.open(str(video_path))

    try:
        stream = container.streams.video[0]
        if not using_hw_accel:
            stream.thread_type = "AUTO"
            print(f"[Reader] CPU fallback is active. Using up to {stream.codec_context.thread_count} threads for decoding.")
        
        total_frames = stream.frames if stream.frames > 0 else int(stream.duration * stream.average_rate)
        with tqdm(total=total_frames, desc="[Reader]", position=1, leave=False, dynamic_ncols=True) as pbar:
            for frame_index, frame in enumerate(container.decode(video=0)):
                shm_name = free_raw_shm_queue.get()
                shm = shared_memory.SharedMemory(name=shm_name)
                np.copyto(np.ndarray(frame_shape, dtype=frame_dtype, buffer=shm.buf), frame.to_ndarray(format='bgr24'))
                ready_raw_shm_queue.put((frame_index, shm_name))
                shm.close()
                pbar.update(1)
    except Exception as e:
        print(f"[Reader] Runtime Error: {e}")
    finally:
        ready_raw_shm_queue.put(None)
        if container: container.close()

import numpy as np
import cv2
from multiprocessing import shared_memory

# This function would be in your main script
def _blur_worker_logic(results_queue, processed_shm_queue, free_raw_shm_queue, free_processed_shm_queue, frame_shape, frame_dtype):
    """
    Applies blur to faces based on tracking results from YOLO.
    This version is designed to work with the output of `model.track()`.
    """
    while True:
        # Get the next task from the GPU process.
        # The data format is now (frame_index, shm_name, tracked_boxes_object)
        data = results_queue.get()
        if data is None:
            break  # Sentinel value received, exit the loop.

        index, raw_shm_name, tracked_boxes = data
        processed_shm_name = None  # Ensure it's defined in case of early errors

        try:
            # 1. Acquire a free shared memory block for the processed (blurred) frame.
            processed_shm_name = free_processed_shm_queue.get()

            # 2. Attach to both the input (raw) and output (processed) shared memory blocks.
            raw_shm = shared_memory.SharedMemory(name=raw_shm_name)
            processed_shm = shared_memory.SharedMemory(name=processed_shm_name)

            # 3. Create NumPy array "views" into the shared memory. This is a zero-copy operation.
            frame_in = np.ndarray(frame_shape, dtype=frame_dtype, buffer=raw_shm.buf)
            frame_out = np.ndarray(frame_shape, dtype=frame_dtype, buffer=processed_shm.buf)
            
            # 4. Start by copying the original frame to the output buffer.
            # We will apply the blur directly to this `frame_out` buffer.
            np.copyto(frame_out, frame_in)

            # 5. --- CORE LOGIC CHANGE ---
            # Check if the tracker returned any valid tracks for this frame.
            # The `tracked_boxes.id` attribute will not be None if tracks exist.
            if tracked_boxes.id is not None:
                # Loop through each box that is part of a confirmed track.
                for box in tracked_boxes:
                    # Get the coordinates from the box object.
                    x1, y1, x2, y2 = np.array(box.xyxy[0].cpu(), dtype=int)
                    
                    # Optional: get the track ID for debugging or advanced logic
                    # track_id = int(box.id[0].cpu())
                    # print(f"Frame {index}: Blurring face with Track ID {track_id}")

                    # Ensure coordinates are within the frame boundaries to prevent errors.
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame_shape[1], x2), min(frame_shape[0], y2)
                    
                    # Select the region of interest (the face) from the output frame.
                    roi = frame_out[y1:y2, x1:x2]

                    # Apply blur only if the region is valid (has a size greater than 0).
                    if roi.size > 0:
                        # Dynamically calculate a suitable kernel size for the Gaussian blur.
                        # This makes the blur look good on both small and large faces.
                        k_size = max(5, int(min(roi.shape[0], roi.shape[1]) * 0.5))
                        # The kernel size must be an odd number.
                        k_size = k_size if k_size % 2 != 0 else k_size + 1
                        
                        # Apply the blur in-place to the region of interest in the output frame.
                        frame_out[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k_size, k_size), 30)
            
            # 6. Send the processed frame to the writer process.
            processed_shm_queue.put((index, processed_shm_name))

            # 7. Detach from the shared memory blocks.
            raw_shm.close()
            processed_shm.close()
            
            # 8. CRITICAL: Return the raw memory block to the pool for the Reader to reuse.
            free_raw_shm_queue.put(raw_shm_name)

        except Exception as e:
            # Robust error handling: ensure memory blocks are returned to the pools.
            print(f"[Blur Worker] Error on frame {index}: {e}")
            # Return the raw buffer so it's not lost.
            free_raw_shm_queue.put(raw_shm_name)
            # If we acquired a processed buffer before the error, return it too.
            if processed_shm_name:
                free_processed_shm_queue.put(processed_shm_name)

def _pyav_writer_logic(output_path, properties, processed_shm_queue, free_processed_shm_queue, total_frames, frame_shape, frame_dtype):
    """Writes frames to a video file using a GPU-accelerated encoder with a CPU fallback."""
    w, h, fps = properties
    frame_buffer = {}; next_frame_to_write = 0
    try:
        with av.open(str(output_path), mode='w') as container:
            stream = None
            try:
                print(f"[Writer] Attempting to use GPU encoder: '{HW_ENCODER_CODEC}'")
                stream = container.add_stream(HW_ENCODER_CODEC, rate=fps)
                stream.options = ENCODER_OPTIONS
                print(f"[Writer] Success! Using GPU for video encoding with options: {ENCODER_OPTIONS}")
            except Exception as e:
                print(f"[Writer] GPU encoder failed: {e}. Falling back to CPU encoder: '{CPU_ENCODER_CODEC}'")
                stream = container.add_stream(CPU_ENCODER_CODEC, rate=fps)
                stream.options = CPU_ENCODER_OPTIONS
                print(f"[Writer] Using CPU for video encoding with options: {CPU_ENCODER_OPTIONS}")
            
            stream.width = w
            stream.height = h
            stream.pix_fmt = 'yuv420p'  # Crucial for compatibility

            with tqdm(total=total_frames, desc="[Writer]", position=2, leave=False, dynamic_ncols=True) as pbar:
                while next_frame_to_write < total_frames:
                    try:
                        data = processed_shm_queue.get(timeout=120)
                        if data is None: break
                        
                        index, shm_name = data
                        frame_buffer[index] = shm_name
                        
                        while next_frame_to_write in frame_buffer:
                            current_shm_name = frame_buffer.pop(next_frame_to_write)
                            shm = shared_memory.SharedMemory(name=current_shm_name)
                            frame_np = np.ndarray(frame_shape, dtype=frame_dtype, buffer=shm.buf)
                            av_frame = av.VideoFrame.from_ndarray(frame_np, format='bgr24')
                            
                            for packet in stream.encode(av_frame):
                                container.mux(packet)
                            
                            pbar.update(1)
                            next_frame_to_write += 1
                            shm.close()
                            free_processed_shm_queue.put(current_shm_name)
                    except queue.Empty:
                        print("\n[Writer] Timed out waiting for frames. Pipeline may be stalled.")
                        break
                
                # Flush any remaining frames in the encoder
                for packet in stream.encode():
                    container.mux(packet)
    except Exception as e:
        print(f"[Writer] FATAL ERROR during video writing: {e}")


# =================================================================================
# --- Main Execution Block ---
# =================================================================================
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    signal.signal(signal.SIGINT, cleanup_shm)
    signal.signal(signal.SIGTERM, cleanup_shm)

    if not INPUT_VIDEO_PATH.exists():
        raise SystemExit(f"FATAL: Input video not found: '{INPUT_VIDEO_PATH}'")

    total_cores = os.cpu_count() or 1
    try:
        with av.open(str(INPUT_VIDEO_PATH)) as c:
            s = c.streams.video[0]
            total_frames = s.frames if s.frames > 0 else int(s.duration * s.average_rate)
            fps, h, w = s.average_rate, s.height, s.width
            frame_shape, frame_dtype = (h, w, 3), np.uint8
            frame_size_bytes = np.ndarray(shape=frame_shape, dtype=frame_dtype).nbytes
            print(f"Video Info: {w}x{h} @ {float(fps):.2f} FPS, {total_frames} frames.")
    except Exception as e:
        raise SystemExit(f"FATAL: Could not analyze video: {e}")

    if total_cores < 6:
        print(f"Warning: Only {total_cores} cores. Using simpler allocation.")
        reader_cores = [0]
        main_cores = [1] if total_cores > 1 else [0]
        writer_cores = [2] if total_cores > 2 else main_cores
        all_worker_cores = list(range(3, total_cores)) if total_cores > 3 else writer_cores
        NUM_WORKERS = len(all_worker_cores) if all_worker_cores else 1
    else:
        print("Using 'Balanced I/O' CPU core allocation.")
        reader_cores, writer_cores, main_cores = [0, 1], [2, 3], [4]
        all_worker_cores = list(range(5, total_cores))
        if not all_worker_cores: all_worker_cores = [total_cores - 1]
        NUM_WORKERS = len(all_worker_cores)
    
    print("-" * 50)
    print(f"CPU Allocation: Reader {reader_cores}, Main/GPU {main_cores}, Writer {writer_cores}, {NUM_WORKERS} Workers {all_worker_cores}")
    set_process_affinity_and_priority(os.getpid(), main_cores, 'high')

    def create_shm_pool(pool_size, prefix):
        pool, q = [], mp.Queue()
        for i in range(pool_size):
            name = f'yolo_shm_{prefix}_{i}_{os.getpid()}' # Add PID to name for uniqueness
            try:
                shm = shared_memory.SharedMemory(name=name, create=True, size=frame_size_bytes)
                pool.append(shm)
                g_all_shm_names.append(name)
                q.put(name)
            except FileExistsError:
                # If a previous unclean exit left shm, unlink and recreate
                try:
                    shared_memory.SharedMemory(name=name).unlink()
                except FileNotFoundError: pass
                shm = shared_memory.SharedMemory(name=name, create=True, size=frame_size_bytes)
                pool.append(shm)
                g_all_shm_names.append(name)
                q.put(name)
        return pool, q
        
    raw_shm_pool, free_raw_shm_queue = create_shm_pool(RAW_SHM_POOL_SIZE, 'raw')
    processed_shm_pool, free_processed_shm_queue = create_shm_pool(PROCESSED_SHM_POOL_SIZE, 'processed')
    ready_raw_shm_queue = mp.Queue(maxsize=QUEUE_SIZE)
    results_queue = mp.Queue(maxsize=QUEUE_SIZE)
    processed_shm_queue = mp.Queue(maxsize=QUEUE_SIZE)

    reader_args = (HW_DECODER_CODEC, INPUT_VIDEO_PATH, frame_shape, frame_dtype, free_raw_shm_queue, ready_raw_shm_queue)
    reader = mp.Process(target=reader_process_wrapper, args=(reader_cores, *reader_args), name="Reader")
    
    writer_props = (w, h, fps)
    writer_args = (OUTPUT_VIDEO_PATH, writer_props, processed_shm_queue, free_processed_shm_queue, total_frames, frame_shape, frame_dtype)
    writer = mp.Process(target=writer_process_wrapper, args=(writer_cores, *writer_args), name="Writer")
    
    blur_workers = [mp.Process(target=blur_worker_process_wrapper, args=([all_worker_cores[i % len(all_worker_cores)]], results_queue, processed_shm_queue, free_raw_shm_queue, free_processed_shm_queue, frame_shape, frame_dtype), name=f"BlurWorker-{i}") for i in range(NUM_WORKERS)]
    
    processes = [reader, writer] + blur_workers
    for p in processes: p.start()

    model = YOLO(MODEL_PATH)
    pbar_gpu = tqdm(total=total_frames, desc="[GPU]", position=0, dynamic_ncols=True)
    shm_map = {}
    main_loop_active = True
    
    try:
        while main_loop_active:
            batch_data = []
            try:
                start_time = time.monotonic()
                while len(batch_data) < MAX_BATCH_SIZE and (time.monotonic() - start_time) < BATCH_COLLECTION_TIMEOUT:
                    data = ready_raw_shm_queue.get(timeout=0.001)
                    if data is None:
                        main_loop_active = False
                        break
                    batch_data.append(data)
            except queue.Empty:
                if not main_loop_active and not batch_data:
                    break
            
            if not batch_data and not main_loop_active: break
            if not batch_data: continue

            batch_indices, batch_shm_names, batch_frames_views = [], [], []
            for index, shm_name in batch_data:
                if shm_name not in shm_map:
                    shm_map[shm_name] = shared_memory.SharedMemory(name=shm_name)
                batch_indices.append(index)
                batch_shm_names.append(shm_name)
                batch_frames_views.append(np.ndarray(frame_shape, dtype=frame_dtype, buffer=shm_map[shm_name].buf))
            
            if not batch_frames_views: continue
            
            results_batch = model.track(
                batch_frames_views,
                conf=CONF_THRESHOLD,    # Initial detection confidence
                persist=True,           # Crucial: tells the tracker to remember tracks between batches
                tracker="bytetrack.yaml", # Specify the tracker configuration
                verbose=False
            )
            
            for i, results in enumerate(results_batch):
                results_queue.put((batch_indices[i], batch_shm_names[i], results.boxes))
            
            pbar_gpu.update(len(batch_frames_views))
    finally:
        pbar_gpu.close()
        print("\n[Main] All frames processed by GPU. Waiting for child processes to finish...")
        
        # Close all opened shm views in the main process
        for shm in shm_map.values():
            shm.close()

        for _ in range(NUM_WORKERS):
            results_queue.put(None)
        
        # Wait for processes to finish gracefully
        reader.join(timeout=30)
        for p in blur_workers:
            p.join(timeout=30)
        
        processed_shm_queue.put(None)
        writer.join(timeout=120)

        # Check if processes are still alive and terminate if needed
        for p in processes:
            if p.is_alive():
                print(f"Warning: Process {p.name} did not exit gracefully. Terminating.")
                p.terminate()

        print("[Main] All child processes have joined.")
        cleanup_shm()
        print("-" * 50)
        print(f"Processing complete. Output saved to: '{OUTPUT_VIDEO_PATH}'")
        print("-" * 50)