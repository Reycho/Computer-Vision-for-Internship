import os
import json
from collections import Counter
from tqdm import tqdm

# =============================================================================
# --- CONFIGURATION ---
# =============================================================================
# 1. Define the input and output folders.
SOURCE_JSON_DIR = '/home/ryan/yolov12/MIND_FLOW'
OUTPUT_LABELS_DIR = '/home/ryan/yolov12/templabels'

# 2. Set the minimum number of instances for a subcategory to be included.
INSTANCE_COUNT_THRESHOLD = 2000
# =============================================================================


def get_image_details(data, json_filename):
    """
    Extracts image name and dimensions from a loaded JSON object,
    handling different formats.
    """
    # Format 1: MIND_FLOW style (has 'frameInfo')
    if 'frameInfo' in data:
        frame_info = data.get('frameInfo', {})
        image_name = frame_info.get('frameName')
        metadata = frame_info.get('metadata', {})
        width = metadata.get('width')
        height = metadata.get('height')
        return image_name, width, height

    # Format 2: Camera Key style (e.g., {"FrontCam01": {...}})
    elif isinstance(data, dict) and len(data) == 1:
        # The image name must be derived from the JSON filename itself
        image_name = os.path.basename(json_filename).replace('.json', '.jpeg')
        # This format doesn't seem to contain dimensions, return None to signal this
        return image_name, None, None

    return None, None, None

def get_objects_list(data):
    """
    Extracts the list of annotation objects from a loaded JSON,
    handling different formats.
    """
    # Format 1: MIND_FLOW style
    if 'objects' in data:
        return data.get('objects', [])

    # Format 2: Camera Key style
    if isinstance(data, dict) and len(data) == 1:
        cam_key = next(iter(data), None)
        if cam_key:
            return data[cam_key].get('labelData', [])

    return []


def extract_subcategory(obj):
    """
    Extracts the specific subcategory name from a single object dictionary.
    This is the most critical part.
    """
    properties = obj.get('properties', {})
    
    # Check for static objects first, as their structure is unique
    alias = obj.get('aliasName')
    if alias == 'static_object' or obj.get('labelName') == '静态障碍物':
        # MIND_FLOW format for static objects
        category_prop = properties.get('category', {})
        value_list = category_prop.get('value', [])
        if value_list:
            # "cone/锥桶" -> "cone"
            return value_list[0].split('/')[0]
        # Camera Key format for static objects
        for key in properties:
            if key.startswith('category.'):
                return key.split('.')[-1]

    # Handle other object types (vehicles, VRUs)
    item_key = properties.get('itemKey')
    if item_key: # This is the primary key in the "Camera Key" format
        return item_key
    
    # Fallback for MIND_FLOW vehicles
    if alias == 'vehicle':
        for key in properties:
            if key.startswith('category.'):
                return key.split('.')[-1]

    return None


def get_yolo_bbox_string(obj, img_width, img_height):
    """
    Extracts and normalizes bounding box coordinates.
    """
    geom = obj.get('geometry', {})
    
    # MIND_FLOW format: {'left': ..., 'top': ..., 'width': ..., 'height': ...}
    if all(k in geom for k in ['left', 'top', 'width', 'height']):
        x_center = (geom['left'] + geom['width'] / 2) / img_width
        y_center = (geom['top'] + geom['height'] / 2) / img_height
        width_norm = geom['width'] / img_width
        height_norm = geom['height'] / img_height
        return f"{x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}"

    # Camera Key format: {'points': [{'x':..,'y':..}, ...]}
    elif 'points' in geom:
        points = geom.get('points', [])
        if not points or len(points) != 4: return None
        x_coords = [p['x'] for p in points]
        y_coords = [p['y'] for p in points]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        box_width = x_max - x_min
        box_height = y_max - y_min
        center_x = x_min + box_width / 2
        center_y = y_min + box_height / 2
        
        norm_center_x = center_x / img_width
        norm_center_y = center_y / img_height
        norm_width = box_width / img_width
        norm_height = box_height / img_height
        return f"{norm_center_x:.6f} {norm_center_y:.6f} {norm_width:.6f} {norm_height:.6f}"

    return None

def main():
    """Main execution function."""
    os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)
    json_files = [f for f in os.listdir(SOURCE_JSON_DIR) if f.endswith('.json')]
    
    if not json_files:
        print(f"Error: No JSON files found in '{SOURCE_JSON_DIR}'.")
        return

    # Pass 1: Count all subcategories
    print(f"--- Pass 1: Counting subcategories in {len(json_files)} files... ---")
    subcategory_counts = Counter()
    for filename in tqdm(json_files, desc="Counting"):
        json_path = os.path.join(SOURCE_JSON_DIR, filename)
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            objects = get_objects_list(data)
            for obj in objects:
                subcategory = extract_subcategory(obj)
                if subcategory:
                    subcategory_counts[subcategory] += 1
        except Exception:
            continue

    print("\n--- Analysis Complete ---")
    if not subcategory_counts:
        print("\nFATAL ERROR: Could not extract any subcategories. Please double-check JSON formats and the `extract_subcategory` function.")
        return
        
    print("Found the following subcategories and their counts:")
    for subcat, count in sorted(subcategory_counts.items(), key=lambda item: item[1], reverse=True):
        print(f"  - '{subcat}': {count} instances")

    # Dynamically create class mapping based on threshold
    print(f"\nFiltering for subcategories with >= {INSTANCE_COUNT_THRESHOLD} instances...")
    
    # We only care about static objects for this training run.
    static_object_types = {'cone', 'water_barrier', 'parking_barrier', 'pole', 'crash_barrel', 'A_board', 'trashcan', 'wheel_stopper'}
    
    qualified_subcategories = sorted([
        subcat for subcat, count in subcategory_counts.items() 
        if count >= INSTANCE_COUNT_THRESHOLD and subcat in static_object_types
    ])
    
    if not qualified_subcategories:
        print(f"\nError: No static object subcategories met the threshold of {INSTANCE_COUNT_THRESHOLD}. Exiting.")
        return

    class_mapping = {name: i for i, name in enumerate(qualified_subcategories)}
    
    print("The following classes will be created for training:")
    for name, class_id in class_mapping.items():
        print(f"  - Class ID {class_id}: '{name}'")

    # Pass 2: Convert qualified subcategories to YOLO format
    print(f"\n--- Pass 2: Converting {len(qualified_subcategories)} classes to YOLO format... ---")
    for filename in tqdm(json_files, desc="Converting files"):
        json_path = os.path.join(SOURCE_JSON_DIR, filename)
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            image_name, img_width, img_height = get_image_details(data, filename)
            
            # Critical check: if dimensions are missing, we cannot proceed for this file.
            if not all([image_name, img_width, img_height]):
                continue

            objects = get_objects_list(data)
            yolo_annotations = []
            for obj in objects:
                subcategory = extract_subcategory(obj)
                if subcategory in class_mapping:
                    class_id = class_mapping[subcategory]
                    bbox_line = get_yolo_bbox_string(obj, img_width, img_height)
                    if bbox_line:
                        yolo_annotations.append(f"{class_id} {bbox_line}")

            if yolo_annotations:
                output_filename = os.path.splitext(image_name)[0] + '.txt'
                output_path = os.path.join(OUTPUT_LABELS_DIR, output_filename)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(yolo_annotations))
        except Exception as e:
            # print(f"\nWarning: Failed to convert {filename}. Error: {e}")
            continue

    print("\n--- Conversion Complete! ---")
    print("\n--- YAML File Content (for your data.yaml) ---")
    print("# This YAML is auto-generated based on the subcategories that met the threshold.")
    print("# Update the path to your dataset root directory if necessary.")
    print("path: /path/to/your/dataset/root") 
    print("train: images/train")
    print("val: images/val")
    print(f"\nnc: {len(class_mapping)}")
    print("names:")
    for name in qualified_subcategories:
        print(f"  - {name}")

if __name__ == '__main__':
    main()