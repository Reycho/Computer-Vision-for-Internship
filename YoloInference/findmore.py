import os
import json
from collections import defaultdict, Counter

def discover_all_subcategories(directory_path):
    """
    Scans all JSON files in a directory to find and count all subcategories
    nested within the 'properties' of each object.

    Args:
        directory_path (str): The path to the folder containing the JSON files.

    Returns:
        A dictionary where keys are the main 'aliasName' and values are
        Counters of their specific subcategories.
    """
    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found at '{directory_path}'")
        return None

    # This will store our findings, e.g., {'vehicle': Counter({'car': 50, 'truck': 10})}
    all_subcategories = defaultdict(Counter)
    
    json_files = [f for f in os.listdir(directory_path) if f.endswith('.json')]
    total_files = len(json_files)
    
    print(f"Found {total_files} JSON files. Starting deep scan for all subcategories...")

    for i, filename in enumerate(json_files):
        file_path = os.path.join(directory_path, filename)

        if (i + 1) % 500 == 0 or (i + 1) == total_files:
            print(f"  Scanned {i + 1}/{total_files} files...")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for obj in data.get('objects', []):
                alias_name = obj.get('aliasName')
                if not alias_name:
                    continue
                
                properties = obj.get('properties', {})
                
                # Search for any key in properties that contains 'category'
                for prop_key, prop_value in properties.items():
                    if 'category' in prop_key and isinstance(prop_value, dict) and 'value' in prop_value:
                        
                        subcategory = None # Initialize subcategory

                        # NEW LOGIC: Prioritize subcategory from the key itself (e.g., "category.truck")
                        if prop_key.startswith('category.'):
                            # The subcategory is the part after the dot.
                            # e.g., "category.truck" -> "truck"
                            parts = prop_key.split('.')
                            if len(parts) > 1:
                                subcategory = parts[1]
                        
                        # FALLBACK: If key is generic (e.g., "category"), use the original logic
                        else:
                            # Extract the base category, e.g., "crash_barrel" from "crash_barrel/防撞桶"
                            full_string = prop_value['value'][0]
                            subcategory = full_string.split('/')[0]

                        # If we successfully found a subcategory, add it and break
                        if subcategory:
                            all_subcategories[alias_name][subcategory] += 1
                            # Once we've found the category for this object, move to the next object
                            break 

        except Exception as e:
            print(f"\nWarning: An error occurred with file {filename}: {e}")
            
    return all_subcategories

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # IMPORTANT: Point this to the directory with ALL your JSON files.
    # It's okay if they are not split into train/val yet.
    json_directory = '/home/ryan/Downloads/C6-5000-5000_2025-06-06-15-15-54 (5)/C6-5000-5000/MIND_FLOW'

    findings = discover_all_subcategories(json_directory)

    if findings:
        print("\n--- Discovery Complete ---")
        print("Found the following structure of aliases and subcategories:\n")
        
        # Sort by alias name for consistent output
        sorted_aliases = sorted(findings.items())
        
        for alias, subcategory_counter in sorted_aliases:
            print(f"Alias: '{alias}'")
            if not subcategory_counter:
                print("  (No subcategories found)")
            else:
                # Sort subcategories by name
                sorted_subcategories = sorted(subcategory_counter.items())
                for subcat_name, count in sorted_subcategories:
                    print(f"  - Subcategory: '{subcat_name}', Count: {count}")
            print("-" * 20)
    else:
        print("\nScan complete, but no subcategories could be found.")