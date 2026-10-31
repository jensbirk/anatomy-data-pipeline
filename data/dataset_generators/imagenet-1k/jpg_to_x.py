import os
import shutil
import argparse
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def process_image(input_path, target_path):
    try:
        with Image.open(input_path) as img:
            rgb_im = img.convert('RGB')
            # Save as high-quality JPEG
            rgb_im.save(target_path, format="JPEG", quality=95)
        return True, target_path.name
    except Exception as e:
        return False, f"{input_path.name} (Error: {e})"

def build_webdataset(image_paths, source_path, target_dir):
    try:
        import webdataset as wds
        target_dir.mkdir(parents=True, exist_ok=True)
        
        pattern = str(target_dir / "dataset-%04d.tar")
        
        # Gets unique parent folder names (excluding the root source path itself)
        classes = sorted(list(set(
            path.parent.name for path in image_paths if path.parent != source_path
        )))
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        
        print(f"\nFound {len(classes)} classes. Mapping created.")
        
        print("Packing images into WebDataset shards...")
        with wds.ShardWriter(pattern, maxcount=10000) as sink:
            for filepath in tqdm(image_paths, desc="Writing WDS"):
                with open(filepath, "rb") as f:
                    img_data = f.read()
                
                rel_path = filepath.relative_to(source_path)
                key = str(rel_path.with_suffix('')).replace(os.sep, "__")
                
                # Get class label and format for WDS
                class_name = filepath.parent.name
                class_idx = class_to_idx.get(class_name, 0) # Defaults to 0 if image is in root
                
                sink.write({
                    "__key__": key,
                    "jpg": img_data,                             # Force 'jpg' key
                    "cls": str(class_idx).encode("ascii")        # Save label as bytes
                })
                
        print(f"WebDataset created at: {target_dir}")
        
        # Save the class mapping as a text file
        with open(target_dir / "class_mapping.txt", "w") as f:
            for cls_name, idx in class_to_idx.items():
                f.write(f"{idx}: {cls_name}\n")
                
    except ImportError:
        print("Error: 'webdataset' library missing. Run 'pip install webdataset'.")

def build_ffcv(image_paths, source_path, target_dir):
    try:
        from ffcv.writer import DatasetWriter
        from ffcv.fields import RGBImageField, IntField
        
        target_dir.mkdir(parents=True, exist_ok=True)
        beton_path = target_dir / "dataset.beton"
        
        # Build the class-to-index mapping
        classes = sorted(list(set(
            path.parent.name for path in image_paths if path.parent != source_path
        )))
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        print(f"\nFound {len(classes)} classes for FFCV. Mapping created.")
        print("Packing images into FFCV .beton format...")
        
        class SimpleImageDataset:
            def __init__(self, paths, class_map):
                self.paths = paths
                self.class_map = class_map
                
            def __len__(self):
                return len(self.paths)
                
            def __getitem__(self, idx):
                filepath = self.paths[idx]
                class_name = filepath.parent.name
                label = self.class_map.get(class_name, 0)
                # print(f"Processing {filepath} with label {label}")
                
                # FFCV Writer expects PIL Images
                img = Image.open(filepath).convert('RGB')
                return img, label

        dataset = SimpleImageDataset(image_paths, class_to_idx)
        
        writer = DatasetWriter(str(beton_path), {
            'image': RGBImageField(max_resolution=1000),
            'label': IntField()
        }, num_workers=4)
        
        writer.from_indexed_dataset(dataset)
        print(f"FFCV dataset successfully created at: {beton_path}")
        
    except ImportError:
        print("Error: 'ffcv' library missing. Run 'pip install ffcv'.")
    except Exception as e:
        print(f"FFCV Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Convert images, build WebDataset or FFCV (.beton).")
    parser.add_argument("-s", "--source", required=True, help="Path to source directory containing JPEGs.")
    parser.add_argument("-t", "--target", required=True, help="Path to base target directory.")
    parser.add_argument("-f", "--format", choices=["jpg", "wds", "ffcv", "all"], default="wds", 
                        help="Choose output format: jpg, wds, ffcv, or all. (Default: wds)")
    parser.add_argument("-w", "--workers", type=int, default=4, 
                        help="Number of parallel processing threads (Defaults to CPU cores).")
    
    args = parser.parse_args()
    source_path = Path(args.source)
    target_path = Path(args.target)
    
    if not source_path.exists() or not source_path.is_dir():
        print(f"Error: Source directory '{args.source}' does not exist.")
        return

    do_jpg = args.format in ["jpg", "all"]
    do_wds = args.format in ["wds", "all"]
    do_ffcv = args.format in ["ffcv", "all"]

    image_tasks = []
    source_images = []
    
    print("\nScanning directory...")
    
    for item in source_path.rglob("*"):
        if not item.is_file():
            continue
        
        relative_path = item.relative_to(source_path)
        
        if item.suffix.lower() in ['.jpg', '.jpeg']:
            source_images.append(item)
            
            if do_jpg:
                jpg_dir = target_path / "jpg"
                new_path = jpg_dir / relative_path.with_suffix('.jpg')
                new_path.parent.mkdir(parents=True, exist_ok=True)
                image_tasks.append((item, new_path))
        
        elif do_jpg:
            jpg_dir = target_path / "jpg"
            target_file = jpg_dir / relative_path
            target_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target_file)

    if image_tasks and do_jpg:
        print(f"\nStarting parallel JPG conversion using {args.workers} threads...")
        success_count = 0
        
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_file = {
                executor.submit(process_image, src, dest): dest 
                for src, dest in image_tasks
            }
            
            for future in tqdm(as_completed(future_to_file), total=len(image_tasks), desc="Converting to JPG"):
                success, result = future.result()
                if success:
                    success_count += 1
                else:
                    tqdm.write(f"Failed: {result}") 

        print(f"\nSuccessfully converted {success_count}/{len(image_tasks)} images to JPG.")

    if do_wds and source_images:
        wds_dir = target_path / "wds"
        build_webdataset(source_images, source_path, wds_dir)

    if do_ffcv and source_images:
        ffcv_dir = target_path / "ffcv"
        # Pass source_path to the worker
        build_ffcv(source_images, source_path, ffcv_dir)

    if not source_images:
        print("\nNo JPEG files found in the source directory.")

if __name__ == "__main__":
    main()