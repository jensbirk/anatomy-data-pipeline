import os
import shutil
import argparse
from pathlib import Path
import SimpleITK as sitk
import h5py
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def process_med_image(input_path, target_path, target_format):
    try:
        # Read the 3D NIfTI volume
        img = sitk.ReadImage(str(input_path))
        
        if target_format == "nii":
            # Write out as uncompressed NIfTI
            sitk.WriteImage(img, str(target_path))
            
        elif target_format == "h5":
            # Extract raw numpy array and save to HDF5
            img_array = sitk.GetArrayFromImage(img)
            with h5py.File(str(target_path), "w") as hf:
                dset = hf.create_dataset("volume", data=img_array, compression="gzip")
                # Retain critical physical spacing and origin metadata
                dset.attrs["spacing"] = img.GetSpacing()
                dset.attrs["origin"] = img.GetOrigin()
                
        return True, target_path.name
    except Exception as e:
        return False, f"{input_path.name} (Error: {e})"

def main():
    # Command line arguments
    parser = argparse.ArgumentParser(description="Mirror a directory and convert .nii.gz files to .nii and/or .h5.")
    parser.add_argument("-s", "--source", required=True, help="Path to the source directory containing .nii.gz files.")
    parser.add_argument("-t", "--target", required=True, help="Path to the base target directory.")
    parser.add_argument("-f", "--format", choices=["nii", "h5", "all"], default="nii", 
                        help="Choose output format: nii, h5, or all. (Default: nii)")
    parser.add_argument("-w", "--workers", type=int, default=4, 
                        help="Number of parallel processing threads (Defaults to CPU cores).")
    
    args = parser.parse_args()
    
    source_path = Path(args.source)
    target_path = Path(args.target)
    
    if not source_path.exists() or not source_path.is_dir():
        print(f"Error: Source directory '{args.source}' does not exist.")
        return

    img_formats = ["nii", "h5"] if args.format == "all" else [args.format]
    image_tasks = []
    
    print("\nScanning directory and copying non-image files to format subdirectories...")
    
    for item in source_path.rglob("*"):
        if not item.is_file():
            continue
        
        relative_path = item.relative_to(source_path)
        
        if item.name.lower().endswith('.nii.gz'):
            base_name = item.name[:-7] 
            
            for fmt in img_formats:
                format_base_dir = target_path / fmt
                
                # Every target format is a single file per volume
                new_path = format_base_dir / relative_path.parent / f"{base_name}.{fmt}"
                new_path.parent.mkdir(parents=True, exist_ok=True)
                
                image_tasks.append((item, new_path, fmt))
        else:
            # Copy non-image files
            for fmt in img_formats:
                format_base_dir = target_path / fmt
                target_file = format_base_dir / relative_path
                target_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target_file)

    if image_tasks:
        print(f"\nStarting parallel conversion using {args.workers} threads...")
        success_count = 0
        
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_file = {
                executor.submit(process_med_image, src, dest, fmt): dest 
                for src, dest, fmt in image_tasks
            }
            
            for future in tqdm(as_completed(future_to_file), total=len(image_tasks), desc="Converting Scans"):
                success, result = future.result()
                if success:
                    success_count += 1
                else:
                    tqdm.write(f"Failed: {result}") 

        print(f"\nSuccessfully converted {success_count}/{len(image_tasks)} medical image files.")
    else:
        print("\nNo .nii.gz files found to convert.")

if __name__ == "__main__":
    main()