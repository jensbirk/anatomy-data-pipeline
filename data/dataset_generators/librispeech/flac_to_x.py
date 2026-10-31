import os
import shutil
import argparse
from pathlib import Path
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def process_audio(flac_path, target_path, target_format, bitrate):
    try:
        audio = AudioSegment.from_file(flac_path, format="flac")
        if target_format == "wav":
            audio.export(target_path, format="wav")
        else:
            audio.export(target_path, format=target_format, bitrate=bitrate)
        return True, target_path.name
    except Exception as e:
        return False, f"{flac_path.name} (Error: {e})"

def main():
    parser = argparse.ArgumentParser(description="Mirror a directory and convert FLAC files to MP3, WAV, and/or Arrow in subdirectories.")
    parser.add_argument("-s", "--source", required=True, help="Path to the source directory containing FLAC files.")
    parser.add_argument("-t", "--target", required=True, help="Path to the base target directory.")
    parser.add_argument("-f", "--format", choices=["mp3", "wav", "arrow", "all"], default="mp3", 
                        help="Choose output format: mp3, wav, arrow, or all. (Default: mp3)")
    parser.add_argument("-b", "--bitrate", default="320k", help="Bitrate for MP3 conversion (Default: 320k).")
    parser.add_argument("-w", "--workers", type=int, default=4, 
                        help="Number of parallel processing threads (Defaults to CPU cores).")
    
    args = parser.parse_args()
    
    source_path = Path(args.source)
    target_path = Path(args.target)
    
    if not source_path.exists() or not source_path.is_dir():
        print(f"Error: Source directory '{args.source}' does not exist.")
        return

    # 1. Determine which formats to build
    audio_formats = []
    create_arrow = False
    
    if args.format == "all":
        audio_formats = ["mp3", "wav"]
        create_arrow = True
    elif args.format == "arrow":
        audio_formats = ["wav"] # Arrow requires WAVs to be built first
        create_arrow = True
    else:
        audio_formats = [args.format]

    audio_tasks = []
    
    print("\nScanning directory and copying non-audio files to format subdirectories...")
    
    # 2. Traverse and route to format-specific subdirectories
    for item in source_path.rglob("*"):
        if not item.is_file():
            continue
        
        relative_path = item.relative_to(source_path)
        
        if item.suffix.lower() == '.flac':
            for fmt in audio_formats:
                # E.g., target_dir/mp3/Artist/Album/Song.mp3
                format_base_dir = target_path / fmt
                new_file = format_base_dir / relative_path.with_suffix(f'.{fmt}')
                new_file.parent.mkdir(parents=True, exist_ok=True)
                audio_tasks.append((item, new_file, fmt))
        else:
            # Copy non-audio files (txt, jpg) into each format's subdirectory tree
            for fmt in audio_formats:
                format_base_dir = target_path / fmt
                target_file = format_base_dir / relative_path
                target_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target_file)

    # 3. Parallel Audio Processing
    if audio_tasks:
        print(f"\nStarting parallel conversion using {args.workers} threads...")
        success_count = 0
        
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_file = {
                executor.submit(process_audio, src, dest, fmt, args.bitrate): dest 
                for src, dest, fmt in audio_tasks
            }
            
            for future in tqdm(as_completed(future_to_file), total=len(audio_tasks), desc="Converting Audio"):
                success, result = future.result()
                if success:
                    success_count += 1
                else:
                    tqdm.write(f"Failed: {result}") 

        print(f"\nSuccessfully converted {success_count}/{len(audio_tasks)} audio files.")
    else:
        print("\nNo FLAC files found to convert.")

    # 4. Package into Arrow Database
    if create_arrow:
        print("\nPackaging files into an Apache Arrow dataset...")
        try:
            from datasets import load_dataset
            import logging
            logging.getLogger("datasets").setLevel(logging.ERROR)
            
            # Read from the generated WAV subdirectory
            wav_dir = target_path / "wav"
            # Save to a dedicated Arrow subdirectory
            arrow_dir = target_path / "arrow"
            
            dataset = load_dataset("audiofolder", data_dir=str(wav_dir))
            dataset.save_to_disk(str(arrow_dir))
            
            print(f"Arrow dataset successfully created at: {arrow_dir}")
        except ImportError:
            print("Error: The 'datasets' library is missing. Please run 'pip install datasets soundfile'.")
        except Exception as e:
            print(f"Error creating Arrow dataset: {e}")

if __name__ == "__main__":
    main()