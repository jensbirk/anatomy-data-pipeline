import sys
import argparse
import time
import os
import shutil
import yaml
from pathlib import Path
import subprocess
from datetime import datetime
import pwd



NTFY_URL = "ntfy.sh/jbia" 

from system.system_monitor import SystemMonitor
from utils.utils import clear_page_cache
from system.loggers.config import ConfigLogger


def _user_scope_available(wait_s=180):
    deadline = time.time() + wait_s
    while True:
        try:
            r = subprocess.run(["systemd-run", "--user", "--scope", "--quiet", "--", "true"],
                               capture_output=True, text=True)
        except FileNotFoundError:
            return False
        if r.returncode == 0:
            return True
        if time.time() > deadline:
            print(f"   systemd-run --user failed ({r.returncode}): {r.stderr.strip()}")
            return False
        print("   systemd-run --user not ready, retrying in 10 s")
        time.sleep(10)


def _run_with_memory_limit(cmd, memory_limit_gb, script_rel, project_root):
    memory_high = f"{memory_limit_gb}G"
    print(f"   Memory limit enabled: high={memory_high} (no hard max, free to swap)")
    unit_name = "dio-" + script_rel.stem + "-" + str(int(time.time()))
    scope = ["--scope", "--property=MemoryHigh=" + memory_high, "--unit=" + unit_name, "--"]
    if _user_scope_available():
        prefix = ["systemd-run", "--user"] + scope
    else:
        print("   systemd-run --user unavailable, using sudo systemd-run")
        prefix = ["sudo", "-n", "systemd-run"] + scope
    subprocess.run(prefix + cmd, check=True, cwd=project_root)
    return True


def _record_exit(log_dir, returncode, timeout_min):
    if returncode == 0:
        reason = "ok"
    elif returncode == 124:
        reason = f"timeout after {timeout_min} min"
    elif returncode in (137, -9) or returncode == 128 + 9:
        reason = "killed (SIGKILL: out of memory or systemd-oomd)"
    else:
        reason = "failed"
    (Path(log_dir) / "exit_status.txt").write_text(f"returncode={returncode}\nreason={reason}\n")
    print(f"   Exit: {returncode} ({reason})")


def get_project_root():
    return Path(__file__).resolve().parent.parent

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def send_notification(message):
    try:
        subprocess.run(
            ["curl", "-d", message, NTFY_URL], 
            check=False, 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL
        )
        print("Notification sent.")
    except Exception as e:
        print(f"Failed to send notification: {e}")

def clean_cache_directory(cache_path_str):
    cache_path = Path(cache_path_str)
    print(cache_path)
    if not cache_path.is_dir() or str(cache_path) in ('/', '/tmp', '/home'):
        print(f"Skipping cache cleanup for non-directory or restricted path: {cache_path}")
        return

    print(f"Cleaning up cache directory: {cache_path}")
    try:
        if cache_path.exists() and any(cache_path.iterdir()):
            shutil.rmtree(cache_path)
        cache_path.mkdir(parents=True, exist_ok=True)
        print("Cache directory cleaned and recreated.")
    except Exception as e:
        print(f"Cache cleanup failed for {cache_path}: {e}")

def clear_cache_vmtouch(cache_path_str):
    print(f"Cleaning cache using vmtouch for: {cache_path_str}")
    try:
        subprocess.run(["vmtouch", "-e", cache_path_str], check=True)
        print("vmtouch cache eviction completed.")
    except Exception as e:
        print(f"vmtouch cache eviction failed for {cache_path_str}: {e}")


def code_version(project_root, snapshot_dir):
    def git(*args):
        return subprocess.run(["git", *args], cwd=project_root, capture_output=True,
                              text=True).stdout.strip()
    try:
        commit = git("rev-parse", "HEAD")
        diff = git("diff", "HEAD")
    except FileNotFoundError:
        return {}
    if diff:
        (Path(snapshot_dir) / "uncommitted.diff").write_text(diff + "\n")
    return {"git_commit": commit or None, "git_dirty": bool(diff)}


def run_batch_from_dir(experiment_dir_path, project_root):
    exp_dir = Path(experiment_dir_path).resolve()
    manifest_path = exp_dir / "manifest.yaml"
    metadata_path = exp_dir / "metadata.yml"

    if not manifest_path.exists():
        print(f"Error: manifest.yaml not found in {exp_dir}")
        return

    print(f"Reading manifest from: {manifest_path}")
    manifest = load_yaml(manifest_path)
    
    global_metadata = {}
    if metadata_path.exists():
        print(f"Loading global metadata from: {metadata_path}")
        global_metadata = load_yaml(metadata_path)
    
    runs = manifest.get('runs', [])
    if not runs:
        print("No 'runs' list found in manifest.yaml.")
        return

    print(f"Found {len(runs)} experiments.")
    
    print("\nRunning sanity checks...")
    errors = []

    for i, run in enumerate(runs):
        script_path = project_root / run['script']
        if not script_path.exists():
            errors.append(f"[Run {i+1}] Script missing: {script_path}")

        config_data = run.get('config')
        if not config_data or not isinstance(config_data, dict):
            errors.append(f"[Run {i+1}] Config is missing or not a dictionary in manifest.")

    if errors:
        print(f"\nFound {len(errors)} errors. Execution aborted.")
        for error in errors:
            print(error)
        return

    print("Sanity checks passed. Starting execution...\n")

    for i, run_config in enumerate(runs):
        base_name = run_config.get('name', f"run_{i+1}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        run_name = f"{base_name}_{timestamp}"
        run_dir = exp_dir / run_name
        
        code_snap_dir = run_dir / "code_snapshot"
        logs_dir = run_dir / "logs"
        
        os.makedirs(code_snap_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        
        current_run_metadata = global_metadata.copy()
        current_run_metadata.update({k: v for k, v in run_config.items() if k != 'config'})
        
        current_run_metadata['run_id'] = run_name
        current_run_metadata.update(code_version(project_root, code_snap_dir))

        with open(run_dir / "run_metadata.yaml", "w") as f:
            yaml.dump(current_run_metadata, f)
        
        generated_config_path = run_dir / "config.yaml"
        with open(generated_config_path, "w") as f:
            yaml.dump(run_config['config'], f, default_flow_style=False)
        
        print(f"\n--- Experiment {i+1}/{len(runs)}: {run_name} ---")
        print(f"Run Directory: {run_dir}")

        try:
            execute_experiment(
                script_path=run_config['script'], 
                config_path=generated_config_path, 
                project_root=project_root,
                run_output_dirs={
                    "root": run_dir,
                    "logs": logs_dir,
                    "snapshot": code_snap_dir
                },
                memory_limit_gb=run_config.get('memory_limit_gb'),
                timeout_min=run_config.get('timeout_min'),
            )
        except Exception as e:
            print(f"Experiment failed, continuing with next... ({e})")
        print("----------------------------")


def execute_experiment(script_path, config_path, project_root, run_output_dirs, memory_limit_gb=None,
                       timeout_min=None):
    script_rel = Path(script_path)
    config_full = Path(config_path) 
    config = load_yaml(config_full)
    
    log_dir = run_output_dirs["logs"]
    snapshot_dir = run_output_dirs["snapshot"]

    print(f"Script: {script_rel}")
    print(f"Config: {config_full}")

    # clean_cache_directory(config['output_dir']) 
    clear_page_cache() 
    # clear_cache_vmtouch("/dtu/p1/jebian/data/train/")
    # clear_cache_vmtouch("/dtu/p1/jebian/data/alpaca/arrow/")

    config_logger = ConfigLogger(
        config_path=config_full,
        log_dir=snapshot_dir, 
    )

    try:
        saved_snapshot = config_logger.save(
            extra_metadata={
                "script": str((project_root / script_rel).resolve()),
                "real_log_output": str(log_dir.resolve()),
            }
        )
        print(f"   Config/Code snapshot saved to {saved_snapshot}")
    except Exception as exc:
        print(f"Config logging failed: {exc}")
        raise

    try:
        print(f"Running experiment process...")
        with SystemMonitor(log_dir, process_identifier=script_rel.name):
            cmd = [
                sys.executable, 
                str(script_rel), 
                "--config", str(config_full),
                "--log_dir", str(log_dir)
            ]
            if timeout_min:
                # a run that cannot fit its memory budget may thrash in swap indefinitely
                cmd = ["timeout", "--kill-after=60", str(int(timeout_min * 60))] + cmd

            try:
                if memory_limit_gb is not None:
                    _run_with_memory_limit(cmd, memory_limit_gb, script_rel, project_root)
                else:
                    subprocess.run(cmd, check=True, cwd=project_root)
                _record_exit(log_dir, 0, timeout_min)
                send_notification(f"Experiment Done: {script_rel.stem}")
            except subprocess.CalledProcessError as e:
                _record_exit(log_dir, e.returncode, timeout_min)
                print(f"Failed: {script_rel}")
                send_notification(f"Experiment Failed: {script_rel.stem}")
                raise

    finally:
        print("\n--- Post-Run Cleanup ---")
        # clean_cache_directory(config['output_dir'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, help="Path to experiment directory containing manifest.yaml")
    
    parser.add_argument("--script", type=str, help="Single script to run")
    parser.add_argument("--config", type=str, help="Single config file path (legacy mode)")
    
    args, unknown = parser.parse_known_args()

    root_dir = get_project_root()
    print(f"Project Root detected as: {root_dir}")

    # repo root on PYTHONPATH
    os.environ["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(root_dir), os.environ.get("PYTHONPATH")) if p)

    if args.exp_dir:
        run_batch_from_dir(args.exp_dir, root_dir)
    elif args.script and args.config:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_log_dir = root_dir / "logs" / f"manual_run_{timestamp}"
        
        temp_dirs = {
            "root": temp_log_dir,
            "logs": temp_log_dir / "logs",
            "snapshot": temp_log_dir / "code_snapshot"
        }
        for d in temp_dirs.values(): os.makedirs(d, exist_ok=True)
        
        conf_path = Path(args.config)
        if not conf_path.is_absolute():
            conf_path = root_dir / conf_path

        execute_experiment(args.script, conf_path, root_dir, temp_dirs)
    else:
        print("Error: You must provide --exp_dir (containing manifest.yaml) OR --script + --config")