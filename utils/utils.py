import os
import subprocess

# Helper that drops the page cache (sync; echo 3 > /proc/sys/vm/drop_caches),
# allowed via sudoers without a password. Override with DROP_CACHES=/path.
DROP_CACHES = os.environ.get("DROP_CACHES", "/home/jbia/project-angel/p1/utils/drop_caches")


def clear_page_cache():
    print("Clearing Host Page Cache...", flush=True)
    if os.path.exists(DROP_CACHES):
        cmd = ["sudo", "-n", DROP_CACHES]
    else:
        cmd = ["sudo", "-n", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"]
    try:
        # -n: fail instead of prompting for a password
        subprocess.run(cmd, check=True)
        print("Cache Cleared.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Warning: Failed to clear cache. Set DROP_CACHES or check your sudoers file.")
