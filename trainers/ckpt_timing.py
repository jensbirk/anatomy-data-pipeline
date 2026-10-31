import csv
import os
import time

import torch
import torch.distributed.checkpoint as dcp


class CheckpointTimer:
    def __init__(self, args):
        self.enabled = bool(getattr(args, "checkpoint_timing", False))
        self.warmup = int(getattr(args, "timing_warmup_steps", 20) or 0)
        self.use_async = bool(getattr(args, "async_checkpoint", False))
        self.output_dir = getattr(args, "output_dir", None)
        log_dir = getattr(args, "log_dir", None)
        self.csv_path = os.path.join(log_dir, "checkpoint_timing.csv") if log_dir else None
        self.rows = []
        self.last = None
        self.pending = None  # (future, call_time, row)

    @staticmethod
    def _now():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    def should_save(self, step, save_steps):
        if not save_steps or save_steps <= 0:
            return False
        if not self.enabled:
            return step % save_steps == 0
        return step > self.warmup and (step - self.warmup) % save_steps == 0

    def step_done(self, step):
        if not self.enabled:
            return
        now = self._now()
        if self.last is not None:
            self.rows.append({"kind": "step", "step": step, "seconds": now - self.last,
                              "wait_prev_s": "", "write_s": ""})
        self.last = now

    def _wait_pending(self):
        if self.pending is None:
            return 0.0
        future, call_time, row = self.pending
        t0 = time.perf_counter()
        future.result()
        done = time.perf_counter()
        # write_s is filled by the callback; fall back to the observed time
        if row["write_s"] == "":
            row["write_s"] = done - call_time
        self.pending = None
        return done - t0

    def save(self, state, step):
        os.makedirs(self.output_dir, exist_ok=True)
        t0 = self._now()
        row = {"kind": "save", "step": step, "seconds": "", "wait_prev_s": "", "write_s": ""}
        if self.use_async:
            row["wait_prev_s"] = self._wait_pending()
            call_time = time.perf_counter()
            future = dcp.async_save(state_dict=state,
                                    storage_writer=dcp.FileSystemWriter(self.output_dir))
            future.add_done_callback(
                lambda f, r=row, c=call_time: r.__setitem__("write_s", time.perf_counter() - c))
            self.pending = (future, call_time, row)
        else:
            row["wait_prev_s"] = 0.0
            path = os.path.join(self.output_dir, "checkpoint.pt")
            with open(path, "wb") as f:
                torch.save(state, f)
                f.flush()
                os.fsync(f.fileno())
        t1 = self._now()
        row["seconds"] = t1 - t0
        self.rows.append(row)
        self.last = t1  # the next step's time starts after the save

    def finish(self):
        if not self.enabled:
            return
        self._wait_pending()
        if self.csv_path:
            with open(self.csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["kind", "step", "seconds", "wait_prev_s", "write_s"])
                w.writeheader()
                w.writerows(self.rows)
