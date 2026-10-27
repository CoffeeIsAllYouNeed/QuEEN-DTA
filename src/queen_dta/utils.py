"""Runtime helpers: deterministic settings, seeding, GPU and memory handling, keep-awake."""
import contextlib
import ctypes
import gc
import os
import subprocess
import sys
import threading
import time

import numpy as np
import psutil
import torch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
if os.name != "nt":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def release_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def release_gpu():
    me = psutil.Process()
    protected = {me.pid} | {p.pid for p in me.parents()}
    try:
        listing = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader",
            ],
            capture_output=True, text=True, timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        listing = ""
    for token in listing.split():
        if not token.isdigit() or int(token) in protected:
            continue
        try:
            other = psutil.Process(int(token))
            if "python" in other.name().lower():
                print(f"Stopping GPU process {other.pid} ({other.name()})")
                other.kill()
                other.wait(timeout=30)
        except psutil.Error:
            pass
    release_memory()
    if torch.cuda.is_available():
        torch.cuda.ipc_collect()
        free, total = torch.cuda.mem_get_info()
        print(f"GPU memory free: {free / 2**30:.1f} of {total / 2**30:.1f} GB")


def check_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU detected by PyTorch. Install the CUDA build of "
            "PyTorch (pip install torch --index-url "
            "https://download.pytorch.org/whl/cu126), update the NVIDIA "
            "driver, then restart the kernel."
        )
    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    print(
        f"PyTorch GPU: {props.name} | "
        f"{props.total_memory / 2**30:.1f} GB | CUDA {torch.version.cuda}"
    )


class KeepAwake:
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ES_DISPLAY_REQUIRED = 0x00000002

    def __init__(self):
        self.stop_event = threading.Event()
        self.thread = None
        self.process = None

    def hold_windows(self):
        kernel = ctypes.windll.kernel32
        kernel.SetThreadExecutionState(
            self.ES_CONTINUOUS
            | self.ES_SYSTEM_REQUIRED
            | self.ES_DISPLAY_REQUIRED
        )
        self.stop_event.wait()
        kernel.SetThreadExecutionState(self.ES_CONTINUOUS)

    def start(self):
        self.stop_event.clear()
        if os.name == "nt":
            self.thread = threading.Thread(
                target=self.hold_windows, daemon=True
            )
            self.thread.start()
            print("Keep-awake on: the computer will not sleep and the "
                  "screen will stay on until the run ends")
            return
        if sys.platform == "darwin":
            command = ["caffeinate", "-dims", "-w", str(os.getpid())]
        else:
            command = [
                "systemd-inhibit", "--what=idle:sleep",
                "--why=model training", "sleep", "infinity",
            ]
        try:
            self.process = subprocess.Popen(
                command, stderr=subprocess.DEVNULL
            )
        except OSError:
            print("WARNING: could not block sleep on this system")
            return
        time.sleep(0.5)
        if self.process.poll() is not None:
            self.process = None
            print("WARNING: could not block sleep on this system")
            return
        print("Keep-awake on: sleep is blocked until the run ends")

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5)
            self.thread = None
        if self.process is not None:
            self.process.terminate()
            self.process = None
        print("Keep-awake off")


@contextlib.contextmanager
def keep_awake():
    guard = KeepAwake()
    guard.start()
    try:
        yield guard
    finally:
        guard.stop()
