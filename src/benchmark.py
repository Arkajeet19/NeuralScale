"""
Benchmarks real inference speed across backends and resolutions.

Backends compared:
  - PyTorch FP32 (full precision)    -- your current baseline
  - PyTorch FP16 (half precision)    -- halves memory bandwidth, usually
                                         close to free speedup on modern GPUs
  - ONNX Runtime (CPU)                -- for reference / non-GPU deployment
  - ONNX Runtime (CUDA), if available -- the "real-time deployment" backend

Resolution presets approximate realistic DLSS-style scenarios: render at a
lower internal resolution, upscale to a higher display resolution.

Run from the project root (after export_onnx.py has produced models/model.onnx):
    py src/benchmark.py
"""
import os
import sys
import time
import csv
import torch
import numpy as np

sys.path.append(os.path.dirname(__file__))
from model import EDSRBaseline

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "gan_generator.pth")
ONNX_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model.onnx")
RESULTS_CSV = os.path.join(os.path.dirname(__file__), "..", "outputs", "benchmark_results.csv")

RESOLUTIONS = [
    ("240p -> 960p",  240, 135),
    ("320p -> 1280p", 320, 180),
    ("480p -> 1920p", 480, 270),
]

WARMUP_RUNS = 10
TIMED_RUNS = 50


def benchmark_pytorch(model, device, dtype, w, h):
    x = torch.randn(1, 3, h, w, device=device, dtype=dtype)
    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(TIMED_RUNS):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - t0

    avg_ms = (elapsed / TIMED_RUNS) * 1000
    return avg_ms


def benchmark_onnx(session, w, h, use_cuda=False):
    x = np.random.randn(1, 3, h, w).astype(np.float32)

    if not use_cuda:
        # CPU provider: plain session.run is fine, no device transfer involved
        for _ in range(WARMUP_RUNS):
            session.run(None, {"input": x})
        t0 = time.time()
        for _ in range(TIMED_RUNS):
            session.run(None, {"input": x})
        elapsed = time.time() - t0
        return (elapsed / TIMED_RUNS) * 1000

    # CUDA provider: use IO Binding to keep input/output on the GPU across
    # calls. Without this, session.run() silently copies the input from
    # CPU to GPU and the output back on EVERY call, which dominates (and
    # badly skews) timing at larger resolutions.
    try:
        io_binding = session.io_binding()
        x_ortvalue = ort.OrtValue.ortvalue_from_numpy(x, "cuda", 0)
        io_binding.bind_ortvalue_input("input", x_ortvalue)
        io_binding.bind_output("output", "cuda")

        for _ in range(WARMUP_RUNS):
            session.run_with_iobinding(io_binding)

        t0 = time.time()
        for _ in range(TIMED_RUNS):
            session.run_with_iobinding(io_binding)
        elapsed = time.time() - t0
        return (elapsed / TIMED_RUNS) * 1000

    except RuntimeError as e:
        # Some providers (e.g. TensorRT-RTX) don't support explicit CUDA
        # OrtValue allocation -- they manage GPU memory internally instead.
        # Fall back to plain session.run(); note this means timing includes
        # host<->device transfer overhead for this provider specifically,
        # so it's not a perfectly apples-to-apples number against the
        # IO-bound CUDA results, but it's the best available measurement
        # without that provider's own internal binding API.
        print(f"    (IO binding not supported by this provider, falling back to session.run: {e})")
        for _ in range(WARMUP_RUNS):
            session.run(None, {"input": x})
        t0 = time.time()
        for _ in range(TIMED_RUNS):
            session.run(None, {"input": x})
        elapsed = time.time() - t0
        return (elapsed / TIMED_RUNS) * 1000


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type != "cuda":
        print("WARNING: no CUDA device found -- GPU benchmarks will be skipped.")

    model_fp32 = EDSRBaseline(scale=4, num_blocks=16, channels=64).to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model_fp32.load_state_dict(ckpt["model_state"])
    model_fp32.eval()

    model_fp16 = None
    if device.type == "cuda":
        model_fp16 = EDSRBaseline(scale=4, num_blocks=16, channels=64).to(device).half()
        model_fp16.load_state_dict(ckpt["model_state"])
        model_fp16 = model_fp16.half()
        model_fp16.eval()

    onnx_cpu_session = None
    onnx_cuda_session = None
    onnx_trt_session = None
    if os.path.exists(ONNX_PATH):
        global ort
        import onnxruntime as ort
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()  # needed by onnxruntime-trt-rtx to load its bundled runtime

        onnx_cpu_session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
        available_providers = ort.get_available_providers()
        print(f"Available ONNX Runtime providers: {available_providers}")

        if "CUDAExecutionProvider" in available_providers:
            onnx_cuda_session = ort.InferenceSession(ONNX_PATH, providers=["CUDAExecutionProvider"])

        # detect the TensorRT-RTX provider by name rather than hardcoding a
        # guessed string, since this is newer tooling and the exact name
        # can vary by package version
        trt_provider = next((p for p in available_providers if "tensorrt" in p.lower()), None)
        if trt_provider:
            print(f"Found TensorRT provider: {trt_provider}")
            onnx_trt_session = ort.InferenceSession(ONNX_PATH, providers=[trt_provider])
        else:
            print("No TensorRT provider found -- install onnxruntime-trt-rtx for TensorRT-RTX benchmarks.")

        if not onnx_cuda_session and not onnx_trt_session:
            print("No GPU ONNX provider available -- install onnxruntime-gpu or onnxruntime-trt-rtx.")
    else:
        print(f"No ONNX model found at {ONNX_PATH} -- run src/export_onnx.py first for ONNX benchmarks.")

    results = []
    for label, w, h in RESOLUTIONS:
        print(f"\n--- {label} (input {w}x{h}) ---")

        ms = benchmark_pytorch(model_fp32, device, torch.float32, w, h)
        fps = 1000 / ms
        print(f"  PyTorch FP32:      {ms:7.2f} ms/frame  ({fps:6.1f} FPS)")
        results.append([label, "PyTorch FP32", round(ms, 2), round(fps, 1)])

        if model_fp16 is not None:
            ms = benchmark_pytorch(model_fp16, device, torch.float16, w, h)
            fps = 1000 / ms
            print(f"  PyTorch FP16:      {ms:7.2f} ms/frame  ({fps:6.1f} FPS)")
            results.append([label, "PyTorch FP16", round(ms, 2), round(fps, 1)])

        if onnx_cpu_session is not None:
            ms = benchmark_onnx(onnx_cpu_session, w, h, use_cuda=False)
            fps = 1000 / ms
            print(f"  ONNX Runtime CPU:  {ms:7.2f} ms/frame  ({fps:6.1f} FPS)")
            results.append([label, "ONNX Runtime CPU", round(ms, 2), round(fps, 1)])

        if onnx_cuda_session is not None:
            ms = benchmark_onnx(onnx_cuda_session, w, h, use_cuda=True)
            fps = 1000 / ms
            print(f"  ONNX Runtime CUDA: {ms:7.2f} ms/frame  ({fps:6.1f} FPS)")
            results.append([label, "ONNX Runtime CUDA", round(ms, 2), round(fps, 1)])

        if onnx_trt_session is not None:
            ms = benchmark_onnx(onnx_trt_session, w, h, use_cuda=True)
            fps = 1000 / ms
            print(f"  ONNX Runtime TensorRT-RTX: {ms:7.2f} ms/frame  ({fps:6.1f} FPS)")
            results.append([label, "ONNX Runtime TensorRT-RTX", round(ms, 2), round(fps, 1)])

    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["resolution", "backend", "ms_per_frame", "fps"])
        writer.writerows(results)
    print(f"\nResults saved to {RESULTS_CSV}")


if __name__ == "__main__":
    main()