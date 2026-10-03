"""
Exports the trained generator to ONNX format.

Why ONNX: PyTorch's standard execution has overhead from Python's dynamic
graph interpretation on every forward pass. ONNX Runtime instead runs a
pre-compiled, statically-optimized version of the same computation graph --
operator fusion, memory planning, and hardware-specific kernels done once
ahead of time rather than re-interpreted every call. This is the same
category of optimization real-time inference systems (including what
NVIDIA does for DLSS) rely on: PyTorch is great for research/training
flexibility, not for squeezing out every millisecond at inference time.

Run from the project root:
    py src/export_onnx.py
"""
import os
import sys
import torch

sys.path.append(os.path.dirname(__file__))
from model import EDSRBaseline

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "gan_generator.pth")
ONNX_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model.onnx")
SCALE = 4


def main():
    device = torch.device("cpu")  # export on CPU -- avoids any device-specific op quirks
    model = EDSRBaseline(scale=SCALE, num_blocks=16, channels=64).to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # dummy input -- shape doesn't matter much since we declare dynamic_axes below,
    # but height/width need to be consistent with the model's actual receptive field
    dummy_input = torch.randn(1, 3, 128, 128)

    torch.onnx.export(
        model,
        dummy_input,
        ONNX_OUTPUT_PATH,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        },
        opset_version=17,
    )
    print(f"Exported to {ONNX_OUTPUT_PATH}")

    # quick sanity check: verify ONNX Runtime output matches PyTorch output
    import onnxruntime as ort
    import numpy as np

    session = ort.InferenceSession(ONNX_OUTPUT_PATH, providers=["CPUExecutionProvider"])
    test_input = np.random.randn(1, 3, 64, 64).astype(np.float32)

    with torch.no_grad():
        torch_output = model(torch.from_numpy(test_input)).numpy()
    onnx_output = session.run(None, {"input": test_input})[0]

    max_diff = np.abs(torch_output - onnx_output).max()
    print(f"Max difference between PyTorch and ONNX output: {max_diff:.6f}")
    print("(should be a very small number, e.g. < 1e-4 -- confirms the export is correct)")


if __name__ == "__main__":
    main()
