from face_binary_net import MobilenetBinaryNet
import torch
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

model = MobilenetBinaryNet(pretrained=False)
# Load checkpoint from the model directory to avoid relative path issues
ckpt_path = os.path.join(script_dir, "face_binary.pth")
model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

output_path = os.path.join(script_dir, "face_binary.onnx")

torch.onnx.export(
    model,
    dummy_input,
    output_path,
    input_names=["input"],
    output_names=["logits"],
    opset_version=18,
    export_params=True,  # Ensure parameters are embedded
    do_constant_folding=True,
    dynamic_axes={
        "input": {0: "batch"},
        "logits": {0: "batch"},
    },
)

print(f"Model exported to: {output_path}")


data_file = output_path + ".data"
if os.path.exists(data_file):
    print(f"Model has external data file: {data_file}")



# copy the model and weights to "../app/public/models" for easy access from the web app
import shutil

dest_dir = os.path.join(script_dir, "..", "app", "public", "models")
os.makedirs(dest_dir, exist_ok=True)
shutil.copy2(output_path, os.path.join(dest_dir, "face_binary.onnx"))
if os.path.exists(data_file):
    shutil.copy2(data_file, os.path.join(dest_dir, "face_binary.onnx.data"))

print(f"Model and weights copied to: {dest_dir}")