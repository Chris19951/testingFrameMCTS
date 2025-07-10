import torch
import tensorflow as tf
import numpy as np
from torch_agent import ConnectFour, ResNet_Torch
from tf_agent import ResNet_tf
from onnx_tf.backend import prepare
import onnx


game = ConnectFour()

# Dein trainiertes PyTorch-Modell initialisieren und laden
model = ResNet_Torch(game=ConnectFour(), num_resBlocks=9, num_hidden=128, device="cpu")
model.load_state_dict(torch.load("model_7_ConnectFour.pt", map_location="cpu"))
model.eval()

# Dummy-Eingabe (Shape muss deiner Encoded-State entsprechen: [1, 3, 6, 7])
dummy_input = game.get_initial_state()
dummy_input = game.get_encoded_state(dummy_input)
dummy_input = torch.tensor(dummy_input).unsqueeze(0)

# Export nach ONNX
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["policy", "value"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "policy": {0: "batch_size"},
        "value": {0: "batch_size"},
    },
    opset_version=13
)

