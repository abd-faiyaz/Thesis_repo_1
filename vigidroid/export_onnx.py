import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from model.bytecnn import ByteCNN

def main():
    print("Initializing model...")
    model = ByteCNN(embed_dim=8, num_classes=2)
    pth_path = os.path.join(os.path.dirname(__file__), 'bytecnn_basemodel_2020.pth')
    onnx_path = os.path.join(os.path.dirname(__file__), 'bytecnn_basemodel_2020.onnx')

    print(f"Loading weights from {pth_path}...")
    state_dict = torch.load(pth_path, map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])    
    else:
        model.load_state_dict(state_dict)
    
    model.eval()
    print("Exporting ONNX model...")
    dummy_input = torch.zeros((1, 1024), dtype=torch.long)
    torch.onnx.export(
        model, dummy_input, onnx_path,
        export_params=True, opset_version=14, do_constant_folding=True,
        input_names=['input_bytes'], output_names=['output'],
        dynamic_axes={'input_bytes': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Model successfully saved to {onnx_path}")

if __name__ == '__main__':
    main()
