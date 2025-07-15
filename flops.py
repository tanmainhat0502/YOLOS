import torch
from thop import profile
import argparse
from models import build_model as build_yolos_model

def load_model(args, weight_path):
    # Build YOLOS model
    model, _, _ = build_yolos_model(args)

    # Load checkpoint
    checkpoint = torch.load(weight_path, map_location='cpu')

    # Tìm đúng key để load
    state_dict = checkpoint.get('model') or checkpoint.get('state_dict') or checkpoint
    model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to model .pth file')
    parser.add_argument('--dataset_file', type=str, default='coco', help='coco or custom')
    parser.add_argument('--img_size', type=int, default=640, help='Input image size (square)')
    parser.add_argument('--device', default='cuda', help='Device to use')
    parser.add_argument('--backbone_name', default='tiny', type=str)
    parser.add_argument('--det_token_num', default=100, type=int)
    parser.add_argument('--pre_trained', default='')
    parser.add_argument('--init_pe_size', nargs='+', type=int, default=[800, 1333])
    parser.add_argument('--mid_pe_size', nargs='+', type=int, default=[25, 42])
    parser.add_argument('--output_onnx', default='yolos.onnx', type=str, help='Output ONNX file path')
    parser.add_argument('--input_shape', nargs='+', type=int, default=[1, 3, 800, 1333])
    parser.add_argument('--use_checkpoint', action='store_true',
                        help='use checkpoint.checkpoint to save mem')
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    args = parser.parse_args()

    # Khởi tạo model
    model = load_model(args, args.model)

    # Tạo input giả với batch size 1, 3 channels, height = width = img_size
    dummy_input = torch.randn(1, 3, args.img_size, args.img_size)

    # Tính FLOPs và params
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)

    print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
    print(f"Params: {params / 1e6:.2f} M")

if __name__ == '__main__':
    main()
