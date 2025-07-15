import torch
import argparse
import models
from models import build_model as build_yolos_model

def get_args_parser():
    parser = argparse.ArgumentParser('YOLOS ONNX Exporter', add_help=False)
    parser.add_argument('--device', default='cuda', help='Device to use')
    parser.add_argument('--backbone_name', default='tiny', type=str)
    parser.add_argument('--det_token_num', default=100, type=int)
    parser.add_argument('--pre_trained', default='')
    parser.add_argument('--init_pe_size', nargs='+', type=int, default=[800, 1333])
    parser.add_argument('--mid_pe_size', nargs='+', type=int, default=[25, 42])
    parser.add_argument('--output_onnx', default='yolos.onnx', type=str, help='Output ONNX file path')
    parser.add_argument('--input_shape', nargs='+', type=int, default=[1, 3, 800, 1333])
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to YOLOS .pth checkpoint')
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
    return parser

def main(args):
    args.dataset_file = 'coco'
    device = torch.device(args.device)
    
    # 1. Load YOLOS model
    model, _, _ = build_yolos_model(args)
    model.to(device)

    # 2. Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    print("Model loaded. Converting to ONNX...")

    # 3. Dummy input
    dummy_input = torch.randn(*args.input_shape).to(device)

    # 4. Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        args.output_onnx,
        input_names=['input'],
        output_names=['output'],
        opset_version=12,
        export_params=True,
        do_constant_folding=True,
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"✅ ONNX model exported to: {args.output_onnx}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser('YOLOS ONNX Export Script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

exit()
# B2 conver sang FB 16 - pip install onnxconverter-common
import onnx
from onnxconverter_common.float16 import convert_float_to_float16

model_fp32 = onnx.load("run/eval/yolos16.onnx")
model_fp16 = convert_float_to_float16(model_fp32, keep_io_types=True)
onnx.save(model_fp16, "run/eval/yolos32.onnx")
exit()