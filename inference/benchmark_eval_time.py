import argparse
import torch
from torch.utils.data import DataLoader, SequentialSampler
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets import build_dataset, get_coco_api_from_dataset
from models import build_model as build_yolos_model
from evaluate_time import evaluate_timed  
from util.misc import collate_fn
import util.misc as utils

def run(device_str):
    device = torch.device(device_str)

    from main import get_args_parser
    parser = argparse.ArgumentParser('Inference Time Evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    # if isinstance(args.eval_size, int):
    #     args.eval_size = (args.eval_size, args.eval_size)

    # if not hasattr(args, 'init_pe_size') or args.init_pe_size is None:
    #     args.init_pe_size = args.eval_size

    # Override thiết bị
    args.device = device_str

    print(f"\n=== Running on {device_str.upper()} ===")

    utils.init_distributed_mode(args)

    model, criterion, postprocessors = build_yolos_model(args)
    model.to(device)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])

    dataset_val = build_dataset(image_set='val', args=args)
    sampler_val = SequentialSampler(dataset_val)
    data_loader_val = DataLoader(dataset_val, batch_size=args.batch_size, sampler=sampler_val,
                                 collate_fn=collate_fn, drop_last=False, num_workers=args.num_workers)

    base_ds = get_coco_api_from_dataset(dataset_val)

    times = evaluate_timed(model, criterion, postprocessors, data_loader_val, base_ds, device)
    print(f"[{device_str.upper()}] Total avg time per batch: {times['total']:.4f} s\n")

if __name__ == "__main__":
    print("Benchmark on GPU:")
    run("cuda")

    print("Benchmark on CPU:")
    run("cpu")
