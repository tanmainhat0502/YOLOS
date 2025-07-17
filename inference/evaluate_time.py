import torch
import time
from util import misc as utils
from datasets.coco_eval import CocoEvaluator

@torch.no_grad()
def evaluate_timed(model, criterion, postprocessors, data_loader, base_ds, device):
    model.eval()
    criterion.eval()

    preprocess_times = []
    predict_times = []
    postprocess_times = []

    coco_evaluator = CocoEvaluator(base_ds, iou_types=('bbox',))

    for samples, targets in data_loader:
        # ==== Preprocess ====
        start_pre = time.time()
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_pre = time.time()

        # ==== Inference ====
        start_model = time.time()
        outputs = model(samples)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_model = time.time()

        # ==== Postprocess ====
        start_post = time.time()
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors["bbox"](outputs, orig_target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        coco_evaluator.update(res)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_post = time.time()

        # Lưu thời gian
        preprocess_times.append(end_pre - start_pre)
        predict_times.append(end_model - start_model)
        postprocess_times.append(end_post - start_post)

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    print(f"Device: {device}")
    print(f"Average Preprocess time: {1000 * sum(preprocess_times)/len(preprocess_times):.2f} ms")
    print(f"Average Model Predict time: {1000 * sum(predict_times)/len(predict_times):.2f} ms")
    print(f"Average Postprocess time: {1000 * sum(postprocess_times)/len(postprocess_times):.2f} ms")

    return {
        'preprocess': sum(preprocess_times)/len(preprocess_times),
        'predict': sum(predict_times)/len(predict_times),
        'postprocess': sum(postprocess_times)/len(postprocess_times),
        'total': sum(preprocess_times)/len(preprocess_times) +
                 sum(predict_times)/len(predict_times) +
                 sum(postprocess_times)/len(postprocess_times),
    }
