python inference/benchmark_eval_time.py \
    --resume weights/yolos_6k_weight/checkpoint.pth \
    --coco_path F:/Pythera/EmoticGender6k_coco \
    --batch_size 1 \
    --eval_size 600 \
    --init_pe_size 512 864 \
    --mid_pe_size 512 864 \
    --backbone_name small_dWr