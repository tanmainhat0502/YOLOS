torchrun --nproc_per_node=2 --master_port=29500 \
    main.py \
    --coco_path /workspace/yolos/data4training \
    --batch_size 2 \
    --lr 2.5e-5 \
    --epochs 100 \
    --backbone_name small_dWr \
    --pre_trained /workspace/yolos/deit_s_dWr_300.pth \
    --eval_size 600 \
    --init_pe_size 512 864 \
    --mid_pe_size 512 864 \
    --output_dir run \
    --wandb_project YOlos-Detection \
    --wandb_name yolos_ver2
