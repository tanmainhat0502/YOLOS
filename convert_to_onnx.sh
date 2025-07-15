python convert_to_onnx.py \
    --checkpoint run/checkpoint.pth \
    --output_onnx run/eval/yolos.onnx \
    --backbone_name small_dWr \
    --init_pe_size 512 864 \
    --mid_pe_size 512 864 