CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.run \
    --nproc_per_node=auto \
     --master_port 6666 \
    main.py \
        --method-name image_text_co_decomposition \
        --resume output/checkpoint.pth \
        --eval \
