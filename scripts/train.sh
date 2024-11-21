CUDA_VISIBLE_DEVICES=6 python -m torch.distributed.run \
    --nproc_per_node=auto \
     --master_port 6666 \
    main.py \
        --cfg ./configs/experiment_configs.yml \
        --method-name image_text_co_decomposition
