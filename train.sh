export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=.:$PYTHONPATH

# 50
python -m torch.distributed.launch \
        --nproc_per_node=4 \
        --master_port=$((RANDOM + 10000)) \
        tools/train_net.py \
        --config-file configs/fcos/fcos_R_50_FPN_1x.yaml \
        DATALOADER.NUM_WORKERS 8 \
        SOLVER.IMS_PER_BATCH 16 \
        SOLVER.MAX_ITER 120000 \
        OUTPUT_DIR training_dir/resnet50ori


# 101
# python -m torch.distributed.launch \
#         --nproc_per_node=4 \
#         --master_port=$((RANDOM + 10000)) \
#         tools/train_net.py \
#         --config-file configs/fcos/fcos_R_101_FPN_2x.yaml \
#         DATALOADER.NUM_WORKERS 8 \
#         SOLVER.IMS_PER_BATCH 16 \
#         SOLVER.MAX_ITER 200000 \
#         OUTPUT_DIR training_dir/resnet101


# xt 64
# /media/f511/c3b81025-f61b-4168-a999-00e2b8629aac/f511/swart/fdosd/configs/fcos/fcos_imprv_dcnv2_X_101_64x4d_FPN_2x.yaml
# python -m torch.distributed.launch \
#         --nproc_per_node=4 \
#         --master_port=$((RANDOM + 10000)) \
#         tools/train_net.py \
#         --config-file configs/fcos/fcos_imprv_dcnv2_R_101_FPN_2x.yaml \
#         DATALOADER.NUM_WORKERS 8 \
#         SOLVER.IMS_PER_BATCH 16 \
#         SOLVER.MAX_ITER 200000 \
#         OUTPUT_DIR training_dir/resnet101dcn