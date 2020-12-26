export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export PYTHONPATH=.:$PYTHONPATH
# python tools/test_net.py \
#     --config-file configs/fcos/fcos_R_50_FPN_1x.yaml \
#     MODEL.WEIGHT zplan1/fcos_R_50_FPN_1x/model_0095000.pth \
#     TEST.IMS_PER_BATCH 16  

# config=fcos_R_50_FPN_1x.yaml
# config=fcos_R_101_FPN_2x.yaml
config=fcos_imprv_dcnv2_R_101_FPN_2x.yaml

# weight=training_dir/resnet50ori/model_0$1.pth
# weight=training_dir/resnet50cen/model_0$1.pth
# weight=training_dir/resnet101/model_0$1.pth
weight=training_dir/resnet101dcn/model_0$1.pth

python -m torch.distributed.launch \
    --nproc_per_node=4\
    --master_port=$((RANDOM + 10000)) \
    tools/test_net.py \
    --config-file configs/fcos/$config \
    MODEL.WEIGHT $weight \
    TEST.IMS_PER_BATCH 16

    # MODEL.WEIGHT zplan4/temp/model_0015000.pth \

# python -m torch.distributed.launch \
#     --nproc_per_node=4 \
#     --master_port=$((RANDOM + 10000)) \
#     tools/test_net.py \
#     --config-file configs/fcos/fcos_R_50_FPN_1x.yaml \
#     MODEL.WEIGHT demo/FCOS_R_50_FPN_1x.pth \
#     TEST.IMS_PER_BATCH 32

# 50ori 100000 or 92500
# 50cen 97500
# 101cen 185000
# 101dcn 185000


# 180000
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.441
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.619
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.480
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.273
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.484
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.568
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.350
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.576
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.615
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.425
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.664
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.775


