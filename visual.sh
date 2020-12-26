export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=/media/f511/Wenjie/swart/test-fcos:$PYTHONPATH

# image=785.jpg
# bbox=[42,285,382,496]
# cat=0

# image=46804.jpg
# bbox=[53,222,367,463]
# cat=18

image=285.jpg
bbox=[72,3,628,584]
cat=21

# python visualer/visual.py \
#     --config-file configs/fcos/fcos_R_50_FPN_1x.yaml \
#     --image visualer/$image \
#     --bbox $bbox \
#     --cat $cat \
#     MODEL.WEIGHT training_dir/resnet50cen/model_0085000.pth \
#     TEST.IMS_PER_BATCH 10

python visualer/visual.py \
    --config-file configs/fcos/fcos_R_50_FPN_1x.yaml \
    MODEL.WEIGHT training_dir/resnet50cen/model_0085000.pth \
    TEST.IMS_PER_BATCH 10