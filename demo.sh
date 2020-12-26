export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:/media/f511/c3b81025-f61b-4168-a999-00e2b8629aac/f511/swart/FCOS

weight=zplan22/deformable/model_final.pth 
# weight=zplan22/deformable_stride4_numconvs2/model_0003000.pth

python demo/fcos_demo.py \
     --config-file configs/fcos/fcos_R_50_FPN_1x.yaml \
     --weights $weight


