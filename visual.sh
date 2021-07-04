set CUDA_VISIBLE_DEVICES=0
set PYTHONPATH=.;$PYTHONPATH$

set config=fcos/fcos_R_50_FPN_1x.yaml

set weight=50ori/model_0090000.pth

set count=1

python visualer/visual.py ^
    --config-file configs/%config% ^
    MODEL.WEIGHT training_dir/%weight% ^
    TEST.IMS_PER_BATCH %count%