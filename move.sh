if [ $2 -eq 1 ]
then 
    cp $1/backbone.py fcos_core/modeling/backbone
    cp $1/decouple_fpn.py fcos_core/modeling/backbone
    cp $1/resnet.py fcos_core/modeling/backbone
    cp $1/fcos.py fcos_core/modeling/rpn/fcos
    cp $1/layer_misc.py fcos_core/layers
    cp $1/defpn_misc.py fcos_core/layers
    cp $1/defaults.py fcos_core/config
elif [ $2 -eq 2 ]
then
    cp fcos_core/modeling/backbone/backbone.py $1
    cp fcos_core/modeling/backbone/decouple_fpn.py $1
    cp fcos_core/modeling/backbone/resnet.py $1
    cp fcos_core/modeling/rpn/fcos/fcos.py $1
    cp fcos_core/layers/layer_misc.py $1
    cp fcos_core/layers/defpn_misc.py $1
    cp fcos_core/config/defaults.py $1
elif [ $2 -eq 3 ]
then 
    mkdir $1
    cp $3/backbone.py $1
    cp $3/decouple_fpn.py $1
    cp $3/resnet.py $1
    cp $3/fcos.py $1
    cp $3/layer_misc.py $1
    cp $3/defpn_misc.py $1
    cp $3/defaults.py $1
else
    echo "not implementation!"
fi
