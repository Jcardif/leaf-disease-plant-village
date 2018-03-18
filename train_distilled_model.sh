python resnet18_run.py --arch 'resnet18' --dataseparate '20-80' --lr 0.0006 --has_weights='False' --distill 'True' && \
    python resnet18_run.py --arch 'resnet18' --dataseparate '40-60' --lr 0.0006 --has_weights='False' --distill 'True' && \
    python resnet18_run.py --arch 'resnet18' --dataseparate '50-50' --lr 0.0006 --has_weights='False' --distill 'True' && \
    python resnet18_run.py --arch 'resnet18' --dataseparate '60-40' --lr 0.0006 --has_weights='False' --distill 'True' && \
    python resnet18_run.py --arch 'resnet18' --dataseparate '80-20' --lr 0.0006 --has_weights='False' --distill 'True' 
python resnet18_run.py --arch 'resnet18' --dataseparate '10-90' --lr 0.0006 --has_weights='False' --distill 'MSETrue'
