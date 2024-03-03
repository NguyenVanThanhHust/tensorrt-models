# How to run

Download model
```
python alex_pytorch.py
```
Model woulde be save in `/root/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth`

## For python
Convert
```
python torch_2_onnx.py && python onnx_2_trt.py -m alexnet
```

## For c++
```
python get_wts_format.py && mkdir build && cd build && cmake .. && make -j2
```

## Original
```
python trt_x.py -s
```