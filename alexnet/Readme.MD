## How to run

# For python
Download weights to folder `weights`
```
python torch_2_onnx.py && pytorch onnx_2_trt.py
```

# For c++
```
python get_wts_format.py && mkdir build && cd build && cmake .. && make -j2
```