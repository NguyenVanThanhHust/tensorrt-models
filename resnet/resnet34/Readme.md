# Custom resnet 34
Get trained pytorch model from [image-classification](https://github.com/NguyenVanThanhHust/image-classification)

Get .wts file
```
python get_wts.py
```

Build the engine
```
python build_engine.py
```

Run program
```
python infer.py
```

To build with c++
run command
```
mkdir build && cd build && cmake .. && make && ./build_engine
```
To run with c++
```
./infer
```

To build with python 
```
python build_engine.py
```

Current infer with python is broken because I can't install pycuda yet. So use c++ infer instead