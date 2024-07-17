# MLP
Source: https://github.com/wang-xinyu/tensorrtx/tree/master/mlp

## Python
Build engine
```
python build_engine.py
```
Infer 
```
python infer.py
```

## C++
Build engine
```
mkdir build && cd build && cmake .. && make && ./build_engine
```
Infer 
```
cd build && ./build_engine
```