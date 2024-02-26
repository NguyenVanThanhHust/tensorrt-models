# MLP
Learn basic procedures to build tensorRT

There should be a `mlp.wts` in this folder

If you are inside docker, then run
Create tensorrt engine
```
python3 build_mlp_engine.py
```
There should `mlp.engine' file created
To infer
```
python3 infer.py
```

If you are outside docker,
Create tensorrt engine
```
docker exec --workdir /workspace/TensorRT-Practice/mlp/ python3 build_mlp_engine.py
```
There should `mlp.engine' file created
To infer
```
docker exec --workdir /workspace/TensorRT-Practice/mlp/ python3 infer.py
```