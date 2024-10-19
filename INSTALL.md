# INSTALL
Build docker
```
docker build -t tensorrt_dev_img -f tensorrt_dev.Dockerfile .
```

To debug
```
docker run --rm -it --gpus=all --shm-size 8G --network host --env="DISPLAY" --env="QT_X11_NO_MITSHM=1" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --volume="$PWD:/workspace/" tensorrt_dev_img:latest /bin/bash
```

Run app
```
docker run --rm --gpus=all --shm-size 8G --network host --env="DISPLAY" --env="QT_X11_NO_MITSHM=1" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --volume="$PWD:/workspace/" tensorrt_dev_img:latest python3 your_application.py your_arguments
```