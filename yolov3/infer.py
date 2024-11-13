import os
import argparse
import struct

import numpy as np
import tensorrt as trt

# required for the inference using TRT engine
import pycuda.autoinit
import pycuda.driver as cuda

BATCH_SIZE = 1
INPUT_H = 224
INPUT_W = 224
OUTPUT_SIZE = 1000

INPUT_BLOB_NAME = "data"
OUTPUT_BLOB_NAME = "prob"


def get_args():
    parser = argparse.ArgumentParser(prog="resnet50 in Tensor RT")
    parser.add_argument(
        "--model_path", type=str, default="resnet50.engine", help="input wts model file"
    )
    args = parser.parse_args()
    return args

def doInference(context, host_in, host_out, batchSize):
    engine = context.engine
    assert engine.num_bindings == 2

    devide_in = cuda.mem_alloc(host_in.nbytes)
    devide_out = cuda.mem_alloc(host_out.nbytes)
    bindings = [int(devide_in), int(devide_out)]
    stream = cuda.Stream()

    cuda.memcpy_htod_async(devide_in, host_in, stream)
    context.execute_async(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(host_out, devide_out, stream)
    stream.synchronize()

def perform_inference(model_path):
    gLogger = trt.Logger(trt.Logger.INFO)
    runtime = trt.Runtime(gLogger)
    assert runtime

    with open(model_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    assert engine

    context = engine.create_execution_context()
    assert context

    data = np.ones((BATCH_SIZE * 3 * INPUT_H * INPUT_W), dtype=np.float32)
    host_in = cuda.pagelocked_empty(BATCH_SIZE * 3 * INPUT_H * INPUT_W,
                                    dtype=np.float32)
    np.copyto(host_in, data.ravel())
    host_out = cuda.pagelocked_empty(OUTPUT_SIZE, dtype=np.float32)

    doInference(context, host_in, host_out, BATCH_SIZE)

    print(f'Output: \n{host_out[:10]}\n{host_out[-10:]}')


if __name__ == "__main__":
    args = get_args()
    model_path = args.model_path
    perform_inference(model_path=model_path)
