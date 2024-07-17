import os
import argparse
import struct

import numpy as np
import tensorrt as trt

# required for the inference using TRT engine
import pycuda.autoinit
import pycuda.driver as cuda

INPUT_BLOB_NAME="input"
OUTPUT_BLOB_NAME="output"

# Sizes of input and output for TensorRT model
INPUT_SIZE = 1
OUTPUT_SIZE = 1


def get_args():
    parser = argparse.ArgumentParser(prog="Simple MLP in Tensor RT")
    parser.add_argument("--model_path", type=str, default="mlp.engine", help="input wts model file")
    args = parser.parse_args()
    return args

def perform_inference(model_path, input_val):
    def do_inference(context, host_input, host_output):
        inference_engine = context.engine

        assert inference_engine.num_bindings == 2

        # allocate memory in GPU using CUDA bindings
        device_in = cuda.mem_alloc(host_input.nbytes)
        device_out = cuda.mem_alloc(host_output.nbytes)

        # Create binding for input and output
        bindings = [int(device_in), int(device_out)]

        # Create CUDA stream for simultanoues CUDA operations
        stream = cuda.Stream()

        # copy input from host(CPU) to device(GPU) in stream
        cuda.memcpy_htod_async(device_in, host_input, stream)

        # execute inference using context provided by engine
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

        # copy output back from device(GPU) to host(CPU)
        cuda.memcpy_dtoh_async(host_output, device_out, stream)

        # synchronize the stream to prevent issues
        stream.synchronize()

    # Create a runtime 
    gLogger = trt.Logger(trt.Logger.INFO)
    runtime = trt.Runtime(gLogger)
    assert runtime

    # Load the TensorRT engine from the serialized model file
    with open(model_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    assert engine

    # Create execution context
    context = engine.create_execution_context()
    assert context

    # create input array
    data = np.array([input_val], dtype=np.float32)

    # capture free memory for input in GPU
    host_input = cuda.pagelocked_empty((INPUT_SIZE), dtype=np.float32)

    # Copy input from CPU to flatte array in GPU
    np.copyto(host_input, data.ravel())

    # capture free memrory for output in GPU
    host_output = cuda.pagelocked_empty(OUTPUT_SIZE, dtype=np.float32)

    # infer
    do_inference(context, host_input, host_output)

    print(f'\n[INFO]: Predictions using pre-trained model..\n\tInput:\t{input_val}\n\tOutput:\t{host_output[0]:.4f}')

if __name__ == '__main__':
    args = get_args()
    model_path = args.model_path
    perform_inference(model_path=model_path, input_val=4.0)