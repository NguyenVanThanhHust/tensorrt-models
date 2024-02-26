import argparse
import os
import numpy as np
import struct

# required for the model creation
import tensorrt as trt

# required for the inference using TRT engine
import pycuda.autoinit
import pycuda.driver as cuda

gLogger = trt.Logger(trt.Logger.INFO)

# Sizes of input and output for TensorRT model
INPUT_SIZE = 1
OUTPUT_SIZE = 1

def do_inference(infer_context, infer_host_input, infer_host_output):
    inference_engine = infer_context.engine

    assert inference_engine.num_bindings == 2

    # allocate memory in GPU using CUDA bindings
    input_memory_on_device = cuda.mem_alloc(infer_host_input.nbytes)
    output_memory_on_device = cuda.mem_alloc(infer_host_output.nbytes)

    # create bindings for input and outputs
    bindings = [int(input_memory_on_device), int(output_memory_on_device)]
    # Create cuda Stream for simultainous CUDA operations
    stream = cuda.Stream()

    # Copy input from host to device in stream
    cuda.memcpy_htod_async(input_memory_on_device, infer_host_input, stream)

    # execute inference using context provided by engine
    infer_context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

    # copy output back from device (GPU) to host (CPU)
    cuda.memcpy_dtoh_async(infer_host_output, output_memory_on_device, stream)

    # Synchroinze the stream to prevent issues
    # block CUDA and wait for CUDA operations to be completed
    stream.synchronize()

def infer(engine_path, input_val):
    # Create runtime
    runtime = trt.Runtime(gLogger)
    assert runtime

    # Read and deserialize engine for inference
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    assert engine

    # Create execution context
    context = engine.create_execution_context()
    assert context

    # Create input as array
    data = np.array([input_val], dtype=np.float32)
    
    # Capture free memory for input in GPU
    host_input = cuda.pagelocked_empty((INPUT_SIZE), dtype=np.float32)

    # Copy input from CPU to GPU
    np.copyto(host_input, data.ravel())

    # Capture free memory for output in GPU
    host_output = cuda.pagelocked_empty(OUTPUT_SIZE, dtype=np.float32)

    # Do Inference
    do_inference(context, host_input, host_input)

    print(f'\n[INFO]: Predictions using pre-trained model..\n\tInput:\t{input_val}\n\tOutput:\t{host_output[0]:.4f}')

if __name__ == '__main__':
    engine_path = "mlp.engine"
    infer(engine_path, 4.0)
    infer(engine_path, 1.0)