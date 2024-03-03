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
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem) -> None:
        self.host = host_mem
        self.device = device_mem

    def __repr__(self) -> str:
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        print(len(engine.get_tensor_shape(binding)))
        size = trt.volume(engine.get_tensor_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_tensor_dtype(binding))
        # Allocate host and device buffer
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append device buffer to device bindings
        bindings.append(int(device_mem))
        if engine.get_tensor_mode(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream
 
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    print('11')
    [print(dir(out)) for out in outputs]
    print('12')

    # Return only the host outputs.
    return [out.host for out in outputs]


if __name__ == '__main__':
    engine_path = "weights/alexnet.trt"
    random_array = np.random.rand(3, 224, 224)
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    runtime = trt.Runtime(TRT_LOGGER)
    assert runtime
    
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    assert engine

    context = engine.create_execution_context()
    assert context

    data = np.ones((1 * 3 * 224 * 224), dtype=np.float32)
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    inputs[0].host = data

    trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    print(trt_outputs)
    print(f'Output: \n{trt_outputs[0][:10]}\n{trt_outputs[0][-10:]}')