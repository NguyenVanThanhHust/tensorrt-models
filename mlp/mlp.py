import argparse
import os
import numpy as np
import struct

from os.path import isfile, isdir, join 

# required for the model creation
import tensorrt as trt

# required for the inference using TRT engine
import pycuda.autoinit
import pycuda.driver as cuda

# Sizes of input and output for TensorRT model
INPUT_SIZE = 1
OUTPUT_SIZE = 1

# path of .wts (weight file) and .engine (model file)
WEIGHT_PATH = "./mlp.wts"
ENGINE_PATH = "./mlp.engine"

# input and output names are must for the TRT model
INPUT_BLOB_NAME = 'input'
OUTPUT_BLOB_NAME = 'output'

# A logger provided by NVIDIA-TRT
gLogger = trt.Logger(trt.Logger.INFO)

################################
# DEPLOYMENT RELATED ###########
################################
def load_weights(file_path):
    """
    Parse .wts file and store weights in dict format
    """
    assert isfile(file_path), "Invalid file_path, {}".format(file_path)
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f]

    weight_map = {}
    # count for total # of weights
    count = int(lines[0])
    assert count == len(lines) - 1

    # Loop through counts and get the exact num of values against weights
    for i in range(1, count + 1):
        splits = lines[i].split(" ")
        name = splits[0]
        cur_count = int(splits[1])

        # len of splits must be greater than current weight counts
        assert cur_count + 2 == len(splits)

        # loop through all weights and unpack from the hexadecimal values
        values = []
        for j in range(2, len(splits)):
            # hex string to bytes to float
            values.append(struct.unpack(">f", bytes.fromhex(splits[j])))

        # store in format of { 'weight.name': [weights_val0, weight_val1, ..] }
        weight_map[name] = np.array(values, dtype=np.float32)

    return weight_map

def build_mlp_engine(max_batch_size, builder, config, data_type):
    """
    Create TRT model 
    """
    print("[INFO]: Creating MLP using TensorRT...")
    # Load weights map from file
    weight_map = load_weights(WEIGHT_PATH)

    # Build network
    network = builder.create_network()

    # Add input to the network
    data = network.add_input(INPUT_BLOB_NAME, data_type, (1, 1, INPUT_SIZE))
    assert data 

    # add the layer with output-size (number of outputs)
    linear = network.add_fully_connected(input=data,
                                         num_outputs=OUTPUT_SIZE,
                                         kernel=weight_map['linear.weight'],
                                         bias=weight_map['linear.bias'])
    assert linear

    # set the name for output layer
    linear.get_output(0).name = OUTPUT_BLOB_NAME

    # mark this layer as final output layer
    network.mark_output(linear.get_output(0))

    # Set the batch size of current builder
    builder.max_batch_size = max_batch_size

    # create the engine and hardware configs
    engine = builder.build_engine(network, config)

    del network
    del weight_map

    return engine

def api_to_model(max_batch_size):
    """
    Create engine using TensorRT API
    """
    # Create Builder with logger provided by TRT
    builder = trt.Builder(gLogger)

    # Create configurations from Engine Builder
    config = builder.create_builder_config()

    engine = build_mlp_engine(max_batch_size, builder, config, trt.float32)
    assert engine

    # Write the engine into binary file
    print("[INFO]: Writing engine into binary...")
    with open(ENGINE_PATH, "wb") as f:
        # write serialized model in file
        f.write(engine.serialize())

    # free the memory
    del engine
    del builder


################################
# INFERENCE RELATED ############
################################
def perform_inference(input_val):
    """
    Get inference using the pre-trained model
    :param input_val: a number as an input
    :return:
    """

    def do_inference(infer_context, host_input, host_output):
        """
        Perform inference using CUDA context
        """
        inference_engine = infer_context.engine
        # Input and ouput bindings are required for inference
        assert inference_engine.num_bindings == 2

        # allocate memory in GPU using CUDA bindings
        device_in = cuda.mem_alloc(host_input.nbytes)
        device_out = cuda.mem_alloc(host_output.nbytes)

        # create bindings for input and output
        bindings = [int(device_in), int(device_out)]

        # Create CUDA stream for simutanous CUDA operations
        stream = cuda.Stream()

        # Copy input from host(CPU) to device (GPU) in stream
        cuda.memcpy_htod_async(device_in, host_input, stream)

        # execute inference using context provided by engine
        infer_context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        
        # synchronize the stream
        stream.synchronize()

    # create a runtime (required for deserialization of model) with NVIDIA's logger
    runtime = trt.Runtime(gLogger)
    assert runtime

    # read and deserialize engine for inference
    with open(ENGINE_PATH, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    assert engine

    # create execution context -- required for inference executions
    context = engine.create_execution_context()
    assert context

    # Create input as array
    data = np.array([input_val], dtype=np.float32)

    # capture free memory for input in GPU
    host_input = cuda.pagelocked_empty((INPUT_SIZE), dtype=np.float32)

    # copy input array from CPU to GPU 
    np.copyto(host_input, data.ravel())

    host_output = cuda.pagelocked_empty(OUTPUT_SIZE, dtype=np.float32)

    do_inference(context, host_input, host_output)
    print(f'\n[INFO]: Predictions using pre-trained model..\n\tInput:\t{input_val}\n\tOutput:\t{host_output[0]:.4f}')


def get_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-s', action='store_true')
    arg_parser.add_argument('-d', action='store_true')
    arguments = vars(arg_parser.parse_args())
    # check for the arguments
    if not (arguments['s'] ^ arguments['d']):
        print("[ERROR]: Arguments not right!\n")
        print("\tpython mlp.py -s   # serialize model to engine file")
        print("\tpython mlp.py -d   # deserialize engine file and run inference")
        exit()

    return arguments

if __name__ == "__main__":
    args = get_args()
    if args['s']:
        api_to_model(max_batch_size=1)
        print("[INFO]: Successfully created TensorRT engine...")
        print("\n\tRun inference using `python mlp.py -d`\n")
    else:
        perform_inference(input_val=4.0)