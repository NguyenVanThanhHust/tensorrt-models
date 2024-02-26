import argparse
import os
import numpy as np
import struct

# required for the model creation
import tensorrt as trt

# required for the inference using TRT engine
import pycuda.autoinit
import pycuda.driver as cuda

from loguru import logger
# Sizes of input and output for TensorRT model
INPUT_SIZE = 1
OUTPUT_SIZE = 1

# path of .wts (weight file) and .engine (model file)
WEIGHT_PATH = "./mlp.wts"
ENGINE_PATH = "./mlp.engine"

# input and output names are must for the TRT model
INPUT_BLOB_NAME = 'data'
OUTPUT_BLOB_NAME = 'out'

# A logger provided by NVIDIA-TRT
gLogger = trt.Logger(trt.Logger.INFO)


def load_weights(file_path):
    """
    Parse .wts file and store weight in dict format
    """
    logger
    assert os.path.isfile(file_path)
    
    weight_map = {}
    with open(file_path, 'r') as f:
        lines = [l.strip() for l in f]
    
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

def build_mlp_engine(weight_map, max_batch_size=1, data_type=trt.float32):
    """
    Create Multi-Layer Perceptron using the TRT Builder and Configurations
    :param max_batch_size: batch size for built TRT model
    :param builder: to build engine and networks
    :param config: configuration related to Hardware
    :param dt: datatype for model layers
    :return engine: TRT model
    """
    builder = trt.Builder(gLogger)

    config = builder.create_builder_config()


    print("[INFO]: Creating MLP using TensorRT...")
    # load weight maps from the file
    weight_map = load_weights(WEIGHT_PATH)

    # build an empty network using builder
    network = builder.create_network()

    # add input to network using *input_name
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

    # set the batch size of current builder
    builder.max_batch_size = max_batch_size

    # create the engine with model and hardware configs
    engine = builder.build_engine(network, config)
    assert engine

    # free captured memory
    del network
    del weight_map

    # Write the engine into binary file
    print("[INFO]: Writing engine into binary...")
    with open(ENGINE_PATH, "wb") as f:
        # write serialized model in file
        f.write(engine.serialize())

    # free the memory
    del engine

if __name__ == "__main__":
    weight_path = load_weights(WEIGHT_PATH)
    build_mlp_engine(weight_map=weight_path, max_batch_size=1, data_type=trt.float32)