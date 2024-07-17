import os
import argparse
import struct

import numpy as np
import tensorrt as trt

INPUT_BLOB_NAME="input"
OUTPUT_BLOB_NAME="output"

# Sizes of input and output for TensorRT model
INPUT_SIZE = 1
OUTPUT_SIZE = 1

def get_args():
    parser = argparse.ArgumentParser(prog="YOLOv3 in Tensor RT")
    parser.add_argument("--input_path", type=str, default="mlp.wts", help="input wts model file")
    parser.add_argument("--output_path", type=str, default="mlp.engine", help="output path of engine file")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    args = parser.parse_args()
    return args

def load_weight(input_path):
    assert os.path.isfile(input_path), f"file {input_path} doesn't exist"
    weight_map = {}
    with open(input_path, "r") as f:
        lines = [l.rstrip() for l in f]
    
    # Count for total line of weights
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

def create_mlp_engine(builder, input_path, batch_size, config, data_type, ):
    weight_map = load_weight(input_path)

    # Build empty netwokr using builder
    network = builder.create_network()

    # add input to network using input-name
    data = network.add_input(INPUT_BLOB_NAME, data_type, (1, 1, INPUT_SIZE))

    # add the layer with output size
    linear = network.add_fully_connected(input=data,
                                         num_outputs=OUTPUT_SIZE,
                                         kernel=weight_map['linear.weight'],
                                         bias=weight_map['linear.bias'])
    assert linear

    # set the name for output layer
    linear.get_output(0).name = OUTPUT_BLOB_NAME

    # mark this layer as final output layer
    network.mark_output(linear.get_output(0))

    # set batch size of current builder
    builder.max_batch_size = batch_size
    
    # create the engine with model
    engine = builder.build_engine(network, config)

    # free captured memory
    del network
    del weight_map

    return engine

if __name__ == '__main__':
    args = get_args()
    input_path = args.input_path
    output_path = args.output_path
    batch_size = args.batch_size

    # Create builder 
    gLogger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(gLogger)

    # Create configuration from engine builder
    config = builder.create_builder_config()

    # Build TensorRT engine here
    engine = create_mlp_engine(builder, input_path, batch_size, config, trt.float32)

    # Serialize the engine to a file
    with open(output_path, "wb") as f:
        f.write(engine.serialize())
    
    # Free the memory
    del engine
    del builder
    del config