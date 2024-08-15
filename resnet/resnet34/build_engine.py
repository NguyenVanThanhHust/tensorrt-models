import os
import argparse
import struct

import numpy as np
import tensorrt as trt

BATCH_SIZE = 1
INPUT_H = 224
INPUT_W = 224
OUTPUT_SIZE = 1000

INPUT_BLOB_NAME = "data"
OUTPUT_BLOB_NAME = "prob"

EPS = 1e-5


def get_args():
    parser = argparse.ArgumentParser(prog="Resnet34 in Tensor RT")
    parser.add_argument(
        "--input_path", type=str, default="resnet34.wts", help="input wts model file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="resnet34.engine",
        help="output path of engine file",
    )
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


def addBatchNorm2d(network, weight_map, input, layer_name, eps):
    gamma = weight_map[layer_name + ".weight"]
    beta = weight_map[layer_name + ".bias"]
    mean = weight_map[layer_name + ".running_mean"]
    var = weight_map[layer_name + ".running_var"]
    var = np.sqrt(var + eps)

    scale = gamma / var
    shift = -mean / var * gamma + beta
    return network.add_scale(
        input=input, mode=trt.ScaleMode.CHANNEL, shift=shift, scale=scale
    )


def basicBlock(
    network, weight_map, input, in_channels, out_channels, stride, layer_name
):
    conv1 = network.add_convolution_nd(
        input=input,
        num_output_maps=out_channels,
        kernel_shape=(3, 3),
        kernel=weight_map[layer_name + "conv1.weight"],
        bias=trt.Weights(),
    )
    conv1.stride_nd = (stride, stride)
    conv1.padding_nd = (1, 1)

    assert conv1
    bn1 = addBatchNorm2d(
        network, weight_map, conv1.get_output(0), layer_name + "bn1", EPS
    )
    assert bn1
    relu1 = network.add_activation(bn1.get_output(0), type=trt.ActivationType.RELU)
    assert relu1

    conv2 = network.add_convolution_nd(
        input=relu1.get_output(0),
        num_output_maps=out_channels,
        kernel_shape=(3, 3),
        kernel=weight_map[layer_name + "conv2.weight"],
        bias=trt.Weights(),
    )
    assert conv2
    conv2.padding_nd = (1, 1)

    bn2 = addBatchNorm2d(
        network, weight_map, conv2.get_output(0), layer_name + "bn2", EPS
    )
    assert bn2

    if in_channels != out_channels:
        conv3 = network.add_convolution_nd(
            input=input,
            num_output_maps=out_channels,
            kernel_shape=(1, 1),
            kernel=weight_map[layer_name + "downsample.0.weight"],
            bias=trt.Weights(),
        )
        assert conv3
        conv3.stride_nd = (stride, stride)

        bn3 = addBatchNorm2d(
            network, weight_map, conv3.get_output(0), layer_name + "downsample.1", EPS
        )
        assert bn3

        ew1 = network.add_elementwise(
            bn3.get_output(0), bn2.get_output(0), trt.ElementWiseOperation.SUM
        )
    else:
        ew1 = network.add_elementwise(
            input, bn2.get_output(0), trt.ElementWiseOperation.SUM
        )
    assert ew1

    relu2 = network.add_activation(ew1.get_output(0), type=trt.ActivationType.RELU)
    assert relu2

    return relu2


def create_resnet_engine(
    builder,
    input_path,
    batch_size,
    config,
    data_type,
):
    weight_map = load_weight(input_path)

    # Build empty netwokr using builder
    network = builder.create_network()

    # add input to network using input-name
    data = network.add_input(INPUT_BLOB_NAME, data_type, (3, INPUT_H, INPUT_W))
    assert data

    conv1 = network.add_convolution_nd(
        input=data,
        num_output_maps=64,
        kernel_shape=(7, 7),
        kernel=weight_map["conv1.weight"],
        bias=trt.Weights(),
    )
    assert conv1
    conv1.stride_nd = (2, 2)
    conv1.padding_nd = (3, 3)

    bn1 = addBatchNorm2d(network, weight_map, conv1.get_output(0), "bn1", EPS)
    assert bn1

    relu1 = network.add_activation(bn1.get_output(0), type=trt.ActivationType.RELU)
    assert relu1

    pool1 = network.add_pooling_nd(
        input=relu1.get_output(0),
        window_size=trt.DimsHW(3, 3),
        type=trt.PoolingType.MAX,
    )
    assert pool1
    pool1.stride_nd = (2, 2)
    pool1.padding_nd = (1, 1)

    relu2 = basicBlock(network, weight_map, pool1.get_output(0), 64, 64, 1, "layer1.0.")
    relu3 = basicBlock(network, weight_map, relu2.get_output(0), 64, 64, 1, "layer1.1.")
    relu4 = basicBlock(network, weight_map, relu3.get_output(0), 64, 64, 1, "layer1.2.")

    relu5 = basicBlock(network, weight_map, relu4.get_output(0), 64, 128, 2, "layer2.0.")
    relu6 = basicBlock(network, weight_map, relu5.get_output(0), 128, 128, 1, "layer2.1.")
    relu7 = basicBlock(network, weight_map, relu6.get_output(0), 128, 128, 1, "layer2.2.")
    relu8 = basicBlock(network, weight_map, relu7.get_output(0), 128, 128, 1, "layer2.3.")

    relu9 = basicBlock(network, weight_map, relu8.get_output(0), 128, 256, 2, "layer3.0.")
    relu10 = basicBlock(network, weight_map, relu9.get_output(0), 256, 256, 1, "layer3.1.")
    relu11 = basicBlock(network, weight_map, relu10.get_output(0), 256, 256, 1, "layer3.2.")
    relu12 = basicBlock(network, weight_map, relu11.get_output(0), 256, 256, 1, "layer3.3.")
    relu13 = basicBlock(network, weight_map, relu12.get_output(0), 256, 256, 1, "layer3.4.")
    relu14 = basicBlock(network, weight_map, relu13.get_output(0), 256, 256, 1, "layer3.5.")

    relu15 = basicBlock(network, weight_map, relu14.get_output(0), 256, 512, 2, "layer4.0.")
    relu16 = basicBlock(network, weight_map, relu15.get_output(0), 512, 512, 1, "layer4.1.")
    relu17 = basicBlock(network, weight_map, relu16.get_output(0), 512, 512, 1, "layer4.2.")

    pool2 = network.add_pooling_nd(
        relu17.get_output(0), window_size=trt.DimsHW(7, 7), type=trt.PoolingType.AVERAGE
    )
    assert pool2
    pool2.stride_nd = (1, 1)

    fc1 = network.add_fully_connected(
        input=pool2.get_output(0),
        num_outputs=OUTPUT_SIZE,
        kernel=weight_map["fc.weight"],
        bias=weight_map["fc.bias"],
    )
    assert fc1

    fc1.get_output(0).name = OUTPUT_BLOB_NAME
    network.mark_output(fc1.get_output(0))

    # set batch size of current builder
    builder.max_batch_size = batch_size

    # create the engine with model
    engine = builder.build_engine(network, config)

    # free captured memory
    del network
    del weight_map
    return engine


if __name__ == "__main__":
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
    engine = create_resnet_engine(builder, input_path, batch_size, config, trt.float32)

    # Serialize the engine to a file
    with open(output_path, "wb") as f:
        f.write(engine.serialize())

    # Free the memory
    del engine
    del builder
    del config
