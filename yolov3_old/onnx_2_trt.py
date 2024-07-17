# https://github.com/jkjung-avt/tensorrt_demos/blob/master/yolo/onnx_to_tensorrt.py

from __future__ import print_function

import os
import argparse
import time 
from os.path import join, isdir, isfile

import tensorrt as trt

MIN_BATCH_SIZE = 1
OPT_BATCH_SIZE = 1
MAX_BATCH_SIZE = 1

def load_onnx(model_name):
    """
    Read onnx filepath
    """
    onnx_path = '%s.onnx' % model_name
    assert isfile(onnx_path), "Invalid onnx filepath, get {}".format(onnx_path)
    with open(onnx_path, "rb") as handle:
        return handle.read()

def set_net_batch(network, batch_size):
    """Set network input batch size.
    The ONNX file might have been generated with a different batch size,
    say, 64.
    """
    if trt.__version__[0] >= '7':
        shape = list(network.get_input(0).shape)
        shape[0] = batch_size
        network.get_input(0).shape = shape
    return network


def build_engine(model_name, do_int8, dla_core, verbose=True):
    """
    Build a TensorRT engine from ONNX 
    """
    print('Loading the ONNX file...')
    net_c, net_h, net_w = 3, 224, 224
    onnx_data = load_onnx(model_name)
    if onnx_data is None:
        return None

    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger()
    EXPLICIT_BATCH = [] if trt.__version__[0] < '7' else \
        [1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)]
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(*EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        if do_int8 and not builder.platform_has_fast_int8:
            raise RuntimeError('INT8 not supported on this platform')
        if not parser.parse(onnx_data):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
        network = set_net_batch(network, MAX_BATCH_SIZE)

        print('Naming the input tensort as "input".')
        network.get_input(0).name = 'input'

        print('Building the TensorRT engine.  This would take a while...')
        print('(Use "--verbose" or "-v" to enable verbose logging.)')
        builder.max_batch_size = MAX_BATCH_SIZE
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
        config.set_flag(trt.BuilderFlag.FP16)
        profile = builder.create_optimization_profile()
        profile.set_shape(
            'input',                                # input tensor name
            (MIN_BATCH_SIZE, net_c, net_h, net_w),  # min shape
            (OPT_BATCH_SIZE, net_c, net_h, net_w),  # opt shape
            (MAX_BATCH_SIZE, net_c, net_h, net_w))  # max shape
        config.add_optimization_profile(profile)
        if do_int8:
            from calibrator import YOLOEntropyCalibrator
            config.set_flag(trt.BuilderFlag.INT8)
            config.int8_calibrator = YOLOEntropyCalibrator(
                'calib_images', (net_h, net_w),
                'calib_%s.bin' % model_name)
            config.set_calibration_profile(profile)
        if dla_core >= 0:
            config.default_device_type = trt.DeviceType.DLA
            config.DLA_core = dla_core
            config.set_flag(trt.BuilderFlag.STRICT_TYPES)
            print('Using DLA core %d.' % dla_core)
        engine = builder.build_engine(network, config)

        if engine is not None:
            print('Completed creating engine.')
        return engine

def main():
    """Create a TensorRT engine for ONNX-based YOLO."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='enable verbose output (for debugging)')
    parser.add_argument(
        '-c', '--category_num', type=int,
        help='number of object categories (obsolete)')
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help=('[yolov3-tiny|yolov3|yolov3-spp|yolov4-tiny|yolov4|'
              'yolov4-csp|yolov4x-mish|yolov4-p5]-[{dimension}], where '
              '{dimension} could be either a single number (e.g. '
              '288, 416, 608) or 2 numbers, WxH (e.g. 416x256)'))
    parser.add_argument(
        '--int8', action='store_true',
        help='build INT8 TensorRT engine')
    parser.add_argument(
        '--dla_core', type=int, default=-1,
        help='id of DLA core for inference (0 ~ N-1)')
    args = parser.parse_args()

    engine = build_engine(
        args.model, args.int8, args.dla_core, args.verbose)
    if engine is None:
        raise SystemExit('ERROR: failed to build the TensorRT engine!')

    engine_path = '%s.trt' % args.model
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    print('Serialized the TensorRT engine to file: %s' % engine_path)


if __name__ == '__main__':
    main()