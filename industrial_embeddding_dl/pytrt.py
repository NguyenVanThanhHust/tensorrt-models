import argparse
from segcolors import colors
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import pdb
import os
import cv2
import time


class TRTSegmentor(object):
    def __init__(self, 
                onnxpath, 
                colors,
                input_size=(640, 360),
                maxworkspace=(1<<25), 
                precision='FP16',
                device='GPU',
                max_batch_size=1, 
                calibrator=None, 
                dla_core=0,
                ) -> None:
        self.onnxpath = onnxpath
        self.enginepath=onnxpath+f'.{precision}.{device}.{dla_core}.{max_batch_size}.trt'
        #filename to be used for saving and reading engines
        self.nclasses=21
        self.pp_mean=np.array([0.485, 0.456, 0.406]).reshape((1,1,3))
        self.pp_stdev=np.array([0.229, 0.224, 0.225]).reshape((1,1,3))
        #mean and stdev for pre-processing images, see torchvision documentation
        self.colors=colors #colormap for 21 classes of Pascal VOC

        self.in_w = input_size[0]
        self.in_h = input_size[1]

        self.maxworkspace = maxworkspace
        self.precision_str = precision

        self.device = {'GPU': trt.DeviceType.GPU, 'DLA':trt.DeviceType.DLA}[device]
        self.dla_core = dla_core
        self.calibrator = calibrator
        self.allowGPUFallback = 3

        self.engine, self.logger = self.parse_or_load()

        self.context = self.engine.create_execution_context()
        self.trt2np_type = {'FLOAT': np.float32, 'HALF': np.float16, 'INT8': np.int8}
        self.dtype = self.trt2np_type[self.engine.get_binding_dtype(0).name]

        self.allocate_buffers(np.zeros(1, 3, self.in_h, self.in_w), dtype=self.dtype)
        
    def allocate_buffers(self, image):
        input_size = image.shape[-2:]
        output_size = [input_size[0] >> 3, input_size[1] >> 3]
        self.output = np.emtpy((self.nclasses, output_size[0], output_size[1]), dtype=self.dtype)
        self.d_input = cuda.mem_alloc(image.nbytes)
        self.d_output = cuda.mem_alloc(self.output.nbytes)

        self.bindings = [int(self.d_input), int(self.d_output)]
        self.stream = cuda.Stream()

    def preprocess(self, img):
        img = cv2.resize(img, (self.in_w, self.in_h))
        img = img[..., ::-1]
        img=img.astype(np.float32)/255
        img=(img-self.pp_mean)/self.pp_stdev

        img=np.transpose(img,(2,0,1))
        img=np.ascontiguousarray(img[None,...]).astype(self.dtype)

        return img

    def infer(self, image, benchmark=False):
        input_tensor = self.preprocess(image)

        start=time.time()

        cuda.memcpy_htod_async(self.d_input, input_tensor, self.stream)
        self.context.execute_async_v2(self.bindings, self.stream.handle, None)
        cuda.memcpy_dtoh_async(self.output, self.d_output, self.stream)

        self.stream.synchronize()
        if benchmark:
            duration = (time.time() - start)
            return duration

    def infer(self, image, benchmark=False):
        """
        image: unresized,
        """
        intensor=self.preprocess(image)

        start=time.time()

        cuda.memcpy_htod_async(self.d_input, intensor, self.stream)
        self.context.execute_async_v2(self.bindings, self.stream.handle, None)
        cuda.memcpy_dtoh_async(self.output, self.d_output, self.stream)

        self.stream.synchronize()
        
        if benchmark:
            duration=(time.time()-start)
            return duration

    def infer_async(self, intensor):
        #intensor should be preprocessed tensor
        cuda.memcpy_htod_async(self.d_input, intensor, self.stream)
        self.context.execute_async_v2(self.bindings, self.stream.handle, None)
        cuda.memcpy_dtoh_async(self.output, self.d_output, self.stream)

    def draw(self, img):
        shape=(img.shape[1],img.shape[0])
        segres=np.transpose(self.output,(1,2,0)).astype(np.float32)

        segres=cv2.resize(segres, shape)
        mask=segres.argmax(axis=-1)
        colored=self.colors[mask]

        drawn=cv2.addWeighted(img, 0.5, colored, 0.5, 0.0)
        return drawn

    def infervideo(self, infile):
        src=cv2.VideoCapture(infile)
        ret,frame=src.read()
        fps=0.0

        if not ret:
            print('Cannot read file/camera: {}'.format(infile))

        while ret:
            duration=self.infer(frame, benchmark=True)
            drawn=self.draw(frame)
            cv2.imshow('segmented', drawn)
            k=cv2.waitKey(1)
            if k==ord('q'):
                break

            fps=0.9*fps+0.1/(duration)
            print('FPS=:{:.2f}'.format(fps))
            ret,frame=src.read()

    def parse_or_load(self, ):
        logger = trt.Logger(trt.logger.INFO)
        if os.path.exists(self.enginepath):
            logger.log(trt.Logger.INFO, 'Found an existing engine')
            with open(self.enginepath, 'rb') as f:
                # create runtime 
                rt = trt.Runtime(logger)
                # load engine
                engine = rt.deserialize_cuda_engine(f.read())
            return engine, logger
        
        with trt.Builder(logger) as builder:
            builder.max_batch_size = self.max_batch_size

            network_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            with builder.create_network(network_flag) as net:
                with trt.OnnxParser(net, logger) as parser:
                    with open(self.onnxpath, 'rb') as f:
                        if not parser.parse(f.read()):
                            for err in range(parser.num_errors):
                                print(parser.get_error(err))
                        else:
                            logger.log(trt.Logger.INFO, 'Onnx file parsed successfully')

                        net.get_input(0).dtype = trt.DataType.HALF
                        net.get_output(0).dtype = trt.DataType.HALF

                        config = builder.create_builder_config()

                        config.max_workspace_size = self.maxworkspace
                        if self.precision_str in ['FP16', 'INT8']:
                            config.flags = ((1 << self.precision) | (1<<self.allowGPUFallback))
                            config.DLA_core = self.dla_core
                        
                        config.default_device = self.device

                        config.profiling_verbosity = trt.ProfilingVerbosity.VERBOSE

                        if self.precision_str=='INT8' and self.calibrator is None:
                            logger.log(trt.Logger.ERROR, 'Please provide calibrator')
                            #can't proceed without a calibrator
                            quit()
                        elif self.precision_str=='INT8' and self.calibrator is not None:
                            config.int8_calibrator=self.calibrator
                            logger.log(trt.Logger.INFO, 'Using INT8 calibrator provided by user')

                        logger.log(trt.Logger.INFO, 'Checking if network is supported...')
                        if builder.is_network_supported(net, config):
                            logger.log(trt.Logger.INFO, 'Network is supported')
                        else:
                            logger.log(trt.Logger.ERROR, 'Network contains operations that are not supported')
                            logger.log(trt.Logger.ERROR, 'Quitting')
                            quit()

                        logger.log(trt.Logger.INFO, 'Building inference engine...')
                        engine = builder.build_engine(net, config)

                        logger.log(trt.Logger.INFO, 'Inference engine built, successfully')

                        with open(self.enginepath, 'wb') as s:
                            s.write(engine.serialize())
                        logger.log(trt.Logger.INFO, "Inference engine is saved to {}".format(self.enginepath))

class Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, imgdir, n_samples,input_size=(640,360), batch_size=1, iotype=np.float16):
        super().__init__()
        self.imgdir=imgdir
        self.n_samples=n_samples
        self.input_size=input_size
        self.batch_size=batch_size
        self.iotype=iotype
        self.pp_mean=np.array([0.485, 0.456, 0.406]).reshape((1,1,3))
        self.pp_stdev=np.array([0.229, 0.224, 0.225]).reshape((1,1,3))
        self.cache_path='cache.ich'
        self.setup()
        self.images_read=0

    def setup(self):
        all_images=sorted([f for f in os.listdir(self.imgdir) if f.endswith('.jpg')])
        assert len(all_images)>=self.n_samples, f'Not enough images available. Requested {self.n_samples} images for calibration but only {len(all_images)} are avialable in {self.imgdir}'
        used=all_images[:self.n_samples]
        self.images=[os.path.join(self.imgdir,f) for f in used]

        nbytes=self.batch_size*3*self.input_size[0]*self.input_size[1]*self.iotype(1).nbytes
        self.buffer=cuda.mem_alloc(nbytes)

    def preprocess(self, img):
        img=cv2.resize(img,self.input_size)
        img=img[...,::-1] #bgr2rgb
        img=img.astype(np.float32)/255
        img=(img-self.pp_mean)/self.pp_stdev #normalize

        img=np.transpose(img,(2,0,1)) #HWC to CHW format
        img=np.ascontiguousarray(img[None,...]).astype(self.iotype)
        #NCHW data of type used by engine input
        return img

    def get_batch(self, names):
        if self.images_read+self.batch_size < self.n_samples:
            batch=[]
            for idx in range(self.images_read,self.images_read+self.batch_size):
                img=cv2.imread(self.images[idx],1)
                intensor=self.preprocess(img)
                batch.append(intensor)

            batch=np.concatenate(batch, axis=0)
            cuda.memcpy_htod(self.buffer, batch)
            self.images_read+=self.batch_size
            return [int(self.buffer)]
        else:
            return None
        
    def get_batch_size(self):
        return self.batch_size

    def read_calibration_cache(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_path, 'wb') as f:
            f.write(cache)

def infervideo_2DLAs(infile, onnxpath, calibrator=None, precision='INT8',display=False):
    src=cv2.VideoCapture(infile)
    seg1=TRTSegmentor(onnxpath, colors, device='DLA', precision=precision ,calibrator=calibrator, dla_core=0)
    seg2=TRTSegmentor(onnxpath, colors, device='DLA', precision=precision ,calibrator=calibrator, dla_core=1)
    ret1,frame1=src.read()
    ret2,frame2=src.read()
    fps=0.0
    
    while ret1 and ret2:
        intensor1=seg1.preprocess(frame1)
        intensor2=seg2.preprocess(frame2)
        
        start=time.time()

        cuda.memcpy_htod_async(seg1.d_input, intensor1, seg1.stream)
        cuda.memcpy_htod_async(seg2.d_input, intensor2, seg2.stream)

        seg1.context.execute_async_v2(seg1.bindings, seg1.stream.handle, None)
        seg2.context.execute_async_v2(seg2.bindings, seg2.stream.handle, None)

        cuda.memcpy_dtoh_async(seg1.output, seg1.d_output, seg1.stream)
        cuda.memcpy_dtoh_async(seg2.output, seg2.d_output, seg2.stream)

        seg1.stream.synchronize()
        seg2.stream.synchronize()

        end=time.time()
        if display:
            drawn1=seg1.draw(frame1)
            drawn2=seg2.draw(frame2)
            cv2.imshow('segmented1', drawn1)
            cv2.imshow('segmented2', drawn2)
            k=cv2.waitKey(1)
            if k==ord('q'):
                break

        fps=0.9*fps+0.1*(2.0/(end-start))
        print('FPS = {:.3f}'.format(fps))

        ret1,frame1=src.read()
        ret2,frame2=src.read()

if __name__ == '__main__':

    parser=argparse.ArgumentParser(description='TensorRT python tutorial')
    
    parser.add_argument('--precision', type=str, 
        default='fp16', choices=['int8', 'fp16', 'fp32'],
        help='precision FP32, FP16 or INT8')

    parser.add_argument('--device', type=str, 
        default='gpu', choices=['gpu', 'dla', 'dla0', 'dla1', '2DLAs'],
        help='GPU, DLA or 2DLAs')

    parser.add_argument('--infile', type=str, required=True,
        help='path of input video file to infer on')

    args=parser.parse_args()

    calibrator=Calibrator('./val2017/', 5000)

    if args.device=='2DLAs':
        precision=args.precision.upper()
        infervideo_2DLAs(args.infile, './segmodel.onnx', calibrator, precision)

    else:
        device=args.device.upper()
        precision=args.precision.upper()
        dla_core=int(device[3:]) if len(device)>3 else 0
        device=device[:3]
        
        seg=TRTSegmentor('./weights/segmodel.onnx', colors, 
            device=device, 
            precision=precision,
            calibrator=calibrator, 
            dla_core=dla_core)
        
        seg.infervideo(args.infile)

    print('Inferred successfully')
                        