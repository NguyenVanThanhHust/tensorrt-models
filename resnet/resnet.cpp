#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"

using std::cout;
using std::endl;

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)
static Logger gLogger;

#define DEVICE 0  // GPU id

using namespace nvinfer1;

const char* INPUT_BLOB_NAME = "input_0";
const char* OUTPUT_BLOB_NAME = "output_0";

void normalizeImage(cv::Mat& image, const cv::Scalar& mean, const cv::Scalar& std) {
    std::vector<cv::Mat> channels;
    cv::split(image, channels);

    for (int i = 0; i < channels.size(); ++i) {
        channels[i] -= mean[i]*255;
        channels[i] /= std[i]*255;
    }

    cv::merge(channels, image);
}


void blobFromImage(cv::Mat& img, float *blob){
    int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;
    for (size_t c = 0; c < channels; c++) 
    {
        for (size_t  h = 0; h < img_h; h++) 
        {
            for (size_t w = 0; w < img_w; w++) 
            {
                blob[c * img_w * img_h + h * img_w + w] =
                    (float)img.at<cv::Vec3b>(h, w)[c];
            }
        }
    }
}

void doInference(IExecutionContext& context, float* input, float* output, const int output_size, cv::Size input_shape) {
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);

    assert(engine.getBindingDataType(inputIndex) == nvinfer1::DataType::kFLOAT);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);
    assert(engine.getBindingDataType(outputIndex) == nvinfer1::DataType::kFLOAT);
    int mBatchSize = engine.getMaxBatchSize();

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], 3 * input_shape.height * input_shape.width * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], output_size*sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, 3 * input_shape.height * input_shape.width * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(1, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

int main(int argc, char** argv)
{
    cv::Scalar mean(0.485, 0.456, 0.406);
    cv::Scalar std(0.229, 0.224, 0.225);

    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};
    
    if (argc == 4 && std::string(argv[2]) == "-i") {
        const std::string engine_file_path {argv[1]};
        std::ifstream file(engine_file_path, std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    } else {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./resnet ../resnet_trt.engine -i ../../../CCTV/videos/WildTrack/cam3_short.mp4  // deserialize file and run inference" << std::endl;
        return -1;
    }

    std::string video_path  {argv[3]};
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened())
    {
        cout<<"Can't open video"<<endl;
        return 1;
    }

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr); 
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    auto out_dims = engine->getBindingDimensions(1);
    auto output_size = 1;
    for(int j=0;j<out_dims.nbDims;j++) {
        output_size *= out_dims.d[j];
    }
    cout<<"Output size: "<<output_size<<endl;

    cv::Mat frame;
    cv::Mat resizedFrame;
    int width = 224;
    int height = 224;
    static float* prob = new float[output_size];

    while (cap.read(frame))
    {
        cv::resize(frame, resizedFrame, cv::Size(width, height));
        cout<<resizedFrame.rows<<" "<<resizedFrame.cols<<endl;
        cv::cvtColor(resizedFrame, resizedFrame, cv::COLOR_BGR2RGB);
        float blob[resizedFrame.total()*3];
        normalizeImage(resizedFrame, mean, std);
        blobFromImage(resizedFrame, blob);
        doInference(*context, blob, prob, output_size, resizedFrame.size());
        auto maxElementIterator = std::max_element(prob, prob + output_size);
        int maxIndex = std::distance(prob, maxElementIterator);
        cout<<"Class "<<maxIndex<<endl;

    }
    cap.release();
    
    return 0;
}