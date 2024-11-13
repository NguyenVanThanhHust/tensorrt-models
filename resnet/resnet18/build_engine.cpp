#include "NvInfer.h"    // TensorRT library
#include "iostream"     // Standard input/output library
#include "logging.h"    // logging file -- by NVIDIA
#include <map>          // for weight maps
#include <fstream>      // for file-handling
#include <chrono>       // for timing the execution
#include <cmath>

// Logger from TRT API
static Logger gLogger;

static const int INPUT_H = 224;
static const int INPUT_W = 224;
static const int OUTPUT_SIZE = 1000;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";

// provided by nvidia for using TensorRT APIs
using namespace nvinfer1;

/** ////////////////////////////
// DEPLOYMENT RELATED /////////
////////////////////////////*/
std::map<std::string, Weights> loadWeights(const std::string file) {
    /**
     * Parse the .wts file and store weights in dict format.
     *
     * @param file path to .wts file
     * @return weight_map: dictionary containing weights and their values
     */

    std::cout << "[INFO]: Loading weights..." << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open Weight file
    std::ifstream input(file);
    assert(input.is_open() && "[ERROR]: Unable to load weight file...");

    // Read number of weights
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    // Loop through number of line, actually the number of weights & biases
    while (count--) {
        // TensorRT weights
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;
        // Read name and type of weights
        std::string w_name;
        input >> w_name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        uint32_t *val = reinterpret_cast<uint32_t *>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x) {
            // Change hex values to uint32 (for higher values)
            input >> std::hex >> val[x];
        }
        wt.values = val;
        wt.count = size;

        // Add weight values against its name (key)
        weightMap[w_name] = wt;
    }
    return weightMap;
}

IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string layer_name, float eps)
{
    float *gamma = (float*)weightMap[layer_name + ".weight"].values;
    float *beta = (float*)weightMap[layer_name + ".bias"].values;
    float *mean = (float*)weightMap[layer_name + ".running_mean"].values;
    float *var = (float*)weightMap[layer_name + ".running_var"].values;
    int len = weightMap[layer_name + ".running_var"].count;
    std::cout<<layer_name<< " len "<<len<<std::endl;

     float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};
    
    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};

    weightMap[layer_name + ".scale"] = scale;
    weightMap[layer_name + ".shift"] = shift;
    weightMap[layer_name + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

IActivationLayer* basicBlock(INetworkDefinition *network, std::map<std::string, Weights>weightMap, ITensor& input, int in_channel, int out_channel, int stride, std::string layer_name)
{
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, out_channel, DimsHW{3, 3}, weightMap[layer_name + "conv1.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{stride, stride});
    conv1->setPaddingNd(DimsHW{1, 1});

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), layer_name + "bn1", 1e-5);

    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), out_channel, DimsHW{3, 3}, weightMap[layer_name + "conv2.weight"], emptywts);
    assert(conv2);
    conv2->setPaddingNd(DimsHW{1, 1});

    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), layer_name + "bn2", 1e-5);

    IElementWiseLayer *ew;
    if (in_channel!=out_channel)
    {
        IConvolutionLayer* conv3 = network->addConvolutionNd(input, out_channel, DimsHW{1, 1}, weightMap[layer_name + "downsample.0.weight"], emptywts);
        assert(conv3);
        conv3->setStrideNd(DimsHW{stride, stride});

        IScaleLayer* bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), layer_name + "downsample.1", 1e-5);
        ew = network->addElementWise(*bn3->getOutput(0), *bn2->getOutput(0), ElementWiseOperation::kSUM);
    }
    else
    {
        ew = network->addElementWise(input, *bn2->getOutput(0), ElementWiseOperation::kSUM);
    }
    IActivationLayer *relu2 = network->addActivation(*ew->getOutput(0), ActivationType::kRELU);
    assert(relu2);
    return relu2;
}

ICudaEngine *createEngine(unsigned int maxBatchSize, IBuilder *builder, IBuilderConfig *config, DataType dt) {
    /**
     * Create Multi-Layer Perceptron using the TRT Builder and Configurations
     *
     * @param maxBatchSize: batch size for built TRT model
     * @param builder: to build engine and networks
     * @param config: configuration related to Hardware
     * @param dt: datatype for model layers
     * @return engine: TRT model
     */

    std::cout << "[INFO]: Creating MLP using TensorRT..." << std::endl;

    // Load Weights from relevant file
    std::map<std::string, Weights> weightMap = loadWeights("../resnet18.wts");

    // Create an empty network
    INetworkDefinition *network = builder->createNetworkV2(0U);

    // Create an input with proper *name
    ITensor *data = network->addInput("data", DataType::kFLOAT, Dims3{3, INPUT_H, INPUT_W});
    assert(data);

    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer* conv1 = network->addConvolutionNd(*data, 64, DimsHW{7, 7}, weightMap["conv1.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{2, 2});
    conv1->setPaddingNd(DimsHW{3, 3});

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "bn1", 1e-5);

    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    IPoolingLayer* pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    assert(pool1);
    pool1->setStrideNd(DimsHW{2, 2});
    pool1->setPaddingNd(DimsHW{1, 1});

    IActivationLayer* relu2 = basicBlock(network, weightMap, *pool1->getOutput(0), 64, 64, 1, "layer1.0.");
    IActivationLayer* relu3 = basicBlock(network, weightMap, *relu2->getOutput(0), 64, 64, 1, "layer1.1.");

    IActivationLayer* relu4 = basicBlock(network, weightMap, *relu3->getOutput(0), 64, 128, 2, "layer2.0.");
    IActivationLayer* relu5 = basicBlock(network, weightMap, *relu4->getOutput(0), 128, 128, 1, "layer2.1.");

    IActivationLayer* relu6 = basicBlock(network, weightMap, *relu5->getOutput(0), 128, 256, 2, "layer3.0.");
    IActivationLayer* relu7 = basicBlock(network, weightMap, *relu6->getOutput(0), 256, 256, 1, "layer3.1.");

    IActivationLayer* relu8 = basicBlock(network, weightMap, *relu7->getOutput(0), 256, 512, 2, "layer4.0.");
    IActivationLayer* relu9 = basicBlock(network, weightMap, *relu8->getOutput(0), 512, 512, 1, "layer4.1.");

    IPoolingLayer* pool2 = network->addPoolingNd(*relu9->getOutput(0), PoolingType::kAVERAGE, DimsHW{7, 7});
    assert(pool2);
    pool2->setStrideNd(DimsHW{1, 1});
    
    IFullyConnectedLayer* fc1 = network->addFullyConnected(*pool2->getOutput(0), 1000, weightMap["fc.weight"], weightMap["fc.bias"]);
    assert(fc1);

    fc1->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*fc1->getOutput(0));

    // Set configurations
    builder->setMaxBatchSize(maxBatchSize);
    // Set workspace size
    config->setMaxWorkspaceSize(1 << 20);

    // Build CUDA Engine using network and configurations
    ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
    assert(engine != nullptr);

    // Don't need the network any more
    // free captured memory
    network->destroy();

    // Release host memory
    for (auto &mem: weightMap) {
        free((void *) (mem.second.values));
    }

    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory **modelStream) {
    /**
     * Create engine using TensorRT APIs
     *
     * @param maxBatchSize: for the deployed model configs
     * @param modelStream: shared memory to store serialized model
     */

    // Create builder with the help of logger
    IBuilder *builder = createInferBuilder(gLogger);

    // Create hardware configs
    IBuilderConfig *config = builder->createBuilderConfig();

    // Build an engine
    ICudaEngine *engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // serialize the engine into binary stream
    (*modelStream) = engine->serialize();

    // free up the memory
    engine->destroy();
    config->destroy();
    builder->destroy();
}

void performSerialization() {
    /**
     * Serialization Function
     */
    // Shared memory object
    IHostMemory *modelStream{nullptr};

    // Write model into stream
    APIToModel(1, &modelStream);
    assert(modelStream != nullptr);


    std::cout << "[INFO]: Writing engine into binary..." << std::endl;

    // Open the file and write the contents there in binary format
    std::ofstream p("resnet18.engine", std::ios::binary);
    if (!p) {
        std::cerr << "could not open plan output file" << std::endl;
        return;
    }
    p.write(reinterpret_cast<const char *>(modelStream->data()), modelStream->size());

    // Release the memory
    modelStream->destroy();

    std::cout << "[INFO]: Successfully created TensorRT engine..." << std::endl;
    std::cout << "\n\tRun inference using `./resnet -d`" << std::endl;

}

int main(int argc, char **argv) {
    performSerialization();
    return 0;
}