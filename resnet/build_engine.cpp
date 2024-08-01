#include "NvInfer.h"    // TensorRT library
#include "iostream"     // Standard input/output library
#include "logging.h"    // logging file -- by NVIDIA
#include <map>          // for weight maps
#include <fstream>      // for file-handling
#include <chrono>       // for timing the execution

// provided by nvidia for using TensorRT APIs
using namespace nvinfer1;

// Logger from TRT API
static Logger gLogger;

const int INPUT_SIZE = 1;
const int OUTPUT_SIZE = 1;

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

ICudaEngine *createMLPEngine(unsigned int maxBatchSize, IBuilder *builder, IBuilderConfig *config, DataType dt) {
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
    std::map<std::string, Weights> weightMap = loadWeights("../mlp.wts");

    // Create an empty network
    INetworkDefinition *network = builder->createNetworkV2(0U);

    // Create an input with proper *name
    ITensor *data = network->addInput("data", DataType::kFLOAT, Dims3{1, 1, 1});
    assert(data);

    // Add layer for MLP
    IFullyConnectedLayer *fc1 = network->addFullyConnected(*data, 1,
                                                           weightMap["linear.weight"],
                                                           weightMap["linear.bias"]);
    assert(fc1);

    // set output with *name
    fc1->getOutput(0)->setName("out");

    // mark the output
    network->markOutput(*fc1->getOutput(0));

    // Set configurations
    builder->setMaxBatchSize(1);
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
    ICudaEngine *engine = createMLPEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // serialize the engine into binary stream
    (*modelStream) = engine->serialize();

    // free up the memory
    engine->destroy();
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
    std::ofstream p("mlp.engine", std::ios::binary);
    if (!p) {
        std::cerr << "could not open plan output file" << std::endl;
        return;
    }
    p.write(reinterpret_cast<const char *>(modelStream->data()), modelStream->size());

    // Release the memory
    modelStream->destroy();

    std::cout << "[INFO]: Successfully created TensorRT engine..." << std::endl;
    std::cout << "\n\tRun inference using `./mlp -d`" << std::endl;

}

int main(int argc, char **argv) {
    performSerialization();
    return 0;
}