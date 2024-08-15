
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

static const int INPUT_H = 224;
static const int INPUT_W = 224;
static const int OUTPUT_SIZE = 1000;

/** ////////////////////////////
// INFERENCE RELATED //////////
////////////////////////////*/
void doInference(IExecutionContext &context, float *input, float *output, int batchSize) {
    /**
     * Perform inference using the CUDA context
     *
     * @param context: context created by engine
     * @param input: input from the host
     * @param output: output to save on host
     * @param batchSize: batch size for TRT model
     */

    // Get engine from the context
    const ICudaEngine &engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void *buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex("data");
    const int outputIndex = engine.getBindingIndex("prob");

    // Create GPU buffers on device -- allocate memory for input and output
    cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float));
    cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float));

    // create CUDA stream for simultaneous CUDA operations
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // copy input from host (CPU) to device (GPU)  in stream
    cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream);

    // execute inference using context provided by engine
    context.enqueue(batchSize, buffers, stream, nullptr);

    // copy output back from device (GPU) to host (CPU)
    cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost,
                    stream);

    // synchronize the stream to prevent issues
    //      (block CUDA and wait for CUDA operations to be completed)
    cudaStreamSynchronize(stream);

    // Release stream and buffers (memory)
    cudaStreamDestroy(stream);
    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputIndex]);
}

void performInference() {
    /**
     * Get inference using the pre-trained model
     */

    // stream to write model
    char *trtModelStream{nullptr};
    size_t size{0};

    // read model from the engine file
    std::ifstream file("resnet34.engine", std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }

    // Subtract mean from image
    static float data[3 * INPUT_H * INPUT_W];
    for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
        data[i] = 1.0;

    // Run inference
    static float prob[OUTPUT_SIZE];

    // create a runtime (required for deserialization of model) with NVIDIA's logger
    IRuntime *runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);

    // deserialize engine for using the char-stream
    ICudaEngine *engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);

    // create execution context -- required for inference executions
    IExecutionContext *context = engine->createExecutionContext();
    assert(context != nullptr);

    for (int i = 0; i < 100; i++) {
        auto start = std::chrono::system_clock::now();
        doInference(*context, data, prob, 1);
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }

    // free the captured space
    context->destroy();
    engine->destroy();
    runtime->destroy();

    // Print histogram of the output distribution
    std::cout << "\nOutput:\n\n";
    for (unsigned int i = 0; i < 10; i++)
    {
        std::cout << prob[i] << ", ";
    }
    std::cout << std::endl;
    for (unsigned int i = 0; i < 10; i++)
    {
        std::cout << prob[OUTPUT_SIZE - 10 + i] << ", ";
    }
}

int main(int argc, char **argv) {
    performInference();
    return 0;
}