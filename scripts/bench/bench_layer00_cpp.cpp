#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

class Logger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cerr << msg << std::endl;
    }
  }
};

static void checkCuda(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    std::cerr << what << ": " << cudaGetErrorString(err) << std::endl;
    std::exit(1);
  }
}

static std::vector<char> readFile(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    std::cerr << "failed to open " << path << std::endl;
    std::exit(1);
  }
  file.seekg(0, std::ios::end);
  size_t size = static_cast<size_t>(file.tellg());
  file.seekg(0, std::ios::beg);
  std::vector<char> data(size);
  file.read(data.data(), size);
  return data;
}

static size_t elemSize(nvinfer1::DataType dtype) {
  switch (dtype) {
    case nvinfer1::DataType::kFLOAT:
      return 4;
    case nvinfer1::DataType::kHALF:
      return 2;
    case nvinfer1::DataType::kINT8:
      return 1;
    case nvinfer1::DataType::kINT32:
      return 4;
    case nvinfer1::DataType::kBOOL:
      return 1;
    case nvinfer1::DataType::kUINT8:
      return 1;
    case nvinfer1::DataType::kFP8:
      return 1;
    case nvinfer1::DataType::kBF16:
      return 2;
    case nvinfer1::DataType::kINT64:
      return 8;
  }
  return 4;
}

static int64_t volume(nvinfer1::Dims dims) {
  int64_t v = 1;
  for (int i = 0; i < dims.nbDims; ++i) {
    v *= dims.d[i] < 0 ? 1 : dims.d[i];
  }
  return v;
}

int main(int argc, char** argv) {
  const char* envEngine = std::getenv("ENGINE");
  std::string enginePath = envEngine ? envEngine : "/home/harvest/models/axera-onnx/trt-engines-bf16/layer_00_fp16.engine";
  int warmup = std::getenv("WARMUP") ? std::atoi(std::getenv("WARMUP")) : 10;
  int iters = std::getenv("ITERS") ? std::atoi(std::getenv("ITERS")) : 100;

  Logger logger;
  auto data = readFile(enginePath);
  auto* runtime = nvinfer1::createInferRuntime(logger);
  auto* engine = runtime->deserializeCudaEngine(data.data(), data.size());
  if (!engine) {
    std::cerr << "deserialize failed" << std::endl;
    return 1;
  }
  auto* context = engine->createExecutionContext();
  if (!context) {
    std::cerr << "context failed" << std::endl;
    return 1;
  }

  cudaStream_t stream;
  checkCuda(cudaStreamCreate(&stream), "cudaStreamCreate");

  std::vector<void*> ptrs(engine->getNbIOTensors(), nullptr);
  for (int i = 0; i < engine->getNbIOTensors(); ++i) {
    const char* name = engine->getIOTensorName(i);
    nvinfer1::Dims dims = engine->getTensorShape(name);
    size_t bytes = static_cast<size_t>(volume(dims)) * elemSize(engine->getTensorDataType(name));
    checkCuda(cudaMalloc(&ptrs[i], bytes), "cudaMalloc");
    checkCuda(cudaMemsetAsync(ptrs[i], 0, bytes, stream), "cudaMemsetAsync");
    if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
      context->setInputShape(name, dims);
    }
    context->setTensorAddress(name, ptrs[i]);
  }
  checkCuda(cudaStreamSynchronize(stream), "sync init");

  for (int i = 0; i < warmup; ++i) {
    if (!context->enqueueV3(stream)) {
      std::cerr << "enqueue failed during warmup" << std::endl;
      return 1;
    }
  }
  checkCuda(cudaStreamSynchronize(stream), "sync warmup");

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) {
    if (!context->enqueueV3(stream)) {
      std::cerr << "enqueue failed" << std::endl;
      return 1;
    }
  }
  checkCuda(cudaStreamSynchronize(stream), "sync bench");
  auto end = std::chrono::steady_clock::now();
  double totalMs = std::chrono::duration<double, std::milli>(end - start).count();

  std::cout << "cpp engine=" << enginePath << "\n";
  std::cout << "iters=" << iters << " total_ms=" << totalMs << " avg_ms=" << (totalMs / iters) << "\n";

  for (void* p : ptrs) {
    cudaFree(p);
  }
  cudaStreamDestroy(stream);
  delete context;
  delete engine;
  delete runtime;
  return 0;
}
