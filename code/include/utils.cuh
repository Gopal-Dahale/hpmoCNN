#ifndef UTILS
#define UTILS

#include <cublas_v2.h>
#include <cudnn.h>
#include <curand.h>

#include <cmath>
#include <iostream>
#include <iomanip>

#define BW (16 * 16)
#define CNMEM_GRANULARITY 512

#define FatalError(s)                                                 \
  do {                                                                \
    std::stringstream _where, _message;                               \
    _where << __FILE__ << ':' << __LINE__;                            \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__; \
    std::cerr << _message.str() << "\nAborting...\n";                 \
    cudaDeviceReset();                                                \
    exit(1);                                                          \
  } while (0)

#define checkCUDNN(expression)                                                     \
  {                                                                                \
    cudnnStatus_t status = (expression);                                           \
    if (status != CUDNN_STATUS_SUCCESS) {                                          \
      std::cerr << "Error in file " << __FILE__ << " on line " << __LINE__ << ": " \
                << cudnnGetErrorString(status) << '\n';                            \
      std::exit(EXIT_FAILURE);                                                     \
    }                                                                              \
  }

#define checkCUBLAS(expression)                                                    \
  {                                                                                \
    cublasStatus_t status = (expression);                                          \
    if (status != CUBLAS_STATUS_SUCCESS) {                                         \
      std::cerr << "Error in file " << __FILE__ << " on line " << __LINE__ << ": " \
                << _cudaGetErrorEnum(status) << '\n';                              \
      std::exit(EXIT_FAILURE);                                                     \
    }                                                                              \
  }

#define checkCURAND(expression)                                                    \
  {                                                                                \
    curandStatus_t status = (expression);                                          \
    if (status != CURAND_STATUS_SUCCESS) {                                         \
      std::cerr << "Error in file " << __FILE__ << " on line " << __LINE__ << ": " \
                << _cudaGetErrorEnum(status) << '\n';                              \
      std::exit(EXIT_FAILURE);                                                     \
    }                                                                              \
  }

struct LayerDimension {
  int N, C, H, W;
  int getTotalSize();
};

template <typename T>
__global__ void fillValue(T *v, int size, int value) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size) return;
  v[i] = value;
}

void outOfMemory();

#endif