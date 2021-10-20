#ifndef NEURAL_NET
#define NEURAL_NET

#include <cudnn.h>

#include <vector>

#include "./user_interface.cuh"

using namespace std;

class NeuralNet {
 public:
  cudaStream_t stream_compute, stream_memory;
  NeuralNet(vector<LayerSpecifier> &layers, DataType data_type, int batch_size,
            TensorFormat tensor_format, float softmax_eps, float init_std_dev);
};

#endif