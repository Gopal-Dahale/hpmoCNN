#ifndef NEURAL_NET
#define NEURAL_NET

#include <cublas_v2.h>
#include <cudnn.h>
#include <curand.h>

#include <iostream>
#include <string>
#include <vector>
#include <queue>
#include <utility>

#include "layer_params.cuh"
#include "user_iface.cuh"
#include "utils.cuh"

struct comp {
    constexpr bool operator()(
        pair<size_t, int> const& a,
        pair<size_t, int> const& b)
        const noexcept
    {
        return (a.first < b.first || a.second > b.second);
    }
};

class NeuralNet
{
public:
  void **layer_input, **dlayer_input, **params, **h_layer_input;
  int *layer_input_size;
  int *y, *pred_y;
  float *loss;
  float softmax_eps;
  void *one_vec;
  float init_std_dev;
  priority_queue<pair, vector<pair<size_t,int>>, comp> layer_input_pq;
  std::vector<LayerOp> layer_type;
  int num_layers;
  bool *offloaded;
  cudnnHandle_t cudnn_handle;
  cublasHandle_t cublas_handle;
  curandGenerator_t curand_gen;

  cudnnDataType_t data_type;
  size_t data_type_size;
  cudnnTensorFormat_t tensor_format;
  int batch_size;

  size_t init_free_bytes, free_bytes, total_bytes;
  size_t workspace_size;
  void *workspace;

  int input_channels, input_h, input_w;
  int num_classes;

  cudaStream_t stream_compute, stream_memory;

  // void **h_layer_input;

  NeuralNet();

  NeuralNet(std::vector<LayerSpecifier> &layers, DataType data_type,
            int batch_size, TensorFormat tensor_format, float softmax_eps,
            float init_std_dev, UpdateRule update_rule);

  void getLoss(void *X, int *y, double learning_rate,
               std::vector<float> &fwd_vdnn_lag,
               std::vector<float> &bwd_vdnn_lag, bool train = true,
               int *correct_count = NULL, float *loss = NULL,bool d0o=false);
  void getLoss(void *X, int *y, double learning_rate, bool train = true,
               int *correct_count = NULL, float *loss = NULL,bool d0o=false);

  void compareOutputCorrect(int *correct_count, int *y);

  float computeLoss();

  void save(std::string path);
  void load(std::string path);

  // data of time
  cudaEvent_t start_compute, stop_compute;
  //   void getComputationTime(void *X, int *y, double learning_rate,
  //                           std::vector<float> &fwd_computation_time,
  //                           std::vector<float> &bwd_computation_time);
  cudaEvent_t start_transfer, stop_transfer;
  //   void getTransferTime(void *X, int *y, double learning_rate,
  //   std::vector<float> &fwd_transfer_time,
  //                        std::vector<float> &bwd_transfer_time);
};

#endif
