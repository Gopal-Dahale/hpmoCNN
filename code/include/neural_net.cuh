#ifndef NEURAL_NET
#define NEURAL_NET

#include <cublas_v2.h>
#include <cudnn.h>
#include <curand.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

#include <fstream>
#include <iostream>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "layer_params.cuh"
#include "user_iface.cuh"
#include "utils.cuh"

struct comp {
  constexpr bool operator()(std::pair<size_t, int> const &a,
                            std::pair<size_t, int> const &b) const noexcept {
    return (a.first < b.first || a.second > b.second);
  }
};

class NeuralNet {
 private:
  void init_layers(std::vector<LayerSpecifier> &, UpdateRule);
  void allocate_mem_for_layers(std::vector<LayerSpecifier> &);
  void allocate_workspace_for_layers(std::vector<LayerSpecifier> &);
  void max_heap_policy(std::vector<std::pair<size_t, size_t>> &, std::vector<int> &, int &);
  void conv_forward(int &, float &, float &);
  void fc_forward(int &, float &, float &, float &, float &, double &, double &);
  void pooling_forward(int &, float &, float &);
  void activation_forward(int &, float &, float &);
  void softmax_forward(int &, float &, float &);
  void forward_prop(bool &, std::vector<std::pair<size_t, size_t>> &, float &, float &, float &,
                    float &, double &, double &, std::vector<float> &, float *);
  void backward_prop(void *, float &, float &, float &, float &, double &, double &,
                     std::vector<float> &, float *, double &);
  void softmax_loss_backward();
  void prefetch_policy(int &, void *);
  void conv_backward(int &, float &, float &, double &);
  void fc_backward(int &, float &, float &, float &, float &, double &, double &, double &);
  void pooling_backward(int &, float &, float &);
  void activation_backward(int &, float &, float &);
  void softmax_backward(int &, float &, float &);

 public:
  std::ofstream mem_usage;
  void **layer_input, **dlayer_input, **params, **h_layer_input;
  int *layer_input_size;
  int *y, *pred_y;
  float *loss;
  float softmax_eps;
  void *one_vec;
  float init_std_dev;
  std::priority_queue<std::pair<size_t, int>, std::vector<std::pair<size_t, int>>, comp>
      layer_input_pq;
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

  NeuralNet();

  NeuralNet(std::vector<LayerSpecifier> &layers, DataType data_type, int batch_size,
            TensorFormat tensor_format, float softmax_eps, float init_std_dev,
            UpdateRule update_rule);
  void getLoss(void *X, int *y, double learning_rate, std::vector<float> &fwd_vdnn_lag,
               std::vector<float> &bwd_vdnn_lag,
               std::vector<std::pair<size_t, size_t>> &offload_mem, bool train = true,
               int *correct_count = NULL, float *loss = NULL, float *overhead = NULL);
  void getLoss(void *X, int *y, double learning_rate,
               std::vector<std::pair<size_t, size_t>> &offload_mem, bool train = true,
               int *correct_count = NULL, float *loss = NULL, float *overhead = NULL);
  void compareOutputCorrect(int *correct_count, int *y);
  float computeLoss();
  void save(std::string path);
  void load(std::string path);
};

#endif
