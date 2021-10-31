#ifndef NEURAL_NET
#define NEURAL_NET

#include <cublas_v2.h>
#include <cudnn.h>
#include <curand.h>

#include <iostream>
#include <string>
#include <vector>

#include "layer_params.cuh"
#include "user_iface.cuh"
#include "utils.cuh"

class NeuralNet
{
public:
  void **layer_input, **dlayer_input, **params;
  int *layer_input_size;
  int *y, *pred_y;
  float *loss;
  float softmax_eps;
  void *one_vec;
  float init_std_dev;

  std::vector<LayerOp> layer_type;
  int num_layers;
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

  float *h_loss;
  int *h_pred_y;

  cudaStream_t stream_compute;

  // void **h_layer_input;

  NeuralNet(std::vector<LayerSpecifier> &layers, DataType data_type,
            int batch_size, TensorFormat tensor_format, long long dropout_seed,
            float softmax_eps, float init_std_dev, UpdateRule update_rule);

  void getLoss(void *X, int *y, double learning_rate,
               std::vector<float> &fwd_vdnn_lag,
               std::vector<float> &bwd_vdnn_lag, bool train = true,
               int *correct_count = NULL, float *loss = NULL);
  void getLoss(void *X, int *y, double learning_rate, bool train = true,
               int *correct_count = NULL, float *loss = NULL);

  void compareOutputCorrect(int *correct_count, int *y);

  float computeLoss();

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