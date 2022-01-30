#include <cublas_v2.h>
#include <cudnn.h>
#include <curand.h>
#include <time.h>

#include <cstdio>
#include <fstream>
#include <string>

#include "gpu_timer.cuh"
#include "neural_net.cuh"

void NeuralNet::getLoss(void *X, int *y, double learning_rate,
                        std::vector<std::pair<size_t, size_t>> &offload_mem, bool train,
                        int *correct_count, float *loss, float *overhead) {
  std::vector<float> t1, t2;
  this->getLoss(X, y, learning_rate, t1, t2, offload_mem, train, correct_count, loss, overhead);
}

void NeuralNet::getLoss(void *X, int *y, double learning_rate, std::vector<float> &fwd_dnn_lag,
                        std::vector<float> &bwd_dnn_lag,
                        std::vector<std::pair<size_t, size_t>> &offload_mem, bool train,
                        int *correct_count, float *scalar_loss, float *overhead) {
  cudaMalloc(&layer_input[0], layer_input_size[0] * data_type_size);
  cudaMemcpy(layer_input[0], X, batch_size * input_channels * input_h * input_w * data_type_size,
             cudaMemcpyHostToDevice);

  if (train == true) {
    cudaMemcpy(this->y, y, batch_size * data_type_size, cudaMemcpyHostToDevice);
  }

  float alpha = 1.0, beta = 0.0;
  float Salpha = 1.0, Sbeta = 0.0;
  double Dalpha = 1.0, Dbeta = 0.0;
  forward_prop(train, offload_mem, alpha, beta, Salpha, Sbeta, Dalpha, Dbeta, fwd_dnn_lag,
               overhead);
  GpuTimer timer;
  timer.start();
  while (!layer_input_pq.empty()) {
    layer_input_pq.pop();
  }
  timer.stop();
  float elapsed = timer.elapsed();
  fwd_dnn_lag.emplace_back(elapsed);
  if (overhead != NULL) {
    *overhead += elapsed;
  }

  if (train == false) {
    compareOutputCorrect(correct_count, y);
    return;
  }

  *scalar_loss = computeLoss();  // Loss Computation

  backward_prop(X, alpha, beta, Salpha, Sbeta, Dalpha, Dbeta, bwd_dnn_lag, overhead, learning_rate);

  for (int k = 0; k < num_layers; k++) {
    if (layer_input[k] != NULL) cudaFree(layer_input[k]);
    if (dlayer_input[k] != NULL) cudaFree(dlayer_input[k]);
  }

  // Make offloaded array to all false
  for (int c = 0; c < num_layers; c++) {
    offloaded[c] = false;
  }
}
