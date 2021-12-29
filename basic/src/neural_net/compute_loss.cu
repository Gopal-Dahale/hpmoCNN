#include <cublas_v2.h>
#include <cudnn.h>
#include <curand.h>
#include <time.h>

#include <cstdio>
#include <string>

#include "neural_net.cuh"

template <typename T>
__global__ void computeSoftmaxLoss(T *O, int *y, float *loss, int batch_size,
                                   int num_classes, float eps)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= batch_size)
    return;

  loss[i] = -logf(O[i * num_classes + y[i]] + eps);
}

float NeuralNet::computeLoss()
{
  if (layer_type[num_layers - 1] == SOFTMAX)
  {
    if (data_type == CUDNN_DATA_FLOAT)
      computeSoftmaxLoss<float><<<ceil(1.0 * batch_size / BW), BW>>>(
          (float *)layer_input[num_layers], this->y, loss, batch_size,
          num_classes, softmax_eps);
    else if (data_type == CUDNN_DATA_DOUBLE)
      computeSoftmaxLoss<double><<<ceil(1.0 * batch_size / BW), BW>>>(
          (double *)layer_input[num_layers], this->y, loss, batch_size,
          num_classes, softmax_eps);
  }
 
  cudaMemPrefetchAsync((float *)loss, batch_size, cudaCpuDeviceId);
  float total_loss = 0.0;
  for (int i = 0; i < batch_size; i++)
    total_loss += loss[i];
  return total_loss / batch_size;
}
