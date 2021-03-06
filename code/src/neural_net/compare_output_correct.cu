#include <cstdio>
#include <cublas_v2.h>
#include <cudnn.h>
#include <curand.h>
#include <iostream>
#include <string>
#include <time.h>

#include "neural_net.cuh"

template <typename T>
__global__ void inferClass(T *O, int *pred_y, int batch_size, int num_classes)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= batch_size)
    return;

  T max = O[i * num_classes];
  int index = 0;
  for (int j = 1; j < num_classes; j++)
  {
    if (O[i * num_classes + j] > max)
    {
      max = O[i * num_classes + j];
      index = j;
    }
  }
  pred_y[i] = index;
}

void NeuralNet::compareOutputCorrect(int *correct_count, int *y)
{
  *correct_count = 0;
  int tempf=0,tempd=0;
  static int p = 0;
  if (data_type == CUDNN_DATA_FLOAT)
  {
    float *typecast_O = (float *)layer_input[num_layers - 1];
    inferClass<float><<<ceil(1.0 * batch_size / BW), BW>>>(
        typecast_O, pred_y, batch_size, num_classes);
    cudaMemPrefetchAsync((int *)pred_y, batch_size, cudaCpuDeviceId);
    cudaMemPrefetchAsync((int *)y, batch_size, cudaCpuDeviceId);
    cudaDeviceSynchronize();
    for (int i = 0; i < batch_size; i++)
    {
      if (pred_y[i] == y[i])
      {
        *correct_count = *correct_count + 1;
        tempf+=1;
      }
    }
  }
  else if (data_type == CUDNN_DATA_DOUBLE)
  {
    double *typecast_O = (double *)layer_input[num_layers - 1];
    inferClass<double><<<ceil(1.0 * batch_size / BW), BW>>>(
        typecast_O, pred_y, batch_size, num_classes);
    cudaDeviceSynchronize();
    for (int i = 0; i < batch_size; i++)
    {
      if (pred_y[i] == y[i])
      {
        *correct_count = *correct_count + 1;
        tempd+=1;
      }
    }
    std::cout << "tempd: "  << tempd << "\n";
  }
  p++;
}
