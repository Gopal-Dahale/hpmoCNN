#include <cuda.h>

#include <iostream>
#include <vector>

#include "hpmo_cnn.cuh"

using namespace std;

int main() {
  vector<LayerSpecifier> layer_specifier;
  {
    ConvDescriptor layer0;
    layer0.initializeValues(1, 64, 3, 3, 28, 28, 1, 1, 1, 1);
    LayerSpecifier temp;
    temp.initPointer(CONV);
    *((ConvDescriptor *)temp.params) = layer0;
    layer_specifier.push_back(temp);
  }
  {
    ConvDescriptor layer1;
    layer1.initializeValues(1, 64, 3, 3, 28, 28, 1, 1, 1, 1);
    LayerSpecifier temp;
    temp.initPointer(CONV);
    *((ConvDescriptor *)temp.params) = layer1;
    layer_specifier.push_back(temp);
  }
  {
    PoolingDescriptor layer2;
    layer2.initializeValues(64, 2, 2, 28, 28, 0, 0, 2, 2, POOLING_MAX);
    LayerSpecifier temp;
    temp.initPointer(POOLING);
    *((PoolingDescriptor *)temp.params) = layer2;
    layer_specifier.push_back(temp);
  }
  {
    FCDescriptor layer3;
    layer3.initializeValues(64 * 64 * (28 / 2), 128);
    LayerSpecifier temp;
    temp.initPointer(FULLY_CONNECTED);
    *((FCDescriptor *)temp.params) = layer3;
    layer_specifier.push_back(temp);
  }
  {
    FCDescriptor layer4;
    layer4.initializeValues(128, 10);
    LayerSpecifier temp;
    temp.initPointer(FULLY_CONNECTED);
    *((FCDescriptor *)temp.params) = layer4;
    layer_specifier.push_back(temp);
  }
  {
    SoftmaxDescriptor layer5;
    layer5.initializeValues(SOFTMAX_ACCURATE, SOFTMAX_MODE_INSTANCE, 1000, 1, 1);
    LayerSpecifier temp;
    temp.initPointer(SOFTMAX);
    *((SoftmaxDescriptor *)temp.params) = layer5;
    layer_specifier.push_back(temp);
  }

  int batch_size = 128;
  float softmax_eps = 1e-8;
  float init_std_dev = 0.1;

  NeuralNet(layer_specifier, DATA_FLOAT, batch_size, TENSOR_NCHW, softmax_eps, init_std_dev);

  return 0;
}
