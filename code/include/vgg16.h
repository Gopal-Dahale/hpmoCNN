#ifndef VGG_16
#define VGG_16
#include <vector>

class VGG16 {
 public:
  // VGG NET
  std::vector<LayerSpecifier> layer_specifier;
  VGG16() {
    {
      ConvDescriptor part0_conv0;
      part0_conv0.initializeValues(1, 64, 3, 3, 224, 224, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part0_conv0;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part0_conv1;
      part0_conv1.initializeValues(64, 64, 3, 3, 224, 224, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part0_conv1;
      layer_specifier.push_back(temp);
    }
    {
      PoolingDescriptor pool0;
      pool0.initializeValues(64, 2, 2, 224, 224, 0, 0, 2, 2, POOLING_MAX);
      LayerSpecifier temp;
      temp.initPointer(POOLING);
      *((PoolingDescriptor *)temp.params) = pool0;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part1_conv0;
      part1_conv0.initializeValues(64, 128, 3, 3, 112, 112, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part1_conv0;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part1_conv1;
      part1_conv1.initializeValues(128, 128, 3, 3, 112, 112, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part1_conv1;
      layer_specifier.push_back(temp);
    }
    {
      PoolingDescriptor pool1;
      pool1.initializeValues(128, 2, 2, 112, 112, 0, 0, 2, 2, POOLING_MAX);
      LayerSpecifier temp;
      temp.initPointer(POOLING);
      *((PoolingDescriptor *)temp.params) = pool1;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part2_conv0;
      part2_conv0.initializeValues(128, 256, 3, 3, 56, 56, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part2_conv0;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part2_conv1;
      part2_conv1.initializeValues(256, 256, 3, 3, 56, 56, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part2_conv1;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part2_conv2;
      part2_conv2.initializeValues(256, 256, 3, 3, 56, 56, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part2_conv2;
      layer_specifier.push_back(temp);
    }
    {
      PoolingDescriptor pool2;
      pool2.initializeValues(256, 2, 2, 56, 56, 0, 0, 2, 2, POOLING_MAX);
      LayerSpecifier temp;
      temp.initPointer(POOLING);
      *((PoolingDescriptor *)temp.params) = pool2;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part3_conv0;
      part3_conv0.initializeValues(256, 512, 3, 3, 28, 28, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part3_conv0;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part3_conv1;
      part3_conv1.initializeValues(512, 512, 3, 3, 28, 28, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part3_conv1;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part3_conv2;
      part3_conv2.initializeValues(512, 512, 3, 3, 28, 28, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part3_conv2;
      layer_specifier.push_back(temp);
    }
    {
      PoolingDescriptor pool3;
      pool3.initializeValues(512, 2, 2, 28, 28, 0, 0, 2, 2, POOLING_MAX);
      LayerSpecifier temp;
      temp.initPointer(POOLING);
      *((PoolingDescriptor *)temp.params) = pool3;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part4_conv0;
      part4_conv0.initializeValues(512, 512, 3, 3, 14, 14, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part4_conv0;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part4_conv1;
      part4_conv1.initializeValues(512, 512, 3, 3, 14, 14, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part4_conv1;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part4_conv2;
      part4_conv2.initializeValues(512, 512, 3, 3, 14, 14, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part4_conv2;
      layer_specifier.push_back(temp);
    }
    {
      PoolingDescriptor pool3;
      pool3.initializeValues(512, 2, 2, 14, 14, 0, 0, 2, 2, POOLING_MAX);
      LayerSpecifier temp;
      temp.initPointer(POOLING);
      *((PoolingDescriptor *)temp.params) = pool3;
      layer_specifier.push_back(temp);
    }

    {
      FCDescriptor part5_fc0;
      part5_fc0.initializeValues(7 * 7 * 512, 4096, RELU);
      LayerSpecifier temp;
      temp.initPointer(FULLY_CONNECTED);
      *((FCDescriptor *)temp.params) = part5_fc0;
      layer_specifier.push_back(temp);
    }
    {
      FCDescriptor part5_fc1;
      part5_fc1.initializeValues(4096, 4096, RELU);
      LayerSpecifier temp;
      temp.initPointer(FULLY_CONNECTED);
      *((FCDescriptor *)temp.params) = part5_fc1;
      layer_specifier.push_back(temp);
    }
    {
      FCDescriptor part5_fc2;
      part5_fc2.initializeValues(4096, 10);
      LayerSpecifier temp;
      temp.initPointer(FULLY_CONNECTED);
      *((FCDescriptor *)temp.params) = part5_fc2;
      layer_specifier.push_back(temp);
    }
    {
      SoftmaxDescriptor s_max;
      s_max.initializeValues(SOFTMAX_ACCURATE, SOFTMAX_MODE_INSTANCE, 10, 1, 1);
      LayerSpecifier temp;
      temp.initPointer(SOFTMAX);
      *((SoftmaxDescriptor *)temp.params) = s_max;
      layer_specifier.push_back(temp);
    }
  }
};

#endif