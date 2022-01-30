#ifndef VGG_116
#define VGG_116
#include <vector>

#include "neural_net.cuh"

class VGG116 {
 public:
  std::vector<LayerSpecifier> layer_specifier;
  VGG116() {
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
      ConvDescriptor part0_conv2;
      part0_conv2.initializeValues(64, 64, 3, 3, 224, 224, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part0_conv2;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part0_conv3;
      part0_conv3.initializeValues(64, 64, 3, 3, 224, 224, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part0_conv3;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part0_conv4;
      part0_conv4.initializeValues(64, 64, 3, 3, 224, 224, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part0_conv4;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part0_conv5;
      part0_conv5.initializeValues(64, 64, 3, 3, 224, 224, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part0_conv5;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part0_conv6;
      part0_conv6.initializeValues(64, 64, 3, 3, 224, 224, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part0_conv6;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part0_conv7;
      part0_conv7.initializeValues(64, 64, 3, 3, 224, 224, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part0_conv7;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part0_conv8;
      part0_conv8.initializeValues(64, 64, 3, 3, 224, 224, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part0_conv8;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part0_conv9;
      part0_conv9.initializeValues(64, 64, 3, 3, 224, 224, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part0_conv9;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part0_conv10;
      part0_conv10.initializeValues(64, 64, 3, 3, 224, 224, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part0_conv10;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part0_conv11;
      part0_conv11.initializeValues(64, 64, 3, 3, 224, 224, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part0_conv11;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part0_conv12;
      part0_conv12.initializeValues(64, 64, 3, 3, 224, 224, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part0_conv12;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part0_conv13;
      part0_conv13.initializeValues(64, 64, 3, 3, 224, 224, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part0_conv13;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part0_conv14;
      part0_conv14.initializeValues(64, 64, 3, 3, 224, 224, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part0_conv14;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part0_conv15;
      part0_conv15.initializeValues(64, 64, 3, 3, 224, 224, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part0_conv15;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part0_conv16;
      part0_conv16.initializeValues(64, 64, 3, 3, 224, 224, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part0_conv16;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part0_conv17;
      part0_conv17.initializeValues(64, 64, 3, 3, 224, 224, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part0_conv17;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part0_conv18;
      part0_conv18.initializeValues(64, 64, 3, 3, 224, 224, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part0_conv18;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part0_conv19;
      part0_conv19.initializeValues(64, 64, 3, 3, 224, 224, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part0_conv19;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part0_conv20;
      part0_conv20.initializeValues(64, 64, 3, 3, 224, 224, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part0_conv20;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part0_conv21;
      part0_conv21.initializeValues(64, 64, 3, 3, 224, 224, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part0_conv21;
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
      ConvDescriptor part1_conv2;
      part1_conv2.initializeValues(128, 128, 3, 3, 112, 112, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part1_conv2;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part1_conv3;
      part1_conv3.initializeValues(128, 128, 3, 3, 112, 112, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part1_conv3;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part1_conv4;
      part1_conv4.initializeValues(128, 128, 3, 3, 112, 112, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part1_conv4;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part1_conv5;
      part1_conv5.initializeValues(128, 128, 3, 3, 112, 112, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part1_conv5;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part1_conv6;
      part1_conv6.initializeValues(128, 128, 3, 3, 112, 112, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part1_conv6;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part1_conv7;
      part1_conv7.initializeValues(128, 128, 3, 3, 112, 112, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part1_conv7;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part1_conv8;
      part1_conv8.initializeValues(128, 128, 3, 3, 112, 112, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part1_conv8;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part1_conv9;
      part1_conv9.initializeValues(128, 128, 3, 3, 112, 112, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part1_conv9;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part1_conv10;
      part1_conv10.initializeValues(128, 128, 3, 3, 112, 112, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part1_conv10;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part1_conv11;
      part1_conv11.initializeValues(128, 128, 3, 3, 112, 112, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part1_conv11;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part1_conv12;
      part1_conv12.initializeValues(128, 128, 3, 3, 112, 112, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part1_conv12;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part1_conv13;
      part1_conv13.initializeValues(128, 128, 3, 3, 112, 112, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part1_conv13;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part1_conv14;
      part1_conv14.initializeValues(128, 128, 3, 3, 112, 112, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part1_conv14;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part1_conv15;
      part1_conv15.initializeValues(128, 128, 3, 3, 112, 112, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part1_conv15;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part1_conv16;
      part1_conv16.initializeValues(128, 128, 3, 3, 112, 112, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part1_conv16;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part1_conv17;
      part1_conv17.initializeValues(128, 128, 3, 3, 112, 112, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part1_conv17;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part1_conv18;
      part1_conv18.initializeValues(128, 128, 3, 3, 112, 112, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part1_conv18;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part1_conv19;
      part1_conv19.initializeValues(128, 128, 3, 3, 112, 112, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part1_conv19;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part1_conv20;
      part1_conv20.initializeValues(128, 128, 3, 3, 112, 112, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part1_conv20;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part1_conv21;
      part1_conv21.initializeValues(128, 128, 3, 3, 112, 112, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part1_conv21;
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
      ConvDescriptor part2_conv3;
      part2_conv3.initializeValues(256, 256, 3, 3, 56, 56, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part2_conv3;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part2_conv4;
      part2_conv4.initializeValues(256, 256, 3, 3, 56, 56, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part2_conv4;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part2_conv5;
      part2_conv5.initializeValues(256, 256, 3, 3, 56, 56, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part2_conv5;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part2_conv6;
      part2_conv6.initializeValues(256, 256, 3, 3, 56, 56, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part2_conv6;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part2_conv7;
      part2_conv7.initializeValues(256, 256, 3, 3, 56, 56, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part2_conv7;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part2_conv8;
      part2_conv8.initializeValues(256, 256, 3, 3, 56, 56, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part2_conv8;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part2_conv9;
      part2_conv9.initializeValues(256, 256, 3, 3, 56, 56, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part2_conv9;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part2_conv10;
      part2_conv10.initializeValues(256, 256, 3, 3, 56, 56, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part2_conv10;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part2_conv11;
      part2_conv11.initializeValues(256, 256, 3, 3, 56, 56, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part2_conv11;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part2_conv12;
      part2_conv12.initializeValues(256, 256, 3, 3, 56, 56, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part2_conv12;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part2_conv13;
      part2_conv13.initializeValues(256, 256, 3, 3, 56, 56, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part2_conv13;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part2_conv14;
      part2_conv14.initializeValues(256, 256, 3, 3, 56, 56, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part2_conv14;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part2_conv15;
      part2_conv15.initializeValues(256, 256, 3, 3, 56, 56, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part2_conv15;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part2_conv16;
      part2_conv16.initializeValues(256, 256, 3, 3, 56, 56, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part2_conv16;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part2_conv17;
      part2_conv17.initializeValues(256, 256, 3, 3, 56, 56, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part2_conv17;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part2_conv18;
      part2_conv18.initializeValues(256, 256, 3, 3, 56, 56, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part2_conv18;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part2_conv19;
      part2_conv19.initializeValues(256, 256, 3, 3, 56, 56, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part2_conv19;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part2_conv20;
      part2_conv20.initializeValues(256, 256, 3, 3, 56, 56, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part2_conv20;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part2_conv21;
      part2_conv21.initializeValues(256, 256, 3, 3, 56, 56, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part2_conv21;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part2_conv22;
      part2_conv22.initializeValues(256, 256, 3, 3, 56, 56, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part2_conv22;
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
      ConvDescriptor part3_conv3;
      part3_conv3.initializeValues(512, 512, 3, 3, 28, 28, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part3_conv3;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part3_conv4;
      part3_conv4.initializeValues(512, 512, 3, 3, 28, 28, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part3_conv4;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part3_conv5;
      part3_conv5.initializeValues(512, 512, 3, 3, 28, 28, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part3_conv5;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part3_conv6;
      part3_conv6.initializeValues(512, 512, 3, 3, 28, 28, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part3_conv6;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part3_conv7;
      part3_conv7.initializeValues(512, 512, 3, 3, 28, 28, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part3_conv7;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part3_conv8;
      part3_conv8.initializeValues(512, 512, 3, 3, 28, 28, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part3_conv8;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part3_conv9;
      part3_conv9.initializeValues(512, 512, 3, 3, 28, 28, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part3_conv9;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part3_conv10;
      part3_conv10.initializeValues(512, 512, 3, 3, 28, 28, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part3_conv10;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part3_conv11;
      part3_conv11.initializeValues(512, 512, 3, 3, 28, 28, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part3_conv11;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part3_conv12;
      part3_conv12.initializeValues(512, 512, 3, 3, 28, 28, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part3_conv12;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part3_conv13;
      part3_conv13.initializeValues(512, 512, 3, 3, 28, 28, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part3_conv13;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part3_conv14;
      part3_conv14.initializeValues(512, 512, 3, 3, 28, 28, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part3_conv14;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part3_conv15;
      part3_conv15.initializeValues(512, 512, 3, 3, 28, 28, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part3_conv15;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part3_conv16;
      part3_conv16.initializeValues(512, 512, 3, 3, 28, 28, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part3_conv16;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part3_conv17;
      part3_conv17.initializeValues(512, 512, 3, 3, 28, 28, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part3_conv17;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part3_conv18;
      part3_conv18.initializeValues(512, 512, 3, 3, 28, 28, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part3_conv18;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part3_conv19;
      part3_conv19.initializeValues(512, 512, 3, 3, 28, 28, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part3_conv19;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part3_conv20;
      part3_conv20.initializeValues(512, 512, 3, 3, 28, 28, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part3_conv20;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part3_conv21;
      part3_conv21.initializeValues(512, 512, 3, 3, 28, 28, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part3_conv21;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part3_conv22;
      part3_conv22.initializeValues(512, 512, 3, 3, 28, 28, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part3_conv22;
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
      ConvDescriptor part4_conv3;
      part4_conv3.initializeValues(512, 512, 3, 3, 14, 14, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part4_conv3;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part4_conv4;
      part4_conv4.initializeValues(512, 512, 3, 3, 14, 14, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part4_conv4;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part4_conv5;
      part4_conv5.initializeValues(512, 512, 3, 3, 14, 14, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part4_conv5;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part4_conv6;
      part4_conv6.initializeValues(512, 512, 3, 3, 14, 14, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part4_conv6;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part4_conv7;
      part4_conv7.initializeValues(512, 512, 3, 3, 14, 14, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part4_conv7;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part4_conv8;
      part4_conv8.initializeValues(512, 512, 3, 3, 14, 14, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part4_conv8;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part4_conv9;
      part4_conv9.initializeValues(512, 512, 3, 3, 14, 14, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part4_conv9;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part4_conv10;
      part4_conv10.initializeValues(512, 512, 3, 3, 14, 14, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part4_conv10;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part4_conv11;
      part4_conv11.initializeValues(512, 512, 3, 3, 14, 14, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part4_conv11;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part4_conv12;
      part4_conv12.initializeValues(512, 512, 3, 3, 14, 14, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part4_conv12;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part4_conv13;
      part4_conv13.initializeValues(512, 512, 3, 3, 14, 14, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part4_conv13;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part4_conv14;
      part4_conv14.initializeValues(512, 512, 3, 3, 14, 14, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part4_conv14;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part4_conv15;
      part4_conv15.initializeValues(512, 512, 3, 3, 14, 14, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part4_conv15;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part4_conv16;
      part4_conv16.initializeValues(512, 512, 3, 3, 14, 14, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part4_conv16;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part4_conv17;
      part4_conv17.initializeValues(512, 512, 3, 3, 14, 14, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part4_conv17;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part4_conv18;
      part4_conv18.initializeValues(512, 512, 3, 3, 14, 14, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part4_conv18;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part4_conv19;
      part4_conv19.initializeValues(512, 512, 3, 3, 14, 14, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part4_conv19;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part4_conv20;
      part4_conv20.initializeValues(512, 512, 3, 3, 14, 14, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part4_conv20;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part4_conv21;
      part4_conv21.initializeValues(512, 512, 3, 3, 14, 14, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part4_conv21;
      layer_specifier.push_back(temp);
    }
    {
      ConvDescriptor part4_conv22;
      part4_conv22.initializeValues(512, 512, 3, 3, 14, 14, 1, 1, 1, 1, RELU);
      LayerSpecifier temp;
      temp.initPointer(CONV);
      *((ConvDescriptor *)temp.params) = part4_conv22;
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
      part5_fc2.initializeValues(4096, 1000, RELU);
      LayerSpecifier temp;
      temp.initPointer(FULLY_CONNECTED);
      *((FCDescriptor *)temp.params) = part5_fc2;
      layer_specifier.push_back(temp);
    }
    {
      FCDescriptor part5_fc3;
      part5_fc3.initializeValues(1000, 100, RELU);
      LayerSpecifier temp;
      temp.initPointer(FULLY_CONNECTED);
      *((FCDescriptor *)temp.params) = part5_fc3;
      layer_specifier.push_back(temp);
    }
    {
      FCDescriptor part5_fc3;
      part5_fc3.initializeValues(100, 10, RELU);
      LayerSpecifier temp;
      temp.initPointer(FULLY_CONNECTED);
      *((FCDescriptor *)temp.params) = part5_fc3;
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