/**
 * Upsampled MNIST Dataset
 * VGG Net
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "cxxopts.hpp"
#include "solver.cuh"

using namespace std;

typedef unsigned char uchar;

int num_train = 1024, num_test = 512;

void save_metrics(vector<float> &loss, vector<int> &val_acc,
                  vector<float> &batch_times,
                  unordered_map<string, double> &configs, float total_train_time, float overhead) {
  fstream f;

  f.open("../batch_times.txt", ios::out);
  for (auto &i : batch_times) f << i << endl;
  f.close();

  f.open("../loss.txt", ios::out);
  for (auto &i : loss) f << i << endl;
  f.close();

  f.open("../val_acc.txt", ios::out);
  for (auto &i : val_acc) f << i << endl;
  f.close();

  f.open("../configs.txt", ios::out);
  for (auto &i : configs) f << i.first << " " << i.second << endl;
  f.close();

  f.open("../totaltime.txt", ios::out);
  f << total_train_time << endl;
  f.close();

  f.open("../totaloverhead.txt", ios::out);
  f << overhead << endl;
  f.close();

}

void save_mem_usage(NeuralNet* net)
{
  std::ofstream mem_usage;
  mem_usage.open("../mem_usage.txt");

  for (int c = 0; c < net->num_layers + 1; c++)
  {
    size_t feature_map_size, fwd_workspace_size = 0, bwd_workspace_filter = 0, bwd_workspace_data = 0, weights = 0;
    feature_map_size = net->layer_input_size[c] * net->data_type_size;
    if (c != net->num_layers && net->layer_type[c] == CONV)
    {
      ConvLayerParams *cur_params = (ConvLayerParams *)net->params[c];
      fwd_workspace_size = cur_params->fwd_workspace_size;
      bwd_workspace_filter = cur_params->bwd_filter_workspace_size;
      bwd_workspace_data = cur_params->bwd_data_workspace_size;
      weights = cur_params->kernel_size * net->data_type_size;
    }
    else if (c != net->num_layers && net->layer_type[c] == FULLY_CONNECTED)
    {
      FCLayerParams *cur_params = (FCLayerParams *)net->params[c];
      int wt_alloc_size = cur_params->weight_matrix_size;
      if (wt_alloc_size % 2 != 0)
        wt_alloc_size += 1;
      weights = (wt_alloc_size + cur_params->C_out) * net->data_type_size;
    }
    mem_usage << feature_map_size << " " << fwd_workspace_size << " " << bwd_workspace_filter << " " << bwd_workspace_data << " " << weights << "\n";
    // std::cout << feature_map_size << " " << fwd_workspace_size << " " << bwd_workspace_filter << " " << bwd_workspace_data << " " << weights << "\n";
    // total_feature_map_size += layer_input_size[c] * data_type_size;
  }
  mem_usage.close();
}

int reverseInt(int n) {
  const int bytes = 4;
  unsigned char ch[bytes];
  for (int i = 0; i < bytes; i++) {
    ch[i] = (n >> i * 8) & 255;
  }
  int p = 0;
  for (int i = 0; i < bytes; i++) {
    p += (int)ch[i] << (bytes - i - 1) * 8;
  }
  return p;
}

void readMNIST224(vector<vector<uchar>> &train_images,
                  vector<vector<uchar>> &test_images,
                  vector<uchar> &train_labels, vector<uchar> &test_labels,
                  int num_train, int num_test) {
  string filename_train_images =
      "/kaggle/input/mnist224by224testdataset/train-images-224by224-";
  string filename_train_labels = "data/train-labels.idx1-ubyte";

  string filename_test_images =
      "/kaggle/input/mnist224by224testdataset/test-images-224by224-";
  string filename_test_labels = "data/t10k-labels.idx1-ubyte";

  // read train/test images
  int images_per_file = 2000;
  int num_train_files =
      min((int)(ceil(num_train / float(images_per_file))), 30);
  int num_test_files = min((int)(ceil(num_test / float(images_per_file))), 5);

  for (int i = 0; i < 2; i++) {
    int num_files = (i == 0 ? num_train_files : num_test_files);
    for (int j = 0; j < num_files; j++) {
      string filename;
      if (i == 0)
        filename = filename_train_images;
      else
        filename = filename_test_images;
      filename = filename + to_string(j) + ".idx3-ubyte";

      ifstream f(filename.c_str(), ios::binary);
      if (!f.is_open()) printf("Cannot read MNIST from %s\n", filename.c_str());

      // read metadata
      int magic_number = 0, n_images = 0, n_rows = 0, n_cols = 0;
      f.read((char *)&magic_number, sizeof(magic_number));
      magic_number = reverseInt(magic_number);
      f.read((char *)&n_images, sizeof(n_images));
      n_images = reverseInt(n_images);
      f.read((char *)&n_rows, sizeof(n_rows));
      n_rows = reverseInt(n_rows);
      f.read((char *)&n_cols, sizeof(n_cols));
      n_cols = reverseInt(n_cols);

      for (int k = 0; k < n_images; k++) {
        vector<uchar> temp;
        temp.reserve(n_rows * n_cols);
        for (int j = 0; j < n_rows * n_cols; j++) {
          uchar t = 0;
          f.read((char *)&t, sizeof(t));
          temp.push_back(t);
        }
        if (i == 0) {
          train_images.push_back(temp);
          if ((j * n_images + k + 1) >= num_train) break;
        } else {
          test_images.push_back(temp);
          if ((j * n_images + k + 1) >= num_test) break;
        }
      }
      f.close();
    }
  }

  // read train/test labels
  for (int i = 0; i < 2; i++) {
    string filename;
    if (i == 0)
      filename = filename_train_labels;
    else
      filename = filename_test_labels;

    ifstream f(filename.c_str(), ios::binary);
    if (!f.is_open()) printf("Cannot read MNIST from %s\n", filename.c_str());

    // read metadata
    int magic_number = 0, n_labels = 0;
    f.read((char *)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);
    f.read((char *)&n_labels, sizeof(n_labels));
    n_labels = reverseInt(n_labels);

    if (i == 0)
      n_labels = min(n_labels, num_train);
    else
      n_labels = min(n_labels, num_test);

    for (int k = 0; k < n_labels; k++) {
      uchar t = 0;
      f.read((char *)&t, sizeof(t));
      if (i == 0)
        train_labels.push_back(t);
      else
        test_labels.push_back(t);
    }
    f.close();
  }

  assert(train_images.size() == train_labels.size());
  assert(test_images.size() == test_labels.size());
}

int main(int argc, char *argv[]) {
  /******************* Parse command line arguments ********************/
  cxxopts::Options options(
      "hpmoCNN",
      "High Performance Memory Optimal Convolutional Neural Network");

  options.add_options()(
      "batch-size", "Batch Size",
      cxxopts::value<int>()->default_value("64"))  // a bool parameter
      ("softmax-eps", "softmax eps",
       cxxopts::value<float>()->default_value("1e-8"))(
          "init-std-dev", "initial standard deviation",
          cxxopts::value<float>()->default_value("0.01"))(
          "epochs", "Number of epochs",
          cxxopts::value<int>()->default_value("5"))(
          "learning-rate", "Learning Rate",
          cxxopts::value<double>()->default_value("0.01"))(
          "learning-rate-decay", "Learning Rate Decay",
          cxxopts::value<double>()->default_value("1"))(
          "num-train", "Number of training examples to use",
          cxxopts::value<int>()->default_value("1024"))(
          "num-test", "Number of testing examples to use",
          cxxopts::value<int>()->default_value("512"))("help", "Print Usage");

  auto result = options.parse(argc, argv);
  if (result.count("help")) {
    std::cout << options.help() << std::endl;
    exit(0);
  }

  num_train = result["num-train"].as<int>();
  num_test = result["num-test"].as<int>();

  /******************* Read Dataset ************************************/

  bool doo = false;
  int rows = 224, cols = 224, channels = 1;
  vector<vector<uchar>> train_images, test_images;
  vector<uchar> train_labels, test_labels;

  cout << "Reading MNIST dataset...  ";

  readMNIST224(train_images, test_images, train_labels, test_labels, num_train,
               num_test);

  assert(train_images.size() == train_labels.size());
  assert(test_images.size() == test_labels.size());

  float *f_train_images, *f_test_images;
  int *f_train_labels, *f_test_labels;

  int input_size = rows * cols * channels;
  f_train_images = (float *)malloc(num_train * input_size * sizeof(float));
  f_train_labels = (int *)malloc(num_train * sizeof(int));
  f_test_images = (float *)malloc(num_test * input_size * sizeof(float));
  f_test_labels = (int *)malloc(num_test * sizeof(int));

  for (int k = 0; k < num_train; k++) {
    for (int j = 0; j < input_size; j++)
      f_train_images[k * input_size + j] = (float)train_images[k][j];
    f_train_labels[k] = (int)train_labels[k];
  }

  for (int k = 0; k < num_test; k++) {
    for (int j = 0; j < input_size; j++)
      f_test_images[k * input_size + j] = (float)test_images[k][j];
    f_test_labels[k] = (int)test_labels[k];
  }

  float *mean_image;
  mean_image = (float *)malloc(input_size * sizeof(float));

  for (int i = 0; i < input_size; i++) {
    mean_image[i] = 0;
    for (int k = 0; k < num_train; k++)
      mean_image[i] += f_train_images[k * input_size + i];
    mean_image[i] /= num_train;
  }

  for (int i = 0; i < num_train; i++) {
    for (int j = 0; j < input_size; j++)
      f_train_images[i * input_size + j] -= mean_image[j];
  }

  for (int i = 0; i < input_size; i++) {
    mean_image[i] = 0;
    for (int k = 0; k < num_test; k++)
      mean_image[i] += f_test_images[k * input_size + i];
    mean_image[i] /= num_test;
  }

  for (int i = 0; i < num_test; i++) {
    for (int j = 0; j < input_size; j++)
      f_test_images[i * input_size + j] -= mean_image[j];
  }

  cout << "Done" << endl;

  /******************* VGG NET ************************************/
  vector<LayerSpecifier> layer_specifier;
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

  /**************************** Configuration ****************************/
  int batch_size = result["batch-size"].as<int>();
  float softmax_eps = result["softmax-eps"].as<float>();
  float init_std_dev = result["init-std-dev"].as<float>();
  int num_epoch = result["epochs"].as<int>();
  double learning_rate = result["learning-rate"].as<double>();
  double learning_rate_decay = result["learning-rate-decay"].as<double>();

  /************************ Display configuration *************************/
  unordered_map<string, double> configs = {
      {"batch_size", batch_size},
      {"softmax_eps", softmax_eps},
      {"init_std_dev", init_std_dev},
      {"num_epoch", num_epoch},
      {"learning_rate", learning_rate},
      {"learning_rate_decay", learning_rate_decay},
      {"num_train", num_train},
      {"num_test", num_test}};

  for (auto &config : configs) {
    cout << config.first << ": " << config.second << endl;
  }

  /*************************** Train & Test ***************************/
  NeuralNet net(layer_specifier, DATA_FLOAT, batch_size, TENSOR_NCHW,
                softmax_eps, init_std_dev, SGD);
  Solver solver(&net, (void *)f_train_images, f_train_labels,
                (void *)f_train_images, f_train_labels, num_epoch, SGD,
                learning_rate, learning_rate_decay, num_train, num_train);
  vector<float> loss;
  vector<int> val_acc;
  vector<float> batch_times;
  float milli = 0, overhead = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  solver.train(loss, val_acc, batch_times, &overhead);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop);

  int num_correct;
  
  solver.checkAccuracy(f_train_images, f_train_labels, num_train, &num_correct);
  std::cout << "TRAIN NUM CORRECT:" << num_correct << endl;
  solver.checkAccuracy(f_test_images, f_test_labels, num_test, &num_correct);
  std::cout << "TEST NUM CORRECT:" << num_correct << endl;

  /*************************** Save metrics ***************************/
  save_mem_usage(&net);
  save_metrics(loss, val_acc, batch_times, configs, milli, overhead);
}
