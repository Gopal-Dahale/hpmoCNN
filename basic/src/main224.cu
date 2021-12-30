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

#include "solver.cuh"

using namespace std;

typedef unsigned char uchar;

int num_train = 1000, num_test = 500;

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

void readMNIST(vector<vector<uchar>> &train_images,
               vector<vector<uchar>> &test_images, vector<uchar> &train_labels,
               vector<uchar> &test_labels) {
  string filename_train_images = "data/train-images.idx3-ubyte";
  string filename_train_labels = "data/train-labels.idx1-ubyte";

  string filename_test_images = "data/t10k-images.idx3-ubyte";
  string filename_test_labels = "data/t10k-labels.idx1-ubyte";

  // read train/test images
  for (int i = 0; i < 2; i++) {
    string filename;
    if (i == 0)
      filename = filename_train_images;
    else
      filename = filename_test_images;

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
      if (i == 0)
        train_images.push_back(temp);
      else
        test_images.push_back(temp);
    }
    f.close();
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
}

void readMNIST224(vector<vector<uchar>> &train_images,
                  vector<vector<uchar>> &test_images,
                  vector<uchar> &train_labels, vector<uchar> &test_labels) {
  string filename_train_images =
      "/kaggle/input/mnist224by224testdataset/train-images-224by224-";
  string filename_train_labels = "data/train-labels.idx1-ubyte";

  string filename_test_images =
      "/kaggle/input/mnist224by224testdataset/test-images-224by224-";
  string filename_test_labels = "data/t10k-labels.idx1-ubyte";

  // read train/test images
  for (int i = 0; i < 2; i++) {
    int k = (i == 0 ? 30 : 5);
    for (int j = 0; j < k; j++) {
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
      //       std::cout << "images = " << n_images << " rows = " << n_rows << "
      //       cols = " << n_cols;

      for (int k = 0; k < n_images; k++) {
        vector<uchar> temp;
        temp.reserve(n_rows * n_cols);
        for (int j = 0; j < n_rows * n_cols; j++) {
          uchar t = 0;
          f.read((char *)&t, sizeof(t));
          temp.push_back(t);
        }
        if (i == 0)
          train_images.push_back(temp);
        else
          test_images.push_back(temp);
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

auto create_mini_MNIST(vector<vector<uchar>> &images, vector<uchar> &labels,
                       int size) {
  unordered_map<int, vector<int>> m;
  for (int i = 0; i < labels.size(); i++) m[(int)labels[i]].push_back(i);

  int bucket = size / 10;

  random_device rd;            // Initialize the random_device
  mt19937_64 generator(rd());  // Seed the engine
  set<int> results;
  vector<int> indices;

  for (auto &i : m) {
    // Specify the range of numbers to generate, in this case [min, max]
    uniform_int_distribution<int> dist{0, (int)i.second.size()};

    while (results.size() < bucket) results.insert(dist(generator));

    for (auto &j : results) indices.push_back(i.second[j]);

    results.clear();
  }

  assert(indices.size() == size);

  // shuffle indices array with default random engine
  shuffle(indices.begin(), indices.end(), default_random_engine(rd()));

  vector<vector<uchar>> mini_images;
  vector<uchar> mini_labels;

  // extract data from images and labels vectors using indices vector and put
  // them in mini_images and mini_labels
  for (int i = 0; i < indices.size(); i++) {
    mini_images.push_back(images[indices[i]]);
    mini_labels.push_back(labels[indices[i]]);
  }

  assert(mini_images.size() == size);
  assert(mini_labels.size() == size);

  return make_pair(mini_images, mini_labels);
}

void printTimes(vector<float> &time, string filename);
void printvDNNLag(vector<vector<float>> &fwd_vdnn_lag,
                  vector<vector<float>> &bwd_vdnn_lag, string filename);

int main(int argc, char *argv[]) {
  bool doo = false;
  int rows = 224, cols = 224, channels = 1;
  vector<vector<uchar>> train_images, test_images;
  vector<uchar> train_labels, test_labels;

  //   if (argc == 2 && argv[1][0] == '1')
  //     doo = true;
  //   std::cout << "doo: " << doo << "\n";
  //   auto dataset = cifar::read_dataset<std::vector, std::vector, uchar,
  //   uchar>(1000,500); train_images = dataset.training_images; test_images =
  //   dataset.test_images; train_labels = dataset.training_labels;
  //   test_labels = dataset.test_labels;

  //    cout << train_images.size() << " " << train_images[0].size() << "\n";
  //   for(int i=0;i<10;i++)
  //     {
  //         cout << (uint8_t)dataset.training_labels[i] << "\n";
  //     }
  readMNIST224(train_images, test_images, train_labels, test_labels);

  // auto data = create_mini_MNIST(train_images, train_labels, num_train);
  // train_images = data.first;
  // train_labels = data.second;

  // data = create_mini_MNIST(test_images, test_labels, num_test);
  // test_images = data.first;
  // test_labels = data.second;

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

  // VGG Net
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

  int batch_size = ((argc > 1) ? atoi(argv[1]) : 64);
  float softmax_eps = 1e-8;
  float init_std_dev = 0.01;
  NeuralNet net(layer_specifier, DATA_FLOAT, batch_size, TENSOR_NCHW,
                softmax_eps, init_std_dev, SGD);

  int num_epoch = ((argc > 2) ? atoi(argv[2]) : 10);
  double learning_rate = ((argc > 3) ? atof(argv[3]) : 0.01);
  double learning_rate_decay = 1;

  /*********** Display configuration ***********/

  cout << "batch_size: " << batch_size << endl;
  cout << "num_epoch: " << num_epoch << endl;
  cout << "learning_rate: " << learning_rate << endl;

  Solver solver(&net, (void *)f_train_images, f_train_labels,
                (void *)f_train_images, f_train_labels, num_epoch, SGD,
                learning_rate, learning_rate_decay, num_train, num_train);
  vector<float> loss;
  vector<int> val_acc;
  solver.train(loss, val_acc, doo);
  int num_correct;
  solver.checkAccuracy(f_train_images, f_train_labels, num_train, &num_correct);
  std::cout << "TRAIN NUM CORRECT:" << num_correct << endl;
  solver.checkAccuracy(f_test_images, f_test_labels, num_test, &num_correct);
  std::cout << "TEST NUM CORRECT:" << num_correct << endl;

  /** Store and load model from net object */
  //   net.save("model.txt");
  //   NeuralNet net2;
  //   net2.load("model.txt");
  //   Solver solver2(&net2, (void *)f_train_images, f_train_labels,
  //                  (void *)f_train_images, f_train_labels, num_epoch, SGD,
  //                  learning_rate, learning_rate_decay, num_train, num_train);
  //   solver.checkAccuracy(f_test_images, f_test_labels, num_test,
  //   &num_correct); std::cout << "TEST NUM CORRECT:" << num_correct << endl;

  //   solver.getTrainTime(loss, time, 100, fwd_vdnn_lag, bwd_vdnn_lag);
  //   printTimes(time, filename);
  //   printvDNNLag(fwd_vdnn_lag, bwd_vdnn_lag, filename);
}