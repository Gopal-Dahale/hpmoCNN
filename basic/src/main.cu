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
#include "cifar100_reader.cuh"
#include "solver.cuh"

using namespace std;

typedef unsigned char uchar;

int num_train = 1000, num_test = 500;

int reverseInt(int n)
{
  const int bytes = 4;
  unsigned char ch[bytes];
  for (int i = 0; i < bytes; i++)
  {
    ch[i] = (n >> i * 8) & 255;
  }
  int p = 0;
  for (int i = 0; i < bytes; i++)
  {
    p += (int)ch[i] << (bytes - i - 1) * 8;
  }
  return p;
}

void readMNIST(vector<vector<uchar>> &train_images,
               vector<vector<uchar>> &test_images, vector<uchar> &train_labels,
               vector<uchar> &test_labels)
{
  string filename_train_images = "data/train-images.idx3-ubyte";
  string filename_train_labels = "data/train-labels.idx1-ubyte";

  string filename_test_images = "data/t10k-images.idx3-ubyte";
  string filename_test_labels = "data/t10k-labels.idx1-ubyte";

  // read train/test images
  for (int i = 0; i < 2; i++)
  {
    string filename;
    if (i == 0)
      filename = filename_train_images;
    else
      filename = filename_test_images;

    ifstream f(filename.c_str(), ios::binary);
    if (!f.is_open())
      printf("Cannot read MNIST from %s\n", filename.c_str());

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

    for (int k = 0; k < n_images; k++)
    {
      vector<uchar> temp;
      temp.reserve(n_rows * n_cols);
      for (int j = 0; j < n_rows * n_cols; j++)
      {
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
  for (int i = 0; i < 2; i++)
  {
    string filename;
    if (i == 0)
      filename = filename_train_labels;
    else
      filename = filename_test_labels;

    ifstream f(filename.c_str(), ios::binary);
    if (!f.is_open())
      printf("Cannot read MNIST from %s\n", filename.c_str());

    // read metadata
    int magic_number = 0, n_labels = 0;
    f.read((char *)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);
    f.read((char *)&n_labels, sizeof(n_labels));
    n_labels = reverseInt(n_labels);

    for (int k = 0; k < n_labels; k++)
    {
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

auto create_mini_MNIST(vector<vector<uchar>> &images, vector<uchar> &labels,
                       int size)
{
  unordered_map<int, vector<int>> m;
  for (int i = 0; i < labels.size(); i++)
    m[(int)labels[i]].push_back(i);

  int bucket = size / 10;

  random_device rd;           // Initialize the random_device
  mt19937_64 generator(rd()); // Seed the engine
  set<int> results;
  vector<int> indices;

  for (auto &i : m)
  {
    // Specify the range of numbers to generate, in this case [min, max]
    uniform_int_distribution<int> dist{0, (int)i.second.size()};

    while (results.size() < bucket)
      results.insert(dist(generator));

    for (auto &j : results)
      indices.push_back(i.second[j]);

    results.clear();
  }

  assert(indices.size() == size);

  // shuffle indices array with default random engine
  shuffle(indices.begin(), indices.end(), default_random_engine(rd()));

  vector<vector<uchar>> mini_images;
  vector<uchar> mini_labels;

  // extract data from images and labels vectors using indices vector and put
  // them in mini_images and mini_labels
  for (int i = 0; i < indices.size(); i++)
  {
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

int main(int argc, char *argv[])
{
  int rows = 32, cols = 32, channels = 3;
  vector<vector<uchar>> train_images, test_images;
  vector<uchar> train_labels, test_labels;
  
  auto dataset = cifar::read_dataset<std::vector, std::vector, uchar, uchar>(1000,500);
  train_images = dataset.training_images;
  test_images = dataset.test_images;
  train_labels = dataset.training_images;
  test_labels = dataset.test_labels;
//   readMNIST(train_images, test_images, train_labels, test_labels);

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

  for (int k = 0; k < num_train; k++)
  {
    for (int j = 0; j < input_size; j++)
      f_train_images[k * input_size + j] = (float)train_images[k][j];
    f_train_labels[k] = (int)train_labels[k];
  }

  for (int k = 0; k < num_test; k++)
  {
    for (int j = 0; j < input_size; j++)
      f_test_images[k * input_size + j] = (float)test_images[k][j];
    f_test_labels[k] = (int)test_labels[k];
  }

  float *mean_image;
  mean_image = (float *)malloc(input_size * sizeof(float));

  for (int i = 0; i < input_size; i++)
  {
    mean_image[i] = 0;
    for (int k = 0; k < num_train; k++)
      mean_image[i] += f_train_images[k * input_size + i];
    mean_image[i] /= num_train;
  }

  for (int i = 0; i < num_train; i++)
  {
    for (int j = 0; j < input_size; j++)
      f_train_images[i * input_size + j] -= mean_image[j];
  }

  for (int i = 0; i < input_size; i++)
  {
    mean_image[i] = 0;
    for (int k = 0; k < num_test; k++)
      mean_image[i] += f_test_images[k * input_size + i];
    mean_image[i] /= num_test;
  }

  for (int i = 0; i < num_test; i++)
  {
    for (int j = 0; j < input_size; j++)
      f_test_images[i * input_size + j] -= mean_image[j];
  }

  // Simple CNN
  vector<LayerSpecifier> layer_specifier;
  {
    ConvDescriptor layer0;
    layer0.initializeValues(1, 3, 3, 3, 32, 32, 1, 1, 1, 1, RELU);
    LayerSpecifier temp;
    temp.initPointer(CONV);
    *((ConvDescriptor *)temp.params) = layer0;
    layer_specifier.push_back(temp);
  }
  {
    FCDescriptor layer1;
    layer1.initializeValues(3 * 32 * 32, 64, RELU);
    LayerSpecifier temp;
    temp.initPointer(FULLY_CONNECTED);
    *((FCDescriptor *)temp.params) = layer1;
    layer_specifier.push_back(temp);
  }
  {
    FCDescriptor layer2;
    layer2.initializeValues(64, 100);
    LayerSpecifier temp;
    temp.initPointer(FULLY_CONNECTED);
    *((FCDescriptor *)temp.params) = layer2;
    layer_specifier.push_back(temp);
  }
  {
    SoftmaxDescriptor layer2_smax;
    layer2_smax.initializeValues(SOFTMAX_ACCURATE, SOFTMAX_MODE_INSTANCE, 100, 1,
                                 1);
    LayerSpecifier temp;
    temp.initPointer(SOFTMAX);
    *((SoftmaxDescriptor *)temp.params) = layer2_smax;
    layer_specifier.push_back(temp);
  }

  int batch_size = 64;
  float softmax_eps = 1e-8;
  float init_std_dev = 0.01;
  NeuralNet net(layer_specifier, DATA_FLOAT, batch_size, TENSOR_NCHW,
                softmax_eps, init_std_dev, SGD);

  int num_epoch = 1000;
  double learning_rate = 1e-4;
  double learning_rate_decay = 1;

  Solver solver(&net, (void *)f_train_images, f_train_labels,
                (void *)f_train_images, f_train_labels, num_epoch, SGD,
                learning_rate, learning_rate_decay, num_train, num_train);
  vector<float> loss;
  vector<int> val_acc;
  solver.train(loss, val_acc);
  int num_correct;
  solver.checkAccuracy(f_train_images, f_train_labels, num_train, &num_correct);
  std::cout << "TRAIN NUM CORRECT:" << num_correct << endl;
  solver.checkAccuracy(f_test_images, f_test_labels, num_test, &num_correct);
  std::cout << "TEST NUM CORRECT:" << num_correct << endl;

  /** Store and load model from net object */
  net.save("model.txt");
  NeuralNet net2;
  net2.load("model.txt");
  Solver solver2(&net2, (void *)f_train_images, f_train_labels,
                 (void *)f_train_images, f_train_labels, num_epoch, SGD,
                 learning_rate, learning_rate_decay, num_train, num_train);
  solver.checkAccuracy(f_test_images, f_test_labels, num_test, &num_correct);
  std::cout << "TEST NUM CORRECT:" << num_correct << endl;

  //   solver.getTrainTime(loss, time, 100, fwd_vdnn_lag, bwd_vdnn_lag);
  //   printTimes(time, filename);
  //   printvDNNLag(fwd_vdnn_lag, bwd_vdnn_lag, filename);
}

/** NOT NEEDED AS OF NOW */
// void printTimes(vector<float> &time, string filename) {
//   float mean_time = 0.0;
//   float std_dev = 0.0;
//   int N = time.size();
//   for (int i = 0; i < N; i++) {
//     mean_time += time[i];
//   }
//   mean_time /= N;
//   for (int i = 0; i < N; i++) {
//     std_dev += pow(time[i] - mean_time, 2);
//   }
//   std_dev /= N;
//   pow(std_dev, 0.5);
//   cout << "Average time: " << mean_time << endl;
//   cout << "Standard deviation: " << std_dev << endl;

//   filename.append(".dat");
//   fstream f;
//   f.open(filename.c_str(), ios_base::out);

//   for (int i = 0; i < N; i++) {
//     f << time[i] << endl;
//   }
//   f << "mean_time: " << mean_time << endl;
//   f << "standard_deviation: " << std_dev << endl;
//   f.close();
// }

// void printvDNNLag(vector<vector<float> > &fwd_vdnn_lag, vector<vector<float>
// > &bwd_vdnn_lag,
//                   string filename) {
//   filename.append("_lag.dat");

//   fstream f;
//   f.open(filename.c_str(), ios_base::out);

//   int N = fwd_vdnn_lag.size();
//   for (int i = 0; i < N; i++) {
//     for (int j = 0; j < fwd_vdnn_lag[i].size(); j++) {
//       f << "fwd" << j << ": " << fwd_vdnn_lag[i][j] << endl;
//     }
//     for (int j = 0; j < bwd_vdnn_lag[i].size(); j++) {
//       f << "bwd" << j << ": " << bwd_vdnn_lag[i][j] << endl;
//     }
//     f << endl;
//   }
//   f.close();
// }
