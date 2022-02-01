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

#include "hpmo_cnn.cuh"

using namespace std;

typedef unsigned char uchar;

int num_train = 1024, num_test = 512;

auto get_layer_specifier(string neural_net) {
  VGG16 vgg16;
  auto layer_specifier = vgg16.layer_specifier;
  if (neural_net == "vgg19") {
    VGG19 vgg19;
    layer_specifier = vgg19.layer_specifier;
    std::cout << "Network: VGG19" << std::endl;
  } else if (neural_net == "vgg116") {
    VGG116 vgg116;
    layer_specifier = vgg116.layer_specifier;
    std::cout << "Network: VGG116" << std::endl;
  } else {
    std::cout << "Network: VGG16" << std::endl;
  }
  return layer_specifier;
}

int main(int argc, char *argv[]) {
  /******************* Parse command line arguments ********************/
  Parser parser;
  auto options = parser.init();
  auto result = options.parse(argc, argv);  // Get the parsed results

  // Display the help message if the user requested it
  if (result.count("help")) {
    std::cout << options.help() << std::endl;
    exit(0);
  }

  /******************* Read Dataset ************************************/
  num_train = result["num-train"].as<int>();
  num_test = result["num-test"].as<int>();

  cout << "Reading MNIST dataset...  ";

  string path_train_images = "/kaggle/input/mnist224by224testdataset/train-images-224by224-";
  string path_train_labels = "data/train-labels.idx1-ubyte";
  string path_test_images = "/kaggle/input/mnist224by224testdataset/test-images-224by224-";
  string path_test_labels = "data/t10k-labels.idx1-ubyte";

  MNIST224 mnist224(num_train, num_test, path_train_images, path_train_labels, path_test_images,
                    path_test_labels);
  mnist224.read_mnist_224();
  mnist224.normalize();
  cout << "Done" << endl;

  auto neural_net = result["net"].as<std::string>();
  auto layer_specifier = get_layer_specifier(neural_net);

  /**************************** Configuration ****************************/
  int batch_size = result["batch-size"].as<int>();
  float softmax_eps = result["softmax-eps"].as<float>();
  float init_std_dev = result["init-std-dev"].as<float>();
  int num_epoch = result["epochs"].as<int>();
  double learning_rate = result["learning-rate"].as<double>();
  double learning_rate_decay = result["learning-rate-decay"].as<double>();

  /************************ Display configuration *************************/
  unordered_map<string, double> configs = {
      {"batch_size", batch_size},       {"softmax_eps", softmax_eps},
      {"init_std_dev", init_std_dev},   {"num_epoch", num_epoch},
      {"learning_rate", learning_rate}, {"learning_rate_decay", learning_rate_decay},
      {"num_train", num_train},         {"num_test", num_test}};

  for (auto &config : configs) {
    cout << config.first << ": " << config.second << endl;
  }

  /*************************** Train & Test ***************************/
  NeuralNet net(layer_specifier, DATA_FLOAT, batch_size, TENSOR_NCHW, softmax_eps, init_std_dev,
                SGD);
  Solver solver(&net, mnist224, num_epoch, SGD, learning_rate, learning_rate_decay, num_train,
                num_train);
  Metrics metrics(&net, neural_net, configs);
  GpuTimer timer;

  timer.start();
  solver.train(metrics);
  timer.stop();
  metrics.train_time = timer.elapsed();

  solver.checkAccuracy(mnist224.f_train_images, mnist224.f_train_labels, num_train,
                       &metrics.num_correct);
  solver.checkAccuracy(mnist224.f_test_images, mnist224.f_test_labels, num_test,
                       &metrics.num_correct);

  std::cout << "TRAIN NUM CORRECT:" << metrics.num_correct << endl;
  std::cout << "TEST NUM CORRECT:" << metrics.num_correct << endl;

  /*************************** Save metrics ***************************/
  metrics.save_layer_mem_usage();
  metrics.save_metrics();
  metrics.save_offload_mem();
}
