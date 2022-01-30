#include "parser.cuh"

cxxopts::Options Parser::init() {
  cxxopts::Options options("parser", "A simple parser");
  options.add_options()(
      "batch-size", "Batch Size",
      cxxopts::value<int>()->default_value("64")) // a bool parameter
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
          cxxopts::value<int>()->default_value("512"))(
          "net", "network",
          cxxopts::value<std::string>()->default_value("vgg16"))("help", "Print Usage");
  return options;
}
