#ifndef __METRICS__
#define __METRICS__
#include <unordered_map>
#include <utility>
#include <vector>

#include "neural_net.cuh"

class Metrics {
  NeuralNet *net;
  std::string neural_net;
  std::unordered_map<std::string, double> configs;

 public:
  int num_correct;
  float overhead;
  float train_time;
  std::vector<float> loss;
  std::vector<int> val_acc;
  std::vector<float> batch_times;
  std::vector<std::pair<size_t, size_t>> offload_mem;

  Metrics(NeuralNet *net, std::string &neural_net, std::unordered_map<std::string, double> &configs)
      : net(net), neural_net(neural_net), configs(configs) {}

  void save_layer_mem_usage();
  void save_metrics();
  void save_offload_mem();
};

#endif