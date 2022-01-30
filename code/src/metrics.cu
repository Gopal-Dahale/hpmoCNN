#include "metrics.cuh"

void Metrics::save_layer_mem_usage() {
  std::ofstream mem_usage;
  std::string res = neural_net + "/batch_size" + std::to_string(configs["batch_size"]);
  mem_usage.open("../" + res + "/mem_usage.txt");

  for (int c = 0; c < net->num_layers + 1; c++) {
    size_t feature_map_size, fwd_workspace_size = 0, bwd_workspace_filter = 0,
                             bwd_workspace_data = 0, weights = 0;
    feature_map_size = net->layer_input_size[c] * net->data_type_size;
    if (c != net->num_layers && net->layer_type[c] == CONV) {
      ConvLayerParams *cur_params = (ConvLayerParams *)net->params[c];
      fwd_workspace_size = cur_params->fwd_workspace_size;
      bwd_workspace_filter = cur_params->bwd_filter_workspace_size;
      bwd_workspace_data = cur_params->bwd_data_workspace_size;
      weights = cur_params->kernel_size * net->data_type_size;
    } else if (c != net->num_layers && net->layer_type[c] == FULLY_CONNECTED) {
      FCLayerParams *cur_params = (FCLayerParams *)net->params[c];
      int wt_alloc_size = cur_params->weight_matrix_size;
      if (wt_alloc_size % 2 != 0) wt_alloc_size += 1;
      weights = (wt_alloc_size + cur_params->C_out) * net->data_type_size;
    }
    mem_usage << feature_map_size << " " << fwd_workspace_size << " " << bwd_workspace_filter << " "
              << bwd_workspace_data << " " << weights << "\n";
  }
  mem_usage.close();
}

void Metrics::save_metrics() {
  std::fstream f;

  std::string res = neural_net + "/batch_size" + std::to_string(configs["batch_size"]);
  f.open("../" + res + "/batch_times.txt", std::ios::out);
  for (auto &i : batch_times) f << i << '\n';
  f.close();

  f.open("../" + res + "/loss.txt", std::ios::out);
  for (auto &i : loss) f << i << '\n';
  f.close();

  f.open("../" + res + "/val_acc.txt", std::ios::out);
  for (auto &i : val_acc) f << i << '\n';
  f.close();

  f.open("../" + res + "/configs.txt", std::ios::out);
  for (auto &i : configs) f << i.first << " " << i.second << '\n';
  f.close();

  f.open("../" + res + "/totaltime.txt", std::ios::out);
  f << train_time << '\n';
  f.close();

  f.open("../" + res + "/totaloverhead.txt", std::ios::out);
  f << overhead << '\n';
  f.close();
}

void Metrics::save_offload_mem() {
  std::fstream f;
  std::string res = neural_net + "/batch_size" + std::to_string(configs["batch_size"]);
  f.open("../" + res + "/offload_mem.txt", std::ios::out);
  for (auto &i : offload_mem) f << i.first << " " << i.second << '\n';
  f.close();
}