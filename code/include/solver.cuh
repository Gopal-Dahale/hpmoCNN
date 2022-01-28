#ifndef SOLVER
#define SOLVER

#include "neural_net.cuh"

class Solver
{
public:
       NeuralNet *model;
       void *X_train, *X_val;
       int *y_train, *y_val;
       int num_epoch;
       UpdateRule update_rule;
       double learning_rate, learning_rate_decay;
       int num_train, num_val;
       int num_train_batches;
       int num_features;
       cudaEvent_t start, stop;

       Solver(NeuralNet *model, void *X_train, int *y_train, void *X_val, int *y_val,
              int num_epoch, UpdateRule update_rule, double learning_rate,
              double learning_rate_decay, int num_train, int num_val);
       void train(std::vector<float> &loss, std::vector<int> &val_acc,
                  std::vector<float> &batch_times, float *overhead, std::vector<std::pair<size_t, size_t>> &offload_mem);
       float step(int start_X, int start_y, std::vector<float> &fwd_vdnn_lag,
                  std::vector<float> &bwd_vdnn_lag, int *correct_count, bool train,
                  float *overhead, std::vector<std::pair<size_t, size_t>> &offload_mem);
       float step(int start_X, int start_y, int *correct_count, bool train,
                  float *overhead, std::vector<std::pair<size_t, size_t>> &offload_mem);
       void checkAccuracy(void *X, int *y, int num_samples, int *num_correct);
};

#endif
