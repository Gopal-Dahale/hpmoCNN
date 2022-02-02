#include <stdexcept>

#include "gpu_timer.cuh"
#include "solver.cuh"

Solver::Solver(NeuralNet *model, MNIST224 &mnist224, int num_epoch, UpdateRule update_rule,
               double learning_rate, double learning_rate_decay, int num_train, int num_val) {
  std::cout << "Solver::Solver" << '\n';
  if ((model->batch_size == 0) || (model->num_layers == 0))
    throw std::invalid_argument("Model is not initialized. Use parameterized constructor.");
  this->model = model;
  this->X_train = (void *)mnist224.f_train_images;
  this->X_val = (void *)mnist224.f_test_images;
  this->y_train = (int *)mnist224.f_test_images;
  this->y_val = (int *)mnist224.f_test_labels;
  this->num_epoch = num_epoch;
  this->update_rule = update_rule;
  this->learning_rate = learning_rate;
  this->learning_rate_decay = learning_rate_decay;

  this->num_train = num_train;
  this->num_val = num_val;
  this->num_features = model->input_channels * model->input_h * model->input_w;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  std::cout << "Solver::Solver done" << '\n';
}

float Solver::step(int start_X, int start_y, int *correct_count, bool train, float *overhead,
                   std::vector<std::pair<size_t, size_t>> &offload_mem) {
  std::vector<float> t1, t2;
  return this->step(start_X, start_y, t1, t2, correct_count, train, overhead, offload_mem);
}

float Solver::step(int start_X, int start_y, std::vector<float> &fwd_dnn_lag,
                   std::vector<float> &bwd_dnn_lag, int *correct_count, bool train, float *overhead,
                   std::vector<std::pair<size_t, size_t>> &offload_mem) {
  float temp_loss;
  if (model->data_type == CUDNN_DATA_FLOAT) {
    model->getLoss(&(((float *)X_train)[start_X]), &y_train[start_y], learning_rate, fwd_dnn_lag,
                   bwd_dnn_lag, offload_mem, train, correct_count, &temp_loss, overhead);
  } else if (model->data_type == CUDNN_DATA_DOUBLE) {
    model->getLoss(&(((double *)X_train)[start_X]), &y_train[start_y], learning_rate, fwd_dnn_lag,
                   bwd_dnn_lag, offload_mem, train, correct_count, &temp_loss, overhead);
  }
  cudaDeviceSynchronize();
  return temp_loss;
}

void Solver::train(Metrics &metrics) {
  std::cout << "Solver::train" << '\n';
  GpuTimer timer;
  int batch_size = model->batch_size;
  int num_train_batches = num_train / model->batch_size;
  int num_val_batches = num_val / model->batch_size;

  for (int i = 0; i < num_epoch; i++) {
    std::cout << "Epoch " << i << '\n';
    for (int j = 0; j < num_train_batches; j++) {
      std::cout << "Batch " << j << '\n';
      int start_sample = j * num_features * batch_size;

      timer.start(model->stream_compute);
      float temp_loss =
          step(start_sample, j * batch_size, NULL, true, metrics.overhead, metrics.offload_mem);
      timer.stop(model->stream_compute);
      metrics.loss.emplace_back(temp_loss);
      metrics.batch_times.emplace_back(timer.elapsed());
    }

    int correct_count = 0;
    for (int j = 0; j < num_val_batches; j++) {
      int start_sample = j * num_features * batch_size;
      int temp_correct_count = 0;
      float temp_loss = step(start_sample, j * batch_size, &temp_correct_count, false,
                             metrics.overhead, metrics.offload_mem);
      correct_count += temp_correct_count;
    }
    metrics.val_acc.emplace_back(correct_count);
    learning_rate *= learning_rate_decay;
  }
  std::cout << "Solver::train done" << '\n';
}

void Solver::checkAccuracy(void *X, int *y, int num_samples, int *num_correct) {
  int batch_size = model->batch_size;
  int num_iter = num_samples / batch_size;
  *num_correct = 0;
  for (int i = 0; i < num_iter; i++) {
    int start_sample = i * num_features * batch_size;
    int temp_correct_count;
    std::vector<std::pair<size_t, size_t>> offload_mem;
    if (model->data_type == CUDNN_DATA_FLOAT)
      model->getLoss(&(((float *)X)[start_sample]), &y[i * batch_size], learning_rate, offload_mem,
                     false, &temp_correct_count, NULL, NULL);
    else if (model->data_type == CUDNN_DATA_DOUBLE)
      model->getLoss(&(((double *)X)[start_sample]), &y[i * batch_size], learning_rate, offload_mem,
                     false, &temp_correct_count, NULL, NULL);
    *num_correct = *num_correct + temp_correct_count;
  }
}
