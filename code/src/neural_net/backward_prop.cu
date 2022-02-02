#include "gpu_timer.cuh"
#include "neural_net.cuh"
#include "utils.cuh"

template <typename T>
__global__ void softmaxLossBackProp(int *y, T *SO, T *dSO, int batch_size, int output_size,
                                    float eps) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= batch_size) return;
  int cur_class = static_cast<int>(y[i]);
  dSO[i * output_size + cur_class] = -1 / (SO[i * output_size + cur_class] * batch_size + eps);
}

void NeuralNet::softmax_loss_backward() {
  if (layer_type[num_layers - 1] == SOFTMAX) {
    if (data_type == CUDNN_DATA_FLOAT) {
      cudaMemset(dlayer_input[num_layers], 0, batch_size * num_classes * sizeof(float));
      softmaxLossBackProp<float><<<ceil(1.0 * batch_size / BW), BW>>>(
          this->y, (float *)layer_input[num_layers], (float *)dlayer_input[num_layers], batch_size,
          num_classes, softmax_eps);
    } else if (data_type == CUDNN_DATA_DOUBLE) {
      cudaMemset(dlayer_input[num_layers], 0, batch_size * num_classes * sizeof(double));
      softmaxLossBackProp<double><<<ceil(1.0 * batch_size / BW), BW>>>(
          this->y, (double *)layer_input[num_layers], (double *)dlayer_input[num_layers],
          batch_size, num_classes, softmax_eps);
    }
  }
}

void NeuralNet::prefetch_policy(int &i, void *X) {
  if (offloaded[i - 1]) {
    std::cout << "Prefetching layer " << i - 1 << " Size " << std::setprecision(2)
              << layer_input_size[i - 1] * data_type_size / (1024.0 * 1024.0) << "MB"
              << " from CPU" << '\n';
    cudaMalloc(&layer_input[i - 1], layer_input_size[i - 1] * data_type_size);
    if (i - 1 != 0) {
      cudaMemcpyAsync(layer_input[i - 1], h_layer_input[i - 1],
                      layer_input_size[i - 1] * data_type_size, cudaMemcpyHostToDevice,
                      stream_memory);
    } else {
      cudaMemcpyAsync(layer_input[i - 1], X, layer_input_size[i - 1] * data_type_size,
                      cudaMemcpyHostToDevice, stream_memory);
    }
  }
}

void NeuralNet::backward_prop(void *X, float &alpha, float &beta, float &Salpha, float &Sbeta,
                              double &Dalpha, double &Dbeta, std::vector<float> &bwd_dnn_lag,
                              float *overhead, double &learning_rate) {
  std::cout << "Backward Propagation Starts" << '\n';
  cudaMalloc(&dlayer_input[num_layers], batch_size * num_classes * data_type_size);

  for (int i = num_layers - 1; i >= 0; i--) {
    if (i > 0) {
      if (layer_type[i] == ACTV || layer_type[i] == SOFTMAX) {
        dlayer_input[i] = dlayer_input[i + 1];
      }
      prefetch_policy(i, X);

      cudaMalloc(&dlayer_input[i], layer_input_size[i] * data_type_size);
    }

    switch (layer_type[i]) {
      case CONV:
        conv_backward(i, alpha, beta, learning_rate);
        break;
      case FULLY_CONNECTED:
        fc_backward(i, alpha, beta, Salpha, Sbeta, Dalpha, Dbeta, learning_rate);
        break;
      case POOLING:
        pooling_backward(i, alpha, beta);
        break;
      case ACTV:
        activation_backward(i, alpha, beta);
        continue;
        break;
      case SOFTMAX:
        softmax_backward(i, alpha, beta);
        continue;
        break;
      default:
        break;
    }

    cudaStreamSynchronize(stream_compute);

    GpuTimer timer;
    float milli = 0;
    timer.start(stream_memory);
    cudaStreamSynchronize(stream_memory);
    timer.stop(stream_memory);
    milli = timer.elapsed();

    bwd_dnn_lag.push_back(milli);
    if (overhead != NULL) {
      *overhead += milli;
    }

    if (layer_type[i] == CONV) {
      cudaFree(this->workspace);
    }

    cudaFree(layer_input[i + 1]);
    cudaFree(dlayer_input[i + 1]);
    std::cout << "Freed layer " << i + 1 << " Size " << std::setprecision(2)
              << layer_input_size[i + 1] * data_type_size / (1024.0 * 1024.0) << "MB" << '\n';

    if (i == 0) {
      cudaFree(layer_input[i]);
      cudaFree(dlayer_input[i]);
      std::cout << "Freed layer " << i << " Size " << std::setprecision(2)
                << layer_input_size[i] * data_type_size / (1024.0 * 1024.0) << "MB" << '\n';
    }
  }
  std::cout << "Backward Propagation Ends" << '\n';
}

void NeuralNet::conv_backward(int &i, float &alpha, float &beta, double &learning_rate) {
  ConvLayerParams *cur_params = (ConvLayerParams *)params[i];
  if (cur_params->activation_mode != ACTIVATION_NONE) {
    checkCUDNN(cudnnActivationBackward(
        cudnn_handle, cur_params->actv_desc, &alpha, cur_params->output_tensor, layer_input[i + 1],
        cur_params->output_tensor, dlayer_input[i + 1], cur_params->output_tensor,
        layer_input[i + 1], &beta, cur_params->output_tensor, dlayer_input[i + 1]));
  }

  size_t temp_data_wksp;

  if (i == 0)
    temp_data_wksp = 0;
  else
    temp_data_wksp = cur_params->bwd_data_workspace_size;

  this->workspace_size = max(cur_params->bwd_filter_workspace_size, temp_data_wksp);

  cudaMalloc(&(this->workspace), this->workspace_size);
  std::cout << "Allocated workspace of layer " << i << " Size: " << std::setprecision(2)
            << this->workspace_size / (1024.0 * 1024.0) << "MB" << '\n';

  checkCUDNN(cudnnConvolutionBackwardBias(cudnn_handle, &alpha, cur_params->output_tensor,
                                          dlayer_input[i + 1], &beta, cur_params->bias_desc,
                                          cur_params->db));

  checkCUDNN(cudnnConvolutionBackwardFilter(
      cudnn_handle, &alpha, cur_params->input_tensor, layer_input[i], cur_params->output_tensor,
      dlayer_input[i + 1], cur_params->conv_desc, cur_params->bwd_filter_algo, this->workspace,
      this->workspace_size, &beta, cur_params->filter_desc, cur_params->dW));

  if (i > 0)
    checkCUDNN(cudnnConvolutionBackwardData(
        cudnn_handle, &alpha, cur_params->filter_desc, cur_params->W, cur_params->output_tensor,
        dlayer_input[i + 1], cur_params->conv_desc, cur_params->bwd_data_algo, this->workspace,
        workspace_size, &beta, cur_params->input_tensor, dlayer_input[i]));

  cur_params->stepParams(cublas_handle, learning_rate);
}

void NeuralNet::fc_backward(int &i, float &alpha, float &beta, float &Salpha, float &Sbeta,
                            double &Dalpha, double &Dbeta, double &learning_rate) {
  FCLayerParams *cur_params = (FCLayerParams *)params[i];

  if (cur_params->activation_mode != ACTIVATION_NONE) {
    checkCUDNN(cudnnActivationBackward(
        cudnn_handle, cur_params->actv_desc, &alpha, cur_params->output_tensor, layer_input[i + 1],
        cur_params->output_tensor, dlayer_input[i + 1], cur_params->output_tensor,
        layer_input[i + 1], &beta, cur_params->output_tensor, dlayer_input[i + 1]));
  }

  if (data_type == CUDNN_DATA_FLOAT) {
    // Bias backward
    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, cur_params->C_out, 1, batch_size, &Salpha,
                (float *)dlayer_input[i + 1], cur_params->C_out, (float *)one_vec, batch_size,
                &Sbeta, (float *)cur_params->db, cur_params->C_out);

    // Weight backward
    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, cur_params->C_out, cur_params->C_in,
                batch_size, &Salpha, (float *)dlayer_input[i + 1], cur_params->C_out,
                (float *)layer_input[i], cur_params->C_in, &Sbeta, (float *)cur_params->dW,
                cur_params->C_out);

    // Data backward
    if (i > 0)
      cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, cur_params->C_in, batch_size,
                  cur_params->C_out, &Salpha, (float *)cur_params->W, cur_params->C_out,
                  (float *)dlayer_input[i + 1], cur_params->C_out, &Sbeta, (float *)dlayer_input[i],
                  cur_params->C_in);
  }

  else if (data_type == CUDNN_DATA_DOUBLE) {
    // Bias backward
    cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, cur_params->C_out, 1, batch_size, &Dalpha,
                (double *)dlayer_input[i + 1], cur_params->C_out, (double *)one_vec, batch_size,
                &Dbeta, (double *)cur_params->db, cur_params->C_out);

    // Weight backward
    cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, cur_params->C_out, cur_params->C_in,
                batch_size, &Dalpha, (double *)dlayer_input[i + 1], cur_params->C_out,
                (double *)layer_input[i], cur_params->C_in, &Dbeta, (double *)cur_params->dW,
                cur_params->C_out);

    // Data backward
    if (i > 0)
      cublasDgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, cur_params->C_in, batch_size,
                  cur_params->C_out, &Dalpha, (double *)cur_params->W, cur_params->C_out,
                  (double *)dlayer_input[i + 1], cur_params->C_out, &Dbeta,
                  (double *)dlayer_input[i], cur_params->C_in);
  }
  cur_params->stepParams(cublas_handle, learning_rate);
}

void NeuralNet::pooling_backward(int &i, float &alpha, float &beta) {
  PoolingLayerParams *cur_params = (PoolingLayerParams *)params[i];
  checkCUDNN(cudnnPoolingBackward(
      cudnn_handle, cur_params->pool_desc, &alpha, cur_params->output_tensor, layer_input[i + 1],
      cur_params->output_tensor, dlayer_input[i + 1], cur_params->input_tensor, layer_input[i],
      &beta, cur_params->input_tensor, dlayer_input[i]));
}

void NeuralNet::activation_backward(int &i, float &alpha, float &beta) {
  ActivationLayerParams *cur_params = (ActivationLayerParams *)params[i];
  checkCUDNN(cudnnActivationBackward(
      cudnn_handle, cur_params->actv_desc, &alpha, cur_params->input_tensor, layer_input[i + 1],
      cur_params->input_tensor, dlayer_input[i + 1], cur_params->input_tensor, layer_input[i],
      &beta, cur_params->input_tensor, dlayer_input[i]));
}

void NeuralNet::softmax_backward(int &i, float &alpha, float &beta) {
  SoftmaxLayerParams *cur_params = (SoftmaxLayerParams *)params[i];
  checkCUDNN(cudnnSoftmaxBackward(cudnn_handle, cur_params->algo, cur_params->mode, &alpha,
                                  cur_params->input_tensor, layer_input[i + 1],
                                  cur_params->input_tensor, dlayer_input[i + 1], &beta,
                                  cur_params->input_tensor, dlayer_input[i]));
}