#include <cublas_v2.h>
#include <cudnn.h>
#include <curand.h>
#include <time.h>

#include <cstdio>
#include <string>

#include "neural_net.h"

template <typename T>
__global__ void softmaxLossBackProp(int *y, T *SO, T *dSO, int batch_size, int output_size,
                                    float eps) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= batch_size) return;
  int cur_class = static_cast<int>(y[i]);
  dSO[i * output_size + cur_class] = -1 / (SO[i * output_size + cur_class] * batch_size + eps);
}

template <typename T>
__global__ void computeSoftmaxLoss(T *O, int *y, float *loss, int batch_size, int num_classes,
                                   float eps) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= batch_size) return;

  loss[i] = -logf(O[i * num_classes + y[i]] + eps);
}

template <typename T>
__global__ void inferClass(T *O, int *pred_y, int batch_size, int num_classes) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= batch_size) return;

  T max = O[i * num_classes];
  int index = 0;
  for (int j = 1; j < num_classes; j++) {
    if (O[i * num_classes + j] > max) {
      max = O[i * num_classes + j];
      index = j;
    }
  }
  pred_y[i] = index;
}

float NeuralNet::computeLoss() {
  if (layer_type[num_layers - 1] == SOFTMAX) {
    if (data_type == CUDNN_DATA_FLOAT)
      computeSoftmaxLoss<float><<<ceil(1.0 * batch_size / BW), BW>>>(
          (float *)layer_input[num_layers], this->y, loss, batch_size, num_classes, softmax_eps);
    else if (data_type == CUDNN_DATA_DOUBLE)
      computeSoftmaxLoss<double><<<ceil(1.0 * batch_size / BW), BW>>>(
          (double *)layer_input[num_layers], this->y, loss, batch_size, num_classes, softmax_eps);
  }
  cudaMemcpy(h_loss, loss, batch_size * sizeof(float), cudaMemcpyDeviceToHost);
  float total_loss = 0.0;
  for (int i = 0; i < batch_size; i++) total_loss += h_loss[i];
  return total_loss / batch_size;
}

void NeuralNet::compareOutputCorrect(int *correct_count, int *y) {
  *correct_count = 0;

  if (data_type == CUDNN_DATA_FLOAT) {
    float *typecast_O = (float *)layer_input[num_layers - 1];
    inferClass<float>
        <<<ceil(1.0 * batch_size / BW), BW>>>(typecast_O, pred_y, batch_size, num_classes);
    cudaMemcpy(h_pred_y, pred_y, batch_size * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < batch_size; i++) {
      if (h_pred_y[i] == y[i]) *correct_count = *correct_count + 1;
    }
  } else if (data_type == CUDNN_DATA_DOUBLE) {
    double *typecast_O = (double *)layer_input[num_layers - 1];
    inferClass<double>
        <<<ceil(1.0 * batch_size / BW), BW>>>(typecast_O, pred_y, batch_size, num_classes);
    cudaMemcpy(h_pred_y, pred_y, batch_size * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < batch_size; i++) {
      if (h_pred_y[i] == y[i]) *correct_count = *correct_count + 1;
    }
  }
}

NeuralNet::NeuralNet(std::vector<LayerSpecifier> &layers, DataType data_type, int batch_size,
                     TensorFormat tensor_format, long long dropout_seed, float softmax_eps,
                     float init_std_dev, UpdateRule update_rule) {
  cudaStreamCreate(&stream_compute);

  // create handle
  checkCUDNN(cudnnCreate(&cudnn_handle));
  checkCUDNN(cudnnSetStream(cudnn_handle, stream_compute));

  cublasCreate(&cublas_handle);
  cublasSetStream(cublas_handle, stream_compute);

  curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetStream(curand_gen, stream_compute);

  cudaMemGetInfo(&free_bytes, &total_bytes);
  init_free_bytes = free_bytes;
  std::cout << "Free bytes at start: " << free_bytes << std::endl;

  if (data_type == DATA_FLOAT) {
    this->data_type = CUDNN_DATA_FLOAT;
    data_type_size = sizeof(float);
  }

  else if (data_type == DATA_DOUBLE) {
    this->data_type = CUDNN_DATA_DOUBLE;
    data_type_size = sizeof(double);
  }

  if (tensor_format == TENSOR_NCHW)
    this->tensor_format = CUDNN_TENSOR_NCHW;
  else if (tensor_format == TENSOR_NHWC)
    this->tensor_format = CUDNN_TENSOR_NHWC;

  this->batch_size = batch_size;
  this->softmax_eps = softmax_eps;
  this->init_std_dev = init_std_dev;

  num_layers = layers.size();
  // allocation of space for input to each layer
  layer_input = (void **)malloc((num_layers + 1) * sizeof(void *));
  layer_input_size = (int *)malloc((num_layers + 1) * sizeof(int));
  dlayer_input = (void **)malloc((num_layers + 1) * sizeof(void *));
  params = (void **)malloc(num_layers * sizeof(void *));

  LayerDimension prev_output_size;
  LayerDimension current_output_size;
  for (int i = 0; i < num_layers; i++) {
    layer_type.push_back(layers[i].type);
    if (layers[i].type == CONV) {
      ConvDescriptor *user_params = (ConvDescriptor *)layers[i].params;
      params[i] = malloc(sizeof(ConvLayerParams));
      ((ConvLayerParams *)params[i])
          ->initializeValues(cudnn_handle, user_params, this->data_type, batch_size,
                             this->tensor_format, data_type_size, current_output_size, update_rule);
    } else if (layers[i].type == FULLY_CONNECTED) {
      FCDescriptor *user_params = (FCDescriptor *)layers[i].params;
      params[i] = malloc(sizeof(FCLayerParams));
      ((FCLayerParams *)params[i])
          ->initializeValues(user_params, batch_size, this->tensor_format, this->data_type,
                             current_output_size, update_rule);
    } else if (layers[i].type == POOLING) {
      PoolingDescriptor *user_params = (PoolingDescriptor *)layers[i].params;
      params[i] = malloc(sizeof(PoolingLayerParams));
      ((PoolingLayerParams *)params[i])
          ->initializeValues(user_params, this->data_type, this->tensor_format, batch_size,
                             current_output_size);
    } else if (layers[i].type == ACTV) {
      ActivationDescriptor *user_params = (ActivationDescriptor *)layers[i].params;
      params[i] = malloc(sizeof(ActivationLayerParams));
      ((ActivationLayerParams *)params[i])
          ->initializeValues(user_params, this->data_type, this->tensor_format, batch_size,
                             current_output_size);
    }

    else if (layers[i].type == SOFTMAX) {
      SoftmaxDescriptor *user_params = (SoftmaxDescriptor *)layers[i].params;
      params[i] = malloc(sizeof(SoftmaxLayerParams));
      ((SoftmaxLayerParams *)params[i])
          ->initializeValues(user_params, this->data_type, this->tensor_format, batch_size,
                             current_output_size);
      // std::cout << current_output_size.N << ' ' << current_output_size.C << current_output_size.H
      // << current_output_size.W << std::endl;
    }
    if (i == 0) {
      prev_output_size = current_output_size;
    }
    // incomplete - have to check flatten and check exact dimension
    // else if (current_output_size.getTotalSize() != prev_output_size.getTotalSize()) {
    // 	std::cout << "Layer " << i << " output and next layer's input size mismatch\n";
    // 	exit(0);
    // }
  }

  cudaMemGetInfo(&free_bytes, &total_bytes);
  std::cout << "Free bytes just before allocate space: " << free_bytes << std::endl;

  for (int i = 0; i < num_layers; i++) {
    size_t input_size;
    if (layers[i].type == CONV) {
      ConvDescriptor *user_params = (ConvDescriptor *)layers[i].params;
      ((ConvLayerParams *)params[i])
          ->allocateSpace(curand_gen, this->data_type, data_type_size, init_std_dev, free_bytes);

      input_size =
          batch_size * user_params->input_channels * user_params->input_h * user_params->input_w;
      if (i == 0) {
        input_channels = user_params->input_channels;
        input_h = user_params->input_h;
        input_w = user_params->input_w;
      }
    } else if (layers[i].type == FULLY_CONNECTED) {
      FCDescriptor *user_params = (FCDescriptor *)layers[i].params;
      ((FCLayerParams *)params[i])
          ->allocateSpace(curand_gen, this->data_type, data_type_size, init_std_dev, free_bytes);
      input_size = batch_size * user_params->input_channels;
      if (i == 0) {
        input_channels = user_params->input_channels;
        input_h = 1;
        input_w = 1;
      }
    } else if (layers[i].type == POOLING) {
      PoolingDescriptor *user_params = (PoolingDescriptor *)layers[i].params;
      ((PoolingLayerParams *)params[i])->allocateSpace(free_bytes);
      input_size =
          batch_size * user_params->input_channels * user_params->input_h * user_params->input_w;
      if (i == 0) {
        input_channels = user_params->input_channels;
        input_h = user_params->input_h;
        input_w = user_params->input_w;
      }
    } else if (layers[i].type == ACTV) {
      ActivationDescriptor *user_params = (ActivationDescriptor *)layers[i].params;
      ((ActivationLayerParams *)params[i])->allocateSpace(free_bytes);
      input_size = batch_size * user_params->channels * user_params->h * user_params->w;
      if (i == 0) {
        input_channels = user_params->channels;
        input_h = user_params->h;
        input_w = user_params->w;
      }
    } else if (layers[i].type == SOFTMAX) {
      SoftmaxDescriptor *user_params = (SoftmaxDescriptor *)layers[i].params;
      ((SoftmaxLayerParams *)params[i])->allocateSpace(free_bytes);
      input_size = batch_size * user_params->channels * user_params->h * user_params->w;

      // assuming this is last layer, allocate for next layer as well
      cudaMalloc(&layer_input[i + 1], input_size * data_type_size);
      cudaMalloc(&dlayer_input[i + 1], input_size * data_type_size);
      layer_input_size[i + 1] = input_size;
      if (i == 0) {
        input_channels = user_params->channels;
        input_h = user_params->h;
        input_w = user_params->w;
      }
      if (i == num_layers - 1) {
        num_classes = user_params->channels;
      }
    }

    // do not allocate memory initially
    cudaMalloc(&layer_input[i], input_size * data_type_size);
    cudaMalloc(&dlayer_input[i], input_size * data_type_size);
  }
  cudaDeviceSynchronize();
  cudaMemGetInfo(&free_bytes, &total_bytes);
  std::cout << "Free bytes just after allocate space: " << free_bytes << std::endl;
  // very small - could be allocated initially itself
  cudaMalloc((void **)&y, batch_size * sizeof(int));
  cudaMalloc((void **)&pred_y, batch_size * sizeof(int));
  cudaMalloc((void **)&loss, batch_size * sizeof(float));
  cudaMalloc(&one_vec, batch_size * data_type_size);

  if (this->data_type == CUDNN_DATA_FLOAT)
    fillValue<float><<<ceil(1.0 * batch_size / BW), BW>>>((float *)one_vec, batch_size, 1);
  else
    fillValue<double><<<ceil(1.0 * batch_size / BW), BW>>>((double *)one_vec, batch_size, 1);

  cudaMallocHost((void **)&h_loss, batch_size * sizeof(float));
  cudaMallocHost((void **)&h_pred_y, batch_size * sizeof(int));

  // do not allocate workspace initially
  // allocate space for workspace and also keep track of algo
  size_t cur_workspace_size_1, cur_workspace_size_2, cur_workspace_size_3, cur_workspace_size;
  workspace_size = 0;
  for (int i = 0; i < num_layers; i++) {
    if (layers[i].type == CONV) {
      cur_workspace_size_1 =
          ((ConvLayerParams *)params[i])->getWorkspaceSize(free_bytes, ConvLayerParams::FWD);
      cur_workspace_size_2 =
          ((ConvLayerParams *)params[i])->getWorkspaceSize(free_bytes, ConvLayerParams::BWD_DATA);
      cur_workspace_size_3 =
          ((ConvLayerParams *)params[i])->getWorkspaceSize(free_bytes, ConvLayerParams::BWD_FILTER);
      cur_workspace_size =
          max(cur_workspace_size_1, max(cur_workspace_size_2, cur_workspace_size_3));
      if (cur_workspace_size > workspace_size) workspace_size = cur_workspace_size;
    }
  }

  cudaMalloc(&workspace, workspace_size);
  free_bytes = free_bytes - workspace_size;
  cudaDeviceSynchronize();
  cudaMemGetInfo(&free_bytes, &total_bytes);

  // leave 600 MB and use the rest
  std::cout << "Free bytes: " << free_bytes << std::endl;
  free_bytes -= 1024 * 1024 * 600;

  cudaDeviceSynchronize();
  size_t temp_free_bytes;
  cudaMemGetInfo(&temp_free_bytes, &total_bytes);
  std::cout << "Free bytes just before end of NeuralNet: " << temp_free_bytes << std::endl;
  // {
  // 	int n;
  // 	std::cout << "waiting..\n";
  // 	std::cin >> n;
  // }

  // data of time
  cudaEventCreate(&start_compute);
  cudaEventCreate(&stop_compute);

  cudaEventCreate(&start_transfer);
  cudaEventCreate(&stop_transfer);
}

void NeuralNet::getLoss(void *X, int *y, double learning_rate, bool train, int *correct_count,
                        float *loss) {
  std::vector<float> t1, t2;
  this->getLoss(X, y, learning_rate, t1, t2, train, correct_count, loss);
}

void NeuralNet::getLoss(void *X, int *y, double learning_rate, std::vector<float> &fwd_vdnn_lag,
                        std::vector<float> &bwd_vdnn_lag, bool train, int *correct_count,
                        float *scalar_loss) {
  // std::cout << "here\n";
  // std::cout << "Free bytes: " << free_bytes << std::endl;
  cudaMemcpy(layer_input[0], X, batch_size * input_channels * input_h * input_w * data_type_size,
             cudaMemcpyHostToDevice);
  if (train == true) {
    cudaMemcpy(this->y, y, batch_size * data_type_size, cudaMemcpyHostToDevice);
  }
  float alpha = 1.0, beta = 0.0;
  float Salpha = 1.0, Sbeta = 0.0;
  double Dalpha = 1.0, Dbeta = 0.0;

  // forward propagate
  for (int i = 0; i < num_layers; i++) {
    if (train == false && i == num_layers - 1) break;
    // std::cout << "here" << i << std::endl;
    if (layer_type[i] == CONV) {
      // std::cout << "conv\n";
      ConvLayerParams *cur_params = (ConvLayerParams *)params[i];

      // computation
      checkCUDNN(cudnnConvolutionForward(
          cudnn_handle, &alpha, cur_params->input_tensor, layer_input[i], cur_params->filter_desc,
          cur_params->W, cur_params->conv_desc, cur_params->fwd_algo, workspace, workspace_size,
          &beta, cur_params->output_tensor, layer_input[i + 1]));
      checkCUDNN(cudnnAddTensor(cudnn_handle, &alpha, cur_params->bias_desc, cur_params->b, &alpha,
                                cur_params->output_tensor, layer_input[i + 1]));

      // if activation required
      if (cur_params->activation_mode != ACTIVATION_NONE) {
        checkCUDNN(cudnnActivationForward(cudnn_handle, cur_params->actv_desc, &alpha,
                                          cur_params->output_tensor, layer_input[i + 1], &beta,
                                          cur_params->output_tensor, layer_input[i + 1]));
      }

      // std::cout << "Free bytes: " << free_bytes << std::endl;

    }

    else if (layer_type[i] == FULLY_CONNECTED) {
      // std::cout << "FC\n";
      FCLayerParams *cur_params = (FCLayerParams *)params[i];
      // std::cout << "FChere" << i << std::endl;

      if (data_type == CUDNN_DATA_FLOAT) {
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, cur_params->C_out, batch_size,
                    cur_params->C_in, &Salpha, (float *)cur_params->W, cur_params->C_out,
                    (float *)layer_input[i], cur_params->C_in, &Sbeta, (float *)layer_input[i + 1],
                    cur_params->C_out);
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, cur_params->C_out, batch_size, 1,
                    &Salpha, (float *)cur_params->b, cur_params->C_out, (float *)one_vec, 1,
                    &Salpha, (float *)layer_input[i + 1], cur_params->C_out);
      } else if (data_type == CUDNN_DATA_DOUBLE) {
        cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, cur_params->C_out, batch_size,
                    cur_params->C_in, &Dalpha, (double *)cur_params->W, cur_params->C_out,
                    (double *)layer_input[i], cur_params->C_in, &Dbeta,
                    (double *)layer_input[i + 1], cur_params->C_out);
        cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, cur_params->C_out, batch_size, 1,
                    &Dalpha, (double *)cur_params->b, cur_params->C_out, (double *)one_vec, 1,
                    &Dalpha, (double *)layer_input[i + 1], cur_params->C_out);
      }
      if (cur_params->activation_mode != ACTIVATION_NONE) {
        checkCUDNN(cudnnActivationForward(cudnn_handle, cur_params->actv_desc, &alpha,
                                          cur_params->output_tensor, layer_input[i + 1], &beta,
                                          cur_params->output_tensor, layer_input[i + 1]));
      }
      // std::cout << "FChere" << i << std::endl;
    } else if (layer_type[i] == POOLING) {
      // std::cout << "Pooling\n";
      PoolingLayerParams *cur_params = (PoolingLayerParams *)params[i];
      checkCUDNN(cudnnPoolingForward(cudnn_handle, cur_params->pool_desc, &alpha,
                                     cur_params->input_tensor, layer_input[i], &beta,
                                     cur_params->output_tensor, layer_input[i + 1]));
    } else if (layer_type[i] == ACTV) {
      // std::cout << "Actv\n";
      std::cout << "Panic!! ACTV wrong place\n";
      exit(0);
      ActivationLayerParams *cur_params = (ActivationLayerParams *)params[i];
      checkCUDNN(cudnnActivationForward(cudnn_handle, cur_params->actv_desc, &alpha,
                                        cur_params->input_tensor, layer_input[i], &beta,
                                        cur_params->input_tensor, layer_input[i + 1]));
    } else if (layer_type[i] == SOFTMAX) {
      // std::cout << "Softmax\n";
      std::cout << "Panic!! SOFTMAX wrong place\n";
      exit(0);
      if (train == true) {
        SoftmaxLayerParams *cur_params = (SoftmaxLayerParams *)params[i];
        checkCUDNN(cudnnSoftmaxForward(cudnn_handle, cur_params->algo, cur_params->mode, &alpha,
                                       cur_params->input_tensor, layer_input[i], &beta,
                                       cur_params->input_tensor, layer_input[i + 1]));
      }
    }
    // synchronization
    // cudaDeviceSynchronize();

    // if next layer is ACTV or SOFTMAX, complete that and come to synchronization
    // the case in above if for ACTV and SOFTMAX never occurs
    if (layer_type[i + 1] == SOFTMAX) {
      i++;
      if (train == true) {
        layer_input[i + 1] = layer_input[i];
        SoftmaxLayerParams *cur_params = (SoftmaxLayerParams *)params[i];
        checkCUDNN(cudnnSoftmaxForward(cudnn_handle, cur_params->algo, cur_params->mode, &alpha,
                                       cur_params->input_tensor, layer_input[i], &beta,
                                       cur_params->input_tensor, layer_input[i + 1]));
      }
      i--;
    }
  }

  // std::cout << "here" << std::endl;
  if (train == false) {
    compareOutputCorrect(correct_count, y);
    return;
  }
  *scalar_loss = computeLoss();

  if (layer_type[num_layers - 1] == SOFTMAX) {
    // SoftmaxLayerParams *cur_params = (SoftmaxLayerParams *)params[num_layers - 1];
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
  for (int i = num_layers - 1; i >= 0; i--) {
    if (layer_type[i] == CONV) {
      // std::cout << "here\n";
      ConvLayerParams *cur_params = (ConvLayerParams *)params[i];

      if (cur_params->activation_mode != ACTIVATION_NONE) {
        checkCUDNN(cudnnActivationBackward(cudnn_handle, cur_params->actv_desc, &alpha,
                                           cur_params->output_tensor, layer_input[i + 1],
                                           cur_params->output_tensor, dlayer_input[i + 1],
                                           cur_params->output_tensor, layer_input[i + 1], &beta,
                                           cur_params->output_tensor, dlayer_input[i + 1]));
      }

      checkCUDNN(cudnnConvolutionBackwardBias(cudnn_handle, &alpha, cur_params->output_tensor,
                                              dlayer_input[i + 1], &beta, cur_params->bias_desc,
                                              cur_params->db));

      // std::cout << "neural_net: backward conv i:" << i << std::endl;

      checkCUDNN(cudnnConvolutionBackwardFilter(
          cudnn_handle, &alpha, cur_params->input_tensor, layer_input[i], cur_params->output_tensor,
          dlayer_input[i + 1], cur_params->conv_desc, cur_params->bwd_filter_algo, workspace,
          workspace_size, &beta, cur_params->filter_desc, cur_params->dW));
      if (i > 0)
        checkCUDNN(cudnnConvolutionBackwardData(
            cudnn_handle, &alpha, cur_params->filter_desc, cur_params->W, cur_params->output_tensor,
            dlayer_input[i + 1], cur_params->conv_desc, cur_params->bwd_data_algo, workspace,
            workspace_size, &beta, cur_params->input_tensor, dlayer_input[i]));

      // std::cout << "Free bytes: " << free_bytes << std::endl;
      // std::cout << "here\n";
      cur_params->stepParams(cublas_handle, learning_rate);
    }

    else if (layer_type[i] == FULLY_CONNECTED) {
      FCLayerParams *cur_params = (FCLayerParams *)params[i];

      if (cur_params->activation_mode != ACTIVATION_NONE) {
        checkCUDNN(cudnnActivationBackward(cudnn_handle, cur_params->actv_desc, &alpha,
                                           cur_params->output_tensor, layer_input[i + 1],
                                           cur_params->output_tensor, dlayer_input[i + 1],
                                           cur_params->output_tensor, layer_input[i + 1], &beta,
                                           cur_params->output_tensor, dlayer_input[i + 1]));
      }

      if (data_type == CUDNN_DATA_FLOAT) {
        // bias backward
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, cur_params->C_out, 1, batch_size,
                    &Salpha, (float *)dlayer_input[i + 1], cur_params->C_out, (float *)one_vec,
                    batch_size, &Sbeta, (float *)cur_params->db, cur_params->C_out);

        // weight backward
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, cur_params->C_out, cur_params->C_in,
                    batch_size, &Salpha, (float *)dlayer_input[i + 1], cur_params->C_out,
                    (float *)layer_input[i], cur_params->C_in, &Sbeta, (float *)cur_params->dW,
                    cur_params->C_out);

        // data backward
        if (i > 0)
          cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, cur_params->C_in, batch_size,
                      cur_params->C_out, &Salpha, (float *)cur_params->W, cur_params->C_out,
                      (float *)dlayer_input[i + 1], cur_params->C_out, &Sbeta,
                      (float *)dlayer_input[i], cur_params->C_in);
      }

      else if (data_type == CUDNN_DATA_DOUBLE) {
        // bias backward
        cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, cur_params->C_out, 1, batch_size,
                    &Dalpha, (double *)dlayer_input[i + 1], cur_params->C_out, (double *)one_vec,
                    batch_size, &Dbeta, (double *)cur_params->db, cur_params->C_out);

        // weight backward
        cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, cur_params->C_out, cur_params->C_in,
                    batch_size, &Dalpha, (double *)dlayer_input[i + 1], cur_params->C_out,
                    (double *)layer_input[i], cur_params->C_in, &Dbeta, (double *)cur_params->dW,
                    cur_params->C_out);

        // data backward
        if (i > 0)
          cublasDgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, cur_params->C_in, batch_size,
                      cur_params->C_out, &Dalpha, (double *)cur_params->W, cur_params->C_out,
                      (double *)dlayer_input[i + 1], cur_params->C_out, &Dbeta,
                      (double *)dlayer_input[i], cur_params->C_in);
      }
      cur_params->stepParams(cublas_handle, learning_rate);
    }

    else if (layer_type[i] == POOLING) {
      PoolingLayerParams *cur_params = (PoolingLayerParams *)params[i];
      checkCUDNN(cudnnPoolingBackward(cudnn_handle, cur_params->pool_desc, &alpha,
                                      cur_params->output_tensor, layer_input[i + 1],
                                      cur_params->output_tensor, dlayer_input[i + 1],
                                      cur_params->input_tensor, layer_input[i], &beta,
                                      cur_params->input_tensor, dlayer_input[i]));
    }

    else if (layer_type[i] == ACTV) {
      ActivationLayerParams *cur_params = (ActivationLayerParams *)params[i];
      checkCUDNN(cudnnActivationBackward(
          cudnn_handle, cur_params->actv_desc, &alpha, cur_params->input_tensor, layer_input[i + 1],
          cur_params->input_tensor, dlayer_input[i + 1], cur_params->input_tensor, layer_input[i],
          &beta, cur_params->input_tensor, dlayer_input[i]));
      continue;
    }

    else if (layer_type[i] == SOFTMAX) {
      // std::cout << "compute here\n";
      SoftmaxLayerParams *cur_params = (SoftmaxLayerParams *)params[i];
      checkCUDNN(cudnnSoftmaxBackward(cudnn_handle, cur_params->algo, cur_params->mode, &alpha,
                                      cur_params->input_tensor, layer_input[i + 1],
                                      cur_params->input_tensor, dlayer_input[i + 1], &beta,
                                      cur_params->input_tensor, dlayer_input[i]));
      // std::cout << "compute here\n";
      continue;
    }

    // exit(0);
  }
}
