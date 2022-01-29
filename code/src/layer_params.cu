#include <cublas_v2.h>
#include <cudnn.h>
#include <curand.h>

#include "layer_params.cuh"

void ConvLayerParams::initializeValues(cudnnHandle_t cudnn_handle, ConvDescriptor *user_params,
                                       cudnnDataType_t data_type, int batch_size,
                                       cudnnTensorFormat_t tensor_format, size_t data_type_size,
                                       LayerDimension &output_size, UpdateRule update_rule) {
  // create tensor, filter, conv descriptor
  checkCUDNN(cudnnCreateTensorDescriptor(&input_tensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&output_tensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&bias_desc));
  checkCUDNN(cudnnCreateFilterDescriptor(&filter_desc));
  checkCUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));

  C_in = user_params->input_channels;
  C_out = user_params->output_channels;
  filter_h = user_params->kernel_h;
  filter_w = user_params->kernel_w;
  kernel_size = C_out * C_in * filter_h * filter_w;
  this->data_type = data_type;
  this->activation_mode = user_params->activation_mode;

  checkCUDNN(cudnnSetTensor4dDescriptor(input_tensor, tensor_format, data_type, batch_size,
                                        user_params->input_channels, user_params->input_h,
                                        user_params->input_w));

  checkCUDNN(cudnnSetFilter4dDescriptor(filter_desc, data_type, tensor_format,
                                        user_params->output_channels, user_params->input_channels,
                                        user_params->kernel_h, user_params->kernel_w));

  int dilation_h = 1, dilation_w = 1;
  checkCUDNN(cudnnSetConvolution2dDescriptor(
      conv_desc, user_params->pad_h, user_params->pad_w, user_params->stride_y,
      user_params->stride_x, dilation_h, dilation_w, CUDNN_CROSS_CORRELATION, data_type));

  int output_batch_size, output_channels, output_h, output_w;
  checkCUDNN(cudnnGetConvolution2dForwardOutputDim(conv_desc, input_tensor, filter_desc,
                                                   &output_batch_size, &output_channels, &output_h,
                                                   &output_w));

  checkCUDNN(cudnnSetTensor4dDescriptor(output_tensor, tensor_format, data_type, output_batch_size,
                                        output_channels, output_h, output_w));
  checkCUDNN(
      cudnnSetTensor4dDescriptor(bias_desc, tensor_format, data_type, 1, output_channels, 1, 1));

  fwd_req_count = 10;
  fwd_perf = (cudnnConvolutionFwdAlgoPerf_t *)malloc(fwd_req_count *
                                                     sizeof(cudnnConvolutionFwdAlgoPerf_t));
  checkCUDNN(cudnnFindConvolutionForwardAlgorithm(cudnn_handle, input_tensor, filter_desc,
                                                  conv_desc, output_tensor, fwd_req_count,
                                                  &fwd_ret_count, fwd_perf));

  bwd_filter_req_count = 10;
  bwd_filter_perf = (cudnnConvolutionBwdFilterAlgoPerf_t *)malloc(
      bwd_filter_req_count * sizeof(cudnnConvolutionBwdFilterAlgoPerf_t));
  checkCUDNN(cudnnFindConvolutionBackwardFilterAlgorithm(
      cudnn_handle, input_tensor, output_tensor, conv_desc, filter_desc, bwd_filter_req_count,
      &bwd_filter_ret_count, bwd_filter_perf));

  bwd_data_req_count = 10;
  bwd_data_perf = (cudnnConvolutionBwdDataAlgoPerf_t *)malloc(
      bwd_data_req_count * sizeof(cudnnConvolutionBwdDataAlgoPerf_t));
  checkCUDNN(cudnnFindConvolutionBackwardDataAlgorithm(cudnn_handle, filter_desc, output_tensor,
                                                       conv_desc, input_tensor, bwd_data_req_count,
                                                       &bwd_data_ret_count, bwd_data_perf));

  this->update_rule = update_rule;

  cudnnActivationMode_t mode;
  if (activation_mode == SIGMOID)
    mode = CUDNN_ACTIVATION_SIGMOID;
  else if (activation_mode == RELU)
    mode = CUDNN_ACTIVATION_RELU;
  else if (activation_mode == TANH)
    mode = CUDNN_ACTIVATION_TANH;
  else if (activation_mode == CLIPPED_RELU)
    mode = CUDNN_ACTIVATION_CLIPPED_RELU;
  else if (activation_mode == ELU)
    mode = CUDNN_ACTIVATION_ELU;

  if (activation_mode != ACTIVATION_NONE) {
    checkCUDNN(cudnnCreateActivationDescriptor(&actv_desc));
    checkCUDNN(
        cudnnSetActivationDescriptor(actv_desc, mode, CUDNN_PROPAGATE_NAN, user_params->actv_coef));
  }

  output_size.N = output_batch_size, output_size.C = output_channels, output_size.H = output_h,
  output_size.W = output_w;
}

void ConvLayerParams::allocateSpace(curandGenerator_t curand_gen, cudnnDataType_t data_type,
                                    size_t data_type_size, float std_dev, size_t &free_bytes) {
  if (kernel_size % 2 != 0) kernel_size += 1;
  cudaMalloc(&W, kernel_size * data_type_size);
  cudaMalloc(&b, C_out * data_type_size);

  // std::cout << (kernel_size+C_out)*data_type_size << "\n";

  cudaMalloc(&dW, kernel_size * data_type_size);
  cudaMalloc(&db, C_out * data_type_size);

  if (data_type == CUDNN_DATA_FLOAT) {
    curandGenerateNormal(curand_gen, (float *)W, kernel_size, 0, std_dev);
    fillValue<float><<<ceil(1.0 * C_out / BW), BW>>>((float *)b, C_out, 0);
  } else {
    curandGenerateNormalDouble(curand_gen, (double *)W, kernel_size, 0, std_dev);
    fillValue<double><<<ceil(1.0 * C_out / BW), BW>>>((double *)b, C_out, 0);
  }

  free_bytes = free_bytes - 2 * (kernel_size + C_out) * data_type_size;
}

void ConvLayerParams::stepParams(cublasHandle_t cublas_handle, double learning_rate) {
  float Salpha = -learning_rate;
  double Dalpha = -learning_rate;

  if (update_rule == SGD) {
    if (data_type == CUDNN_DATA_FLOAT) {
      cublasSaxpy(cublas_handle, kernel_size, &Salpha, (float *)dW, 1, (float *)W, 1);

      cublasSaxpy(cublas_handle, C_out, &Salpha, (float *)db, 1, (float *)b, 1);
    } else if (data_type == CUDNN_DATA_DOUBLE) {
      cublasDaxpy(cublas_handle, kernel_size, &Dalpha, (double *)dW, 1, (double *)W, 1);

      cublasDaxpy(cublas_handle, C_out, &Dalpha, (double *)db, 1, (double *)b, 1);
    }
  }
}

size_t ConvLayerParams::getWorkspaceSize(size_t &free_bytes,
                                         ConvLayerParams::ConvDirection conv_direction) {
  size_t gb = 1024 * 1024 * 1024;
  if (conv_direction == FWD) {
    // If fwd algo uses workspace more than 6 GB
    if (fwd_perf[0].memory / float(gb) > WORKSPACE_SIZE_LIMIT || fwd_perf[0].memory > free_bytes) {
      // Iteratr over all fwd algo and find the one with less memory than 6 GB
      for (int i = 0; i < fwd_ret_count; i++) {
        // If fwd algo uses less memory than 6 GB
        if (fwd_perf[i].status == CUDNN_STATUS_SUCCESS &&
            fwd_perf[i].memory / float(gb) < WORKSPACE_SIZE_LIMIT) {
          fwd_algo = fwd_perf[i].algo;              // Set the fwd algo
          fwd_workspace_size = fwd_perf[i].memory;  // Set the fwd workspace size
          return fwd_perf[i].memory;                // Return the fwd workspace size
        }
      }

      outOfMemory();
    }

    // if (fwd_perf[0].memory > free_bytes) outOfMemory();
    fwd_algo = fwd_perf[0].algo;
    fwd_workspace_size = fwd_perf[0].memory;
    return fwd_perf[0].memory;

  } else if (conv_direction == BWD_FILTER) {
    if (bwd_filter_perf[0].memory / float(gb) > WORKSPACE_SIZE_LIMIT ||
        bwd_filter_perf[0].memory > free_bytes) {
      for (int i = 0; i < bwd_filter_ret_count; i++) {
        if (bwd_filter_perf[i].status == CUDNN_STATUS_SUCCESS &&
            bwd_filter_perf[i].memory / float(gb) < WORKSPACE_SIZE_LIMIT) {
          bwd_filter_algo = bwd_filter_perf[i].algo;
          bwd_filter_workspace_size = bwd_filter_perf[i].memory;
          return bwd_filter_perf[i].memory;
        }
      }

      outOfMemory();
    }
    // if (bwd_filter_perf[0].memory > free_bytes) outOfMemory();
    bwd_filter_algo = bwd_filter_perf[0].algo;
    bwd_filter_workspace_size = bwd_filter_perf[0].memory;
    return bwd_filter_perf[0].memory;
  } else if (conv_direction == BWD_DATA) {
    if (bwd_filter_perf[0].memory / float(gb) > WORKSPACE_SIZE_LIMIT ||
        bwd_data_perf[0].memory > free_bytes) {
      for (int i = 0; i < bwd_data_ret_count; i++) {
        if (bwd_data_perf[i].status == CUDNN_STATUS_SUCCESS &&
            bwd_data_perf[i].memory / float(gb) < WORKSPACE_SIZE_LIMIT) {
          bwd_data_algo = bwd_data_perf[i].algo;
          bwd_data_workspace_size = bwd_data_perf[i].memory;
          return bwd_data_perf[i].memory;
        }
      }

      outOfMemory();
    }
    // if (bwd_data_perf[0].memory > free_bytes) outOfMemory();
    bwd_data_algo = bwd_data_perf[0].algo;
    bwd_data_workspace_size = bwd_data_perf[0].memory;
    return bwd_data_perf[0].memory;
  }
  return 0;
}

void FCLayerParams::initializeValues(FCDescriptor *user_params, int batch_size,
                                     cudnnTensorFormat_t tensor_format, cudnnDataType_t data_type,
                                     LayerDimension &output_size, UpdateRule update_rule) {
  C_in = user_params->input_channels;
  C_out = user_params->output_channels;
  weight_matrix_size = C_in * C_out;
  this->data_type = data_type;
  this->activation_mode = user_params->activation_mode;

  this->update_rule = update_rule;

  cudnnActivationMode_t mode;
  if (activation_mode == SIGMOID)
    mode = CUDNN_ACTIVATION_SIGMOID;
  else if (activation_mode == RELU)
    mode = CUDNN_ACTIVATION_RELU;
  else if (activation_mode == TANH)
    mode = CUDNN_ACTIVATION_TANH;
  else if (activation_mode == CLIPPED_RELU)
    mode = CUDNN_ACTIVATION_CLIPPED_RELU;
  else if (activation_mode == ELU)
    mode = CUDNN_ACTIVATION_ELU;

  if (activation_mode != ACTIVATION_NONE) {
    checkCUDNN(cudnnCreateActivationDescriptor(&actv_desc));
    checkCUDNN(
        cudnnSetActivationDescriptor(actv_desc, mode, CUDNN_PROPAGATE_NAN, user_params->actv_coef));
    checkCUDNN(cudnnCreateTensorDescriptor(&output_tensor));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_tensor, tensor_format, data_type, batch_size,
                                          user_params->output_channels, 1, 1));
  }

  output_size.N = batch_size, output_size.C = C_out, output_size.H = output_size.W = 1;
}

void FCLayerParams::allocateSpace(curandGenerator_t curand_gen, cudnnDataType_t data_type,
                                  size_t data_type_size, float std_dev, size_t &free_bytes) {
  int wt_alloc_size = weight_matrix_size;
  if (wt_alloc_size % 2 != 0) wt_alloc_size += 1;
  cudaMalloc(&W, wt_alloc_size * data_type_size);
  cudaMalloc(&b, C_out * data_type_size);
  // std::cout << (wt_alloc_size+C_out)*data_type_size << "\n";
  cudaMalloc(&dW, wt_alloc_size * data_type_size);
  cudaMalloc(&db, C_out * data_type_size);

  if (data_type == CUDNN_DATA_FLOAT) {
    curandGenerateNormal(curand_gen, (float *)W, wt_alloc_size, 0, std_dev);
    fillValue<float><<<ceil(1.0 * C_out / BW), BW>>>((float *)b, C_out, 0);
  } else if (data_type == CUDNN_DATA_DOUBLE) {
    curandGenerateNormalDouble(curand_gen, (double *)W, wt_alloc_size, 0, std_dev);
    fillValue<double><<<ceil(1.0 * C_out / BW), BW>>>((double *)b, C_out, 0);
  }
  free_bytes = free_bytes - 2 * (C_in * C_out + C_out) * data_type_size;
}

void FCLayerParams::stepParams(cublasHandle_t cublas_handle, double learning_rate) {
  float Salpha = -learning_rate;
  double Dalpha = -learning_rate;

  if (update_rule == SGD) {
    if (data_type == CUDNN_DATA_FLOAT) {
      cublasSaxpy(cublas_handle, weight_matrix_size, &Salpha, (float *)dW, 1, (float *)W, 1);

      cublasSaxpy(cublas_handle, C_out, &Salpha, (float *)db, 1, (float *)b, 1);
    } else if (data_type == CUDNN_DATA_DOUBLE) {
      cublasDaxpy(cublas_handle, weight_matrix_size, &Dalpha, (double *)dW, 1, (double *)W, 1);

      cublasDaxpy(cublas_handle, C_out, &Dalpha, (double *)db, 1, (double *)b, 1);
    }
  }
}

void PoolingLayerParams::initializeValues(PoolingDescriptor *user_params, cudnnDataType_t data_type,
                                          cudnnTensorFormat_t tensor_format, int batch_size,
                                          LayerDimension &output_size) {
  checkCUDNN(cudnnCreateTensorDescriptor(&input_tensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&output_tensor));

  checkCUDNN(cudnnSetTensor4dDescriptor(input_tensor, tensor_format, data_type, batch_size,
                                        user_params->input_channels, user_params->input_h,
                                        user_params->input_w));

  checkCUDNN(cudnnCreatePoolingDescriptor(&pool_desc));

  cudnnPoolingMode_t mode;
  if (user_params->mode == POOLING_MAX)
    mode = CUDNN_POOLING_MAX;
  else if (user_params->mode == POOLING_AVERAGE_COUNT_INCLUDE_PADDING)
    mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
  else if (user_params->mode == POOLING_AVERAGE_COUNT_EXCLUDE_PADDING)
    mode = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;

  checkCUDNN(cudnnSetPooling2dDescriptor(
      pool_desc, mode, CUDNN_PROPAGATE_NAN, user_params->kernel_h, user_params->kernel_w,
      user_params->pad_h, user_params->pad_w, user_params->stride_y, user_params->stride_x));

  int output_batch_size, output_channels, output_h, output_w;
  checkCUDNN(cudnnGetPooling2dForwardOutputDim(pool_desc, input_tensor, &output_batch_size,
                                               &output_channels, &output_h, &output_w));

  checkCUDNN(cudnnSetTensor4dDescriptor(output_tensor, tensor_format, data_type, output_batch_size,
                                        output_channels, output_h, output_w));

  output_size.N = output_batch_size, output_size.C = output_channels, output_size.H = output_h,
  output_size.W = output_w;
}

void PoolingLayerParams::allocateSpace(size_t &free_bytes) {}

void ActivationLayerParams::initializeValues(ActivationDescriptor *user_params,
                                             cudnnDataType_t data_type,
                                             cudnnTensorFormat_t tensor_format, int batch_size,
                                             LayerDimension &output_size) {
  checkCUDNN(cudnnCreateTensorDescriptor(&input_tensor));

  checkCUDNN(cudnnSetTensor4dDescriptor(input_tensor, tensor_format, data_type, batch_size,
                                        user_params->channels, user_params->h, user_params->w));

  cudnnActivationMode_t mode;
  if (user_params->mode == SIGMOID)
    mode = CUDNN_ACTIVATION_SIGMOID;
  else if (user_params->mode == RELU)
    mode = CUDNN_ACTIVATION_RELU;
  else if (user_params->mode == TANH)
    mode = CUDNN_ACTIVATION_TANH;
  else if (user_params->mode == CLIPPED_RELU)
    mode = CUDNN_ACTIVATION_CLIPPED_RELU;
  else if (user_params->mode == ELU)
    mode = CUDNN_ACTIVATION_ELU;

  checkCUDNN(cudnnCreateActivationDescriptor(&actv_desc));
  checkCUDNN(cudnnSetActivationDescriptor(actv_desc, mode, CUDNN_PROPAGATE_NAN, user_params->coef));

  output_size.N = batch_size, output_size.C = user_params->channels, output_size.H = user_params->h,
  output_size.W = user_params->w;
}

void ActivationLayerParams::allocateSpace(size_t &free_bytes) {}

void SoftmaxLayerParams::initializeValues(SoftmaxDescriptor *user_params, cudnnDataType_t data_type,
                                          cudnnTensorFormat_t tensor_format, int batch_size,
                                          LayerDimension &output_size) {
  if (user_params->algo == SOFTMAX_FAST)
    algo = CUDNN_SOFTMAX_FAST;
  else if (user_params->algo == SOFTMAX_ACCURATE)
    algo = CUDNN_SOFTMAX_ACCURATE;

  if (user_params->mode == SOFTMAX_MODE_INSTANCE)
    mode = CUDNN_SOFTMAX_MODE_INSTANCE;
  else if (user_params->mode == SOFTMAX_MODE_CHANNEL) {
    mode = CUDNN_SOFTMAX_MODE_CHANNEL;
  }

  checkCUDNN(cudnnCreateTensorDescriptor(&input_tensor));
  checkCUDNN(cudnnSetTensor4dDescriptor(input_tensor, tensor_format, data_type, batch_size,
                                        user_params->channels, user_params->h, user_params->w));

  output_size.N = batch_size, output_size.C = user_params->channels, output_size.H = user_params->h,
  output_size.W = user_params->w;
}

void SoftmaxLayerParams::allocateSpace(size_t &free_bytes) {}