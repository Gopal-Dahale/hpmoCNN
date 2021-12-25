#include <cublas_v2.h>
#include <cudnn.h>
#include <curand.h>
#include <time.h>

#include <cstdio>
#include <string>

#include "neural_net.cuh"

NeuralNet::NeuralNet()
{
  this->num_layers = 0;
  this->batch_size = 0;
}

NeuralNet::NeuralNet(std::vector<LayerSpecifier> &layers, DataType data_type,
                     int batch_size, TensorFormat tensor_format,
                     float softmax_eps, float init_std_dev,
                     UpdateRule update_rule)
{
  cudaStreamCreate(&stream_compute);
  cudaStreamCreate(&stream_memory);

  // create handle
  checkCUDNN(cudnnCreate(&cudnn_handle));
  checkCUDNN(cudnnSetStream(cudnn_handle, stream_compute));

  cublasCreate(&cublas_handle);
  cublasSetStream(cublas_handle, stream_compute);

  curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetStream(curand_gen, stream_compute);

  cudaMemGetInfo(&free_bytes, &total_bytes);
  init_free_bytes = free_bytes;
  std::cout << "Free gigabytes at start: "
            << free_bytes / (1024.0 * 1024.0 * 1024.0) << std::endl;

  if (data_type == DATA_FLOAT)
  {
    this->data_type = CUDNN_DATA_FLOAT;
    data_type_size = sizeof(float);
  }

  else if (data_type == DATA_DOUBLE)
  {
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

  // Allocation of space for input to each layer
  layer_input = (void **)malloc((num_layers + 1) * sizeof(void *));
  layer_input_size = (int *)malloc((num_layers + 1) * sizeof(int));
  dlayer_input = (void **)malloc((num_layers + 1) * sizeof(void *));
  params = (void **)malloc(num_layers * sizeof(void *));

  LayerDimension current_output_size;
  for (int i = 0; i < num_layers; i++)
  {
    layer_type.push_back(layers[i].type);
    if (layers[i].type == CONV)
    {
      ConvDescriptor *user_params = (ConvDescriptor *)layers[i].params;
      params[i] = malloc(sizeof(ConvLayerParams));
      ((ConvLayerParams *)params[i])
          ->initializeValues(cudnn_handle, user_params, this->data_type,
                             batch_size, this->tensor_format, data_type_size,
                             current_output_size, update_rule);
    }
    else if (layers[i].type == FULLY_CONNECTED)
    {
      FCDescriptor *user_params = (FCDescriptor *)layers[i].params;
      params[i] = malloc(sizeof(FCLayerParams));
      ((FCLayerParams *)params[i])
          ->initializeValues(user_params, batch_size, this->tensor_format,
                             this->data_type, current_output_size, update_rule);
    }
    else if (layers[i].type == POOLING)
    {
      PoolingDescriptor *user_params = (PoolingDescriptor *)layers[i].params;
      params[i] = malloc(sizeof(PoolingLayerParams));
      ((PoolingLayerParams *)params[i])
          ->initializeValues(user_params, this->data_type, this->tensor_format,
                             batch_size, current_output_size);
    }
    else if (layers[i].type == ACTV)
    {
      ActivationDescriptor *user_params =
          (ActivationDescriptor *)layers[i].params;
      params[i] = malloc(sizeof(ActivationLayerParams));
      ((ActivationLayerParams *)params[i])
          ->initializeValues(user_params, this->data_type, this->tensor_format,
                             batch_size, current_output_size);
    }

    else if (layers[i].type == SOFTMAX)
    {
      SoftmaxDescriptor *user_params = (SoftmaxDescriptor *)layers[i].params;
      params[i] = malloc(sizeof(SoftmaxLayerParams));
      ((SoftmaxLayerParams *)params[i])
          ->initializeValues(user_params, this->data_type, this->tensor_format,
                             batch_size, current_output_size);
    }
  }

  h_layer_input = (void **)malloc((num_layers + 1) * sizeof(void *)); // host
  offloaded =
      (bool *)calloc((num_layers + 1), sizeof(bool)); // Offloaded layers index

  cudaMemGetInfo(&free_bytes, &total_bytes);
  std::cout << "Free gigabytes just before allocate space: "
            << free_bytes / (1024.0 * 1024.0 * 1024.0) << std::endl;

  // Allocate space for parameters
  for (int i = 0; i < num_layers; i++)
  {
    size_t input_size;
    if (layers[i].type == CONV)
    {
      ConvDescriptor *user_params = (ConvDescriptor *)layers[i].params;
      ((ConvLayerParams *)params[i])
          ->allocateSpace(curand_gen, this->data_type, data_type_size,
                          init_std_dev, free_bytes);

      input_size = batch_size * user_params->input_channels *
                   user_params->input_h * user_params->input_w;
      if (i == 0)
      {
        input_channels = user_params->input_channels;
        input_h = user_params->input_h;
        input_w = user_params->input_w;
      }
    }
    else if (layers[i].type == FULLY_CONNECTED)
    {
      FCDescriptor *user_params = (FCDescriptor *)layers[i].params;
      ((FCLayerParams *)params[i])
          ->allocateSpace(curand_gen, this->data_type, data_type_size,
                          init_std_dev, free_bytes);
      input_size = batch_size * user_params->input_channels;
      if (i == 0)
      {
        input_channels = user_params->input_channels;
        input_h = 1;
        input_w = 1;
      }
    }
    else if (layers[i].type == POOLING)
    {
      PoolingDescriptor *user_params = (PoolingDescriptor *)layers[i].params;
      ((PoolingLayerParams *)params[i])->allocateSpace(free_bytes);
      input_size = batch_size * user_params->input_channels *
                   user_params->input_h * user_params->input_w;
      if (i == 0)
      {
        input_channels = user_params->input_channels;
        input_h = user_params->input_h;
        input_w = user_params->input_w;
      }
    }
    else if (layers[i].type == ACTV)
    {
      ActivationDescriptor *user_params =
          (ActivationDescriptor *)layers[i].params;
      ((ActivationLayerParams *)params[i])->allocateSpace(free_bytes);
      input_size =
          batch_size * user_params->channels * user_params->h * user_params->w;
      if (i == 0)
      {
        input_channels = user_params->channels;
        input_h = user_params->h;
        input_w = user_params->w;
      }
    }
    else if (layers[i].type == SOFTMAX)
    {
      SoftmaxDescriptor *user_params = (SoftmaxDescriptor *)layers[i].params;
      ((SoftmaxLayerParams *)params[i])->allocateSpace(free_bytes);
      input_size =
          batch_size * user_params->channels * user_params->h * user_params->w;

      // assuming this is last layer, allocate for next layer as well
      //       cudaMallocManaged(&layer_input[i + 1], input_size *
      //       data_type_size); cudaMallocManaged(&dlayer_input[i + 1],
      //       input_size * data_type_size);
      layer_input_size[i + 1] = input_size;
      if (i == 0)
      {
        input_channels = user_params->channels;
        input_h = user_params->h;
        input_w = user_params->w;
      }
      if (i == num_layers - 1)
        num_classes = user_params->channels;
    }

    //     cudaMallocManaged(&layer_input[i], input_size * data_type_size);
    //     cudaMallocManaged(&dlayer_input[i], input_size * data_type_size);
    layer_input_size[i] = input_size;
  }
  cudaDeviceSynchronize();
  cudaMemGetInfo(&free_bytes, &total_bytes);
  std::cout << "Free gigabytes just after allocate space: "
            << free_bytes / (1024.0 * 1024.0 * 1024.0) << std::endl;

  // Very small - could be allocated initially itself
  cudaMallocManaged((void **)&y, batch_size * sizeof(int));
  cudaMallocManaged((void **)&pred_y, batch_size * sizeof(int));
  cudaMallocManaged((void **)&loss, batch_size * sizeof(float));
  cudaMallocManaged(&one_vec, batch_size * data_type_size);

  if (this->data_type == CUDNN_DATA_FLOAT)
    fillValue<float>
        <<<ceil(1.0 * batch_size / BW), BW>>>((float *)one_vec, batch_size, 1);
  else
    fillValue<double>
        <<<ceil(1.0 * batch_size / BW), BW>>>((double *)one_vec, batch_size, 1);

  // Allocate space for workspace
  size_t cur_workspace_size_1, cur_workspace_size_2, cur_workspace_size_3,
      cur_workspace_size;
  this->workspace_size = 0;
  for (int i = 0; i < num_layers; i++)
  {
    if (layers[i].type == CONV)
    {
      cur_workspace_size_1 =
          ((ConvLayerParams *)params[i])
              ->getWorkspaceSize(free_bytes, ConvLayerParams::FWD);
      cur_workspace_size_2 =
          ((ConvLayerParams *)params[i])
              ->getWorkspaceSize(free_bytes, ConvLayerParams::BWD_DATA);
      cur_workspace_size_3 =
          ((ConvLayerParams *)params[i])
              ->getWorkspaceSize(free_bytes, ConvLayerParams::BWD_FILTER);
      cur_workspace_size = max(cur_workspace_size_1,
                               max(cur_workspace_size_2, cur_workspace_size_3));
      if (cur_workspace_size > workspace_size)
        this->workspace_size = cur_workspace_size;
    }
  }

  cudaMalloc(&(this->workspace), this->workspace_size);
  free_bytes = free_bytes - this->workspace_size;
  cudaDeviceSynchronize();
  cudaMemGetInfo(&free_bytes, &total_bytes);

  // Leave 600 MB and use the rest
  std::cout << "Free gigabytes: " << free_bytes / (1024.0 * 1024.0 * 1024.0)
            << std::endl;
  free_bytes -= 1024 * 1024 * 600;

  cudaDeviceSynchronize();

  size_t temp_free_bytes;
  cudaMemGetInfo(&temp_free_bytes, &total_bytes);
  std::cout << "Free gigabytes just before end of NeuralNet: "
            << temp_free_bytes / (1024.0 * 1024.0 * 1024.0) << std::endl;

  // Data of time
  cudaEventCreate(&start_compute);
  cudaEventCreate(&stop_compute);

  cudaEventCreate(&start_transfer);
  cudaEventCreate(&stop_transfer);
}
