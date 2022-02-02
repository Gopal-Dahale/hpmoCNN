#include "gpu_timer.cuh"
#include "neural_net.cuh"
#include "utils.cuh"

void NeuralNet::max_heap_policy(std::vector<std::pair<size_t, size_t>> &offload_mem,
                                std::vector<int> &free_layer, int &i) {
  size_t buffer_bytes = 1024 * 1024 * 1024;  // 2GB
  cudaMemGetInfo(&free_bytes, &total_bytes);
  size_t temp_free_bytes = free_bytes;                                 // Current free bytes
  size_t free_memory = temp_free_bytes - buffer_bytes - buffer_bytes;  // reserved memory is 2GB
  size_t layer_size = layer_input_size[i + 2] * data_type_size;        // Size of the layer

  // Decrement free_memory by i+2 th layer fwd workspace size.
  // This make sures that i+2 the layer can be allocated smoothly when
  // needed
  if ((i + 2 < num_layers) && (layer_type[i + 2] == CONV)) {
    ConvLayerParams *cur_params = (ConvLayerParams *)params[i + 2];
    layer_size += cur_params->fwd_workspace_size;
  }

  if ((i + 2 < num_layers) && (free_memory <= layer_size)) {
    std::cout << "Not enough memory to allocate layer " << i + 2 << '\n';
    std::cout << "Free memory: " << std::setprecision(2) << free_memory / (1024.0 * 1024.0) << " MB"
              << '\n';
    std::cout << "Layer size: " << std::setprecision(2) << layer_size / (1024.0 * 1024.0) << " MB"
              << '\n';

    offload_mem.push_back({free_memory /*(reserved_memory/(i+1))*/, layer_size});

    /************* Heap logic with workspace fix ********************/

    // While the free memory is less than or equal to the (i+2)th layer
    // input size or the heap is not empty
    while ((free_memory <= layer_size) && (!layer_input_pq.empty())) {
      std::cout << "Deallocated layer " << layer_input_pq.top().second << '\n';
      int temp = layer_input_pq.top().second;  // Get the layer index on top
                                               // of the heap
      free_layer.push_back(temp);              // Add the layer index to the free layer
      // vector

      offload_mem.push_back({temp, layer_input_pq.top().first * data_type_size});

      // Update the free bytes
      temp_free_bytes += layer_input_pq.top().first * data_type_size;
      offloaded[temp] = true;  // Mark the layer as offloaded

      // Copy the layer to host
      // Only offload layer_input and not workspace
      cudaMemcpyAsync(h_layer_input[temp], layer_input[temp],
                      layer_input_size[temp] * data_type_size, cudaMemcpyDeviceToHost,
                      stream_memory);
      layer_input_pq.pop();  // Remove the layer from the heap
      free_memory = temp_free_bytes - buffer_bytes - buffer_bytes;  // (reserved_memory/(i+1));
    }
    std::cout << "Free Memory: " << std::setprecision(2) << free_memory / (1024.0 * 1024.0) << " MB"
              << '\n';
    /*************************************************************/
  }
}

void NeuralNet::forward_prop(bool &train, std::vector<std::pair<size_t, size_t>> &offload_mem,
                             float &alpha, float &beta, float &Salpha, float &Sbeta, double &Dalpha,
                             double &Dbeta, std::vector<float> &fwd_dnn_lag, float *overhead) {
  std::cout << "Forward Propagation Starts" << '\n';
  std::vector<int> free_layer;  // Which layers to free
  for (int i = 0; i < num_layers; i++) {
    float milli = 0;
    if (train == false && i == num_layers - 1) {
      break;
    }
    GpuTimer timer;
    cudaMalloc(&layer_input[i + 1], layer_input_size[i + 1] * data_type_size);
    std::cout << "Allocated layer " << i + 1 << " Size: " << std::setprecision(2)
              << layer_input_size[i + 1] * data_type_size / (1024.0 * 1024.0) << " MB" << '\n';

    // Push the layer_input_size + weights_size to the heap of ith layer
    if (i > 0) {
      layer_input_pq.push({layer_input_size[i], i});
    }

    timer.start();
    max_heap_policy(offload_mem, free_layer, i);
    timer.stop();
    milli = timer.elapsed();

    cudaMemGetInfo(&free_bytes, &total_bytes);
    switch (layer_type[i]) {
      case CONV:
        conv_forward(i, alpha, beta);
        break;
      case FULLY_CONNECTED:
        fc_forward(i, alpha, beta, Salpha, Sbeta, Dalpha, Dbeta);
        break;
      case POOLING:
        pooling_forward(i, alpha, beta);
        break;
      case ACTV:
        activation_forward(i, alpha, beta);
        break;
      default:
        break;
    }

    // if next layer is ACTV or SOFTMAX, complete that and come to
    // synchronization the case in above if for ACTV and SOFTMAX never occurs
    if (layer_type[i + 1] == SOFTMAX) {
      i++;
      if (train == true) {
        layer_input[i + 1] = layer_input[i];
        softmax_forward(i, alpha, beta);
      }
    }
    cudaStreamSynchronize(stream_compute);

    timer.start(stream_memory);
    cudaStreamSynchronize(stream_memory);
    timer.stop(stream_memory);

    milli += timer.elapsed();
    cudaMemGetInfo(&free_bytes, &total_bytes);

    /**************************** Free up memory ****************************/
    if (layer_type[i] == CONV) {
      cudaFree(this->workspace);  // free workspace
    }

    timer.start();
    for (int c = 0; c < free_layer.size(); c++) {
      cudaFree(layer_input[free_layer[c]]);  // free layer_input
    }

    free_layer.clear();  // clear free_layer
    timer.stop();
    milli += timer.elapsed();
    fwd_dnn_lag.push_back(milli);
    if (overhead != NULL) {
      *overhead += milli;
    }
    if (train == false && offloaded[i] == false) {
      cudaFree(layer_input[i]);
    }
    /**********************************************************************/
  }
  std::cout << "Forward Propagation Ends" << '\n';
}

void NeuralNet::conv_forward(int &i, float &alpha, float &beta) {
  ConvLayerParams *cur_params = (ConvLayerParams *)params[i];
  this->workspace_size = cur_params->fwd_workspace_size;
  cudaMalloc(&(this->workspace), cur_params->fwd_workspace_size);
  std::cout << "Allocated workspace of layer " << i << " Size: " << std::setprecision(2)
            << cur_params->fwd_workspace_size / (1024.0 * 1024.0) << " MB" << '\n';

  // Computation
  checkCUDNN(cudnnConvolutionForward(cudnn_handle, &alpha, cur_params->input_tensor, layer_input[i],
                                     cur_params->filter_desc, cur_params->W, cur_params->conv_desc,
                                     cur_params->fwd_algo, this->workspace, this->workspace_size,
                                     &beta, cur_params->output_tensor, layer_input[i + 1]));
  checkCUDNN(cudnnAddTensor(cudnn_handle, &alpha, cur_params->bias_desc, cur_params->b, &alpha,
                            cur_params->output_tensor, layer_input[i + 1]));
  // If activation required
  if (cur_params->activation_mode != ACTIVATION_NONE) {
    checkCUDNN(cudnnActivationForward(cudnn_handle, cur_params->actv_desc, &alpha,
                                      cur_params->output_tensor, layer_input[i + 1], &beta,
                                      cur_params->output_tensor, layer_input[i + 1]));
  }
}

void NeuralNet::fc_forward(int &i, float &alpha, float &beta, float &Salpha, float &Sbeta,
                           double &Dalpha, double &Dbeta) {
  FCLayerParams *cur_params = (FCLayerParams *)params[i];

  if (data_type == CUDNN_DATA_FLOAT) {
    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, cur_params->C_out, batch_size,
                cur_params->C_in, &Salpha, (float *)cur_params->W, cur_params->C_out,
                (float *)layer_input[i], cur_params->C_in, &Sbeta, (float *)layer_input[i + 1],
                cur_params->C_out);
    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, cur_params->C_out, batch_size, 1, &Salpha,
                (float *)cur_params->b, cur_params->C_out, (float *)one_vec, 1, &Salpha,
                (float *)layer_input[i + 1], cur_params->C_out);
  } else if (data_type == CUDNN_DATA_DOUBLE) {
    cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, cur_params->C_out, batch_size,
                cur_params->C_in, &Dalpha, (double *)cur_params->W, cur_params->C_out,
                (double *)layer_input[i], cur_params->C_in, &Dbeta, (double *)layer_input[i + 1],
                cur_params->C_out);
    cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, cur_params->C_out, batch_size, 1, &Dalpha,
                (double *)cur_params->b, cur_params->C_out, (double *)one_vec, 1, &Dalpha,
                (double *)layer_input[i + 1], cur_params->C_out);
  }
  if (cur_params->activation_mode != ACTIVATION_NONE) {
    checkCUDNN(cudnnActivationForward(cudnn_handle, cur_params->actv_desc, &alpha,
                                      cur_params->output_tensor, layer_input[i + 1], &beta,
                                      cur_params->output_tensor, layer_input[i + 1]));
  }
}

void NeuralNet::pooling_forward(int &i, float &alpha, float &beta) {
  PoolingLayerParams *cur_params = (PoolingLayerParams *)params[i];
  checkCUDNN(cudnnPoolingForward(cudnn_handle, cur_params->pool_desc, &alpha,
                                 cur_params->input_tensor, layer_input[i], &beta,
                                 cur_params->output_tensor, layer_input[i + 1]));
}

void NeuralNet::activation_forward(int &i, float &alpha, float &beta) {
  exit(0);
  ActivationLayerParams *cur_params = (ActivationLayerParams *)params[i];
  checkCUDNN(cudnnActivationForward(cudnn_handle, cur_params->actv_desc, &alpha,
                                    cur_params->input_tensor, layer_input[i], &beta,
                                    cur_params->input_tensor, layer_input[i + 1]));
}

void NeuralNet::softmax_forward(int &i, float &alpha, float &beta) {
  SoftmaxLayerParams *cur_params = (SoftmaxLayerParams *)params[i];
  checkCUDNN(cudnnSoftmaxForward(cudnn_handle, cur_params->algo, cur_params->mode, &alpha,
                                 cur_params->input_tensor, layer_input[i], &beta,
                                 cur_params->input_tensor, layer_input[i + 1]));
}