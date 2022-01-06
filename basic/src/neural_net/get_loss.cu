#include <cublas_v2.h>
#include <cudnn.h>
#include <curand.h>
#include <time.h>

#include <cstdio>
#include <string>

#include "neural_net.cuh"

template <typename T>
__global__ void softmaxLossBackProp(int *y, T *SO, T *dSO, int batch_size,
                                    int output_size, float eps) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= batch_size) return;
  int cur_class = static_cast<int>(y[i]);
  dSO[i * output_size + cur_class] =
      -1 / (SO[i * output_size + cur_class] * batch_size + eps);
}

void NeuralNet::getLoss(void *X, int *y, double learning_rate, bool train,
                        int *correct_count, float *loss, bool doo) {
  std::vector<float> t1, t2;
  this->getLoss(X, y, learning_rate, t1, t2, train, correct_count, loss, doo);
}

void NeuralNet::getLoss(void *X, int *y, double learning_rate,
                        std::vector<float> &fwd_dnn_lag,
                        std::vector<float> &bwd_dnn_lag, bool train,
                        int *correct_count, float *scalar_loss, bool doo) {
  cudaMemGetInfo(&free_bytes, &total_bytes);
  int bef0 = free_bytes;
  cudaMalloc(&layer_input[0], layer_input_size[0] * data_type_size);
  cudaMemGetInfo(&free_bytes, &total_bytes);
  int aft0 = free_bytes;
  std::cout << "Allocated to layer 0: " << (bef0 - aft0)
            << " free: " << free_bytes / (1024.0 * 1024.0 * 1024.0) << "\n";
  cudaMemcpy(layer_input[0], X,
             batch_size * input_channels * input_h * input_w * data_type_size,
             cudaMemcpyHostToDevice);
  if (train == true)
    cudaMemcpy(this->y, y, batch_size * data_type_size, cudaMemcpyHostToDevice);

  float alpha = 1.0, beta = 0.0;
  float Salpha = 1.0, Sbeta = 0.0;
  double Dalpha = 1.0, Dbeta = 0.0;

  // Display layer_input_size in bytes
  for (int c = 0; c < num_layers; c++)
    std::cout << "layer_input_size[" << c
              << "] = " << layer_input_size[c] * data_type_size << std::endl;

  /************************ Forward Propagation starts ***********************/
  std::cout << "Forward Propagation starts: " << '\n';
  size_t buffer_bytes = 1024 * 1024 * 1024;  // 2GB
  int ttl_allocated = 0;
  std::vector<int> free_layer;  // Which layers to free
  for (int i = 0; i < num_layers; i++) {
    if (train == false && i == num_layers - 1) break;

    cudaMemGetInfo(&free_bytes, &total_bytes);
    int bef = free_bytes;
    cudaMalloc(&layer_input[i + 1], layer_input_size[i + 1] * data_type_size);
    cudaMemGetInfo(&free_bytes, &total_bytes);
    int aft = free_bytes;
    ttl_allocated += (bef - aft);
    std::cout << "Allocated to layer " << i + 1 << ": " << (bef - aft)
              << " free: " << free_bytes / (1024.0 * 1024.0 * 1024.0) << "\n";

    // Push the layer_input_size + weights_size to the heap of ith layer
    if (i > 0) {
      layer_input_pq.push({layer_input_size[i], i});
      std::cout << "Layer inserted: " << i << "\n";
    }

    cudaMemGetInfo(&free_bytes, &total_bytes);
    std::cout << "Before Offload and computation of layer " << i << " : "
              << free_bytes / (1024.0 * 1024.0 * 1024.0) << '\n';

    size_t temp_free_bytes = free_bytes;                  // Current free bytes
    size_t free_memory = temp_free_bytes - buffer_bytes - buffer_bytes;  // Free memory made 2 GB reserved
    size_t layer_size =
        layer_input_size[i + 2] * data_type_size;  // Size of the layer

    // Decrement free_memory by i+2 th layer fwd workspace size.
    // This make sures that i+2 the layer can be allocated smoothly when
    // needed
    if ((i + 2 < num_layers) && (layer_type[i + 2] == CONV)) {
      ConvLayerParams *cur_params = (ConvLayerParams *)params[i + 2];
      layer_size += cur_params->fwd_workspace_size;
    }

    if ((i + 2 < num_layers) && (free_memory <= layer_size)) {
      std::cout << "GPU memory is low, offloading to CPU" << std::endl;
      std::cout << (free_bytes - buffer_bytes - buffer_bytes) / float(buffer_bytes) << " <= "
                << layer_input_size[i + 2] * data_type_size /
                       float(buffer_bytes)
                << '\n';

      /************* Heap logic with workspace fix ********************/

      bool cond1 = (free_memory <= layer_size);
      bool cond2 = (!layer_input_pq.empty());

      // Display cond1 and cond2
      std::cout << "cond1: " << cond1 << " cond2: " << cond2 << std::endl;

      // Display cond1 && cond2
      std::cout << "Condition: " << (cond1 && cond2) << std::endl;




      // While the free memory is less than or equal to the (i+2)th layer
      // input size or the heap is not empty
      while ((free_memory <= layer_size) && (!layer_input_pq.empty())) {
        int temp = layer_input_pq.top().second;  // Get the layer index on top
                                                 // of the heap
        std::cout << "Layer to offload: " << temp << std::endl;
        std::cout << "Size of the layer to offload: "
                  << layer_input_pq.top().first * data_type_size /
                         float(buffer_bytes)
                  << std::endl;
        free_layer.push_back(temp);  // Add the layer index to the free layer
                                     // vector

        // Update the free bytes
        temp_free_bytes += layer_input_pq.top().first * data_type_size;
        std::cout << "Free gigabytes in GPU: "
                  << temp_free_bytes / float(buffer_bytes) << std::endl;
        offloaded[temp] = true;  // Mark the layer as offloaded

        // Copy the layer to host
        // Only offload layer_input and not workspace
        cudaMemcpyAsync(h_layer_input[temp], layer_input[temp],
                        layer_input_size[temp] * data_type_size,
                        cudaMemcpyDeviceToHost, stream_memory);
        layer_input_pq.pop();  // Remove the layer from the heap
        std::cout << "New Top: " << layer_input_pq.top().second << "\n";
        free_memory = temp_free_bytes - buffer_bytes - buffer_bytes;
      }
      /*************************************************************/
    }

    //     if(i>1 && train == true && doo==true)
    // //     {
    // //       cudaMemGetInfo(&free_bytes, &total_bytes);
    // //       std::cout << "Before Offload: " << free_bytes <<'\n';
    // //       std::cout << "cudaMemPrefetchAsync: " <<
    //     cudaMemPrefetchAsync(layer_input[i-1],
    //     layer_input_size[i-1]*data_type_size, cudaCpuDeviceId,
    //     stream_memory); //<< '\n';
    // //     }
    cudaMemGetInfo(&free_bytes, &total_bytes);
    std::cout << "Before Computation of Layer " << i << ": "
              << free_bytes / (1024.0 * 1024.0 * 1024.0) << "\n";
    if (layer_type[i] == CONV) {
      ConvLayerParams *cur_params = (ConvLayerParams *)params[i];

      this->workspace_size = cur_params->fwd_workspace_size;

      cudaMalloc(&(this->workspace), cur_params->fwd_workspace_size);

      // Computation
      checkCUDNN(cudnnConvolutionForward(
          cudnn_handle, &alpha, cur_params->input_tensor, layer_input[i],
          cur_params->filter_desc, cur_params->W, cur_params->conv_desc,
          cur_params->fwd_algo, this->workspace, this->workspace_size, &beta,
          cur_params->output_tensor, layer_input[i + 1]));
      cudaMemGetInfo(&free_bytes, &total_bytes);
      std::cout << "After cudnnConvolutionForward " << i << ": "
                << free_bytes / (1024.0 * 1024.0 * 1024.0) << "\n";
      checkCUDNN(cudnnAddTensor(cudnn_handle, &alpha, cur_params->bias_desc,
                                cur_params->b, &alpha,
                                cur_params->output_tensor, layer_input[i + 1]));
      cudaMemGetInfo(&free_bytes, &total_bytes);
      std::cout << "After cudnnAddTensor " << i << ": "
                << free_bytes / (1024.0 * 1024.0 * 1024.0) << "\n";
      // If activation required
      if (cur_params->activation_mode != ACTIVATION_NONE) {
        checkCUDNN(cudnnActivationForward(
            cudnn_handle, cur_params->actv_desc, &alpha,
            cur_params->output_tensor, layer_input[i + 1], &beta,
            cur_params->output_tensor, layer_input[i + 1]));
      }
      cudaMemGetInfo(&free_bytes, &total_bytes);
      std::cout << "After cudnnActivationForward " << i << ": "
                << free_bytes / (1024.0 * 1024.0 * 1024.0) << "\n";
    }

    else if (layer_type[i] == FULLY_CONNECTED) {
      FCLayerParams *cur_params = (FCLayerParams *)params[i];

      if (data_type == CUDNN_DATA_FLOAT) {
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, cur_params->C_out,
                    batch_size, cur_params->C_in, &Salpha,
                    (float *)cur_params->W, cur_params->C_out,
                    (float *)layer_input[i], cur_params->C_in, &Sbeta,
                    (float *)layer_input[i + 1], cur_params->C_out);
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, cur_params->C_out,
                    batch_size, 1, &Salpha, (float *)cur_params->b,
                    cur_params->C_out, (float *)one_vec, 1, &Salpha,
                    (float *)layer_input[i + 1], cur_params->C_out);
      } else if (data_type == CUDNN_DATA_DOUBLE) {
        cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, cur_params->C_out,
                    batch_size, cur_params->C_in, &Dalpha,
                    (double *)cur_params->W, cur_params->C_out,
                    (double *)layer_input[i], cur_params->C_in, &Dbeta,
                    (double *)layer_input[i + 1], cur_params->C_out);
        cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, cur_params->C_out,
                    batch_size, 1, &Dalpha, (double *)cur_params->b,
                    cur_params->C_out, (double *)one_vec, 1, &Dalpha,
                    (double *)layer_input[i + 1], cur_params->C_out);
      }
      if (cur_params->activation_mode != ACTIVATION_NONE) {
        //         cudaMemGetInfo(&free_bytes, &total_bytes);
        //         std::cout << "Before Offload: " << free_bytes <<'\n';
        checkCUDNN(cudnnActivationForward(
            cudnn_handle, cur_params->actv_desc, &alpha,
            cur_params->output_tensor, layer_input[i + 1], &beta,
            cur_params->output_tensor, layer_input[i + 1]));
      }
    } else if (layer_type[i] == POOLING) {
      PoolingLayerParams *cur_params = (PoolingLayerParams *)params[i];
      checkCUDNN(
          cudnnPoolingForward(cudnn_handle, cur_params->pool_desc, &alpha,
                              cur_params->input_tensor, layer_input[i], &beta,
                              cur_params->output_tensor, layer_input[i + 1]));
    } else if (layer_type[i] == ACTV) {
      std::cout << "Actv Layer\n";
      std::cout << "Panic!! ACTV wrong place\n";
      exit(0);
      ActivationLayerParams *cur_params = (ActivationLayerParams *)params[i];
      checkCUDNN(cudnnActivationForward(
          cudnn_handle, cur_params->actv_desc, &alpha, cur_params->input_tensor,
          layer_input[i], &beta, cur_params->input_tensor, layer_input[i + 1]));
    }
    // else if (layer_type[i] == SOFTMAX)
    // {
    //   // std::cout << "Softmax\n";
    //   //   std::cout << "Panic!! SOFTMAX wrong place\n";
    //   //   exit(0);
    //   if (train == true)
    //   {
    //     SoftmaxLayerParams *cur_params = (SoftmaxLayerParams *)params[i];
    //     checkCUDNN(cudnnSoftmaxForward(
    //         cudnn_handle, cur_params->algo, cur_params->mode, &alpha,
    //         cur_params->input_tensor, layer_input[i], &beta,
    //         cur_params->input_tensor, layer_input[i + 1]));
    //   }
    // }
    // synchronization
    // cudaDeviceSynchronize();

    // if next layer is ACTV or SOFTMAX, complete that and come to
    // synchronization the case in above if for ACTV and SOFTMAX never occurs
    if (layer_type[i + 1] == SOFTMAX) {
      i++;
      if (train == true) {
        layer_input[i + 1] = layer_input[i];
        SoftmaxLayerParams *cur_params = (SoftmaxLayerParams *)params[i];
        checkCUDNN(cudnnSoftmaxForward(
            cudnn_handle, cur_params->algo, cur_params->mode, &alpha,
            cur_params->input_tensor, layer_input[i], &beta,
            cur_params->input_tensor, layer_input[i + 1]));
      }
      // i--;
    }
    cudaMemGetInfo(&free_bytes, &total_bytes);
    std::cout << "Before Synchronization " << i << ": "
              << free_bytes / (1024.0 * 1024.0 * 1024.0) << "\n";
    cudaStreamSynchronize(stream_compute);
    cudaMemGetInfo(&free_bytes, &total_bytes);
    std::cout << "After Synchronization " << i << ": "
              << free_bytes / (1024.0 * 1024.0 * 1024.0) << "\n";
    cudaStreamSynchronize(stream_memory);
    cudaMemGetInfo(&free_bytes, &total_bytes);
    std::cout << "After Computation of Layer " << i << ": "
              << free_bytes / (1024.0 * 1024.0 * 1024.0) << "\n";

    /**************************** Free up memory ****************************/
    if (layer_type[i] == CONV) cudaFree(this->workspace);  // free workspace

    for (int c = 0; c < free_layer.size(); c++)
      cudaFree(layer_input[free_layer[c]]);  // free layer_input

    if (train == false && offloaded[i] == false) cudaFree(layer_input[i]);
    free_layer.clear();  // clear free_layer
    /**********************************************************************/

    cudaMemGetInfo(&free_bytes, &total_bytes);

    std::cout << "After Offload and computation of layer " << i << " : "
              << free_bytes / (1024.0 * 1024.0 * 1024.0) << '\n';
  }
  std::cout << "Forward Propagation ends: " << '\n';
  /************************ Forward Propagation ends ***********************/

  /************************ Offloaded layers Displayed ***********************/
  int flag = false;
  for (int c = 0; c < num_layers; c++)
    if (offloaded[c]) {
      flag = true;
      break;
    }
  if (flag) {
    std::cout << "\nOffloaded Layers: ";
    for (int c = 0; c < num_layers; c++)
      if (offloaded[c]) std::cout << c << " ";
  } else
    std::cout << "\nNo Offloaded Layers: ";
  std::cout << '\n';

  // Empty the priority queue
  while (!layer_input_pq.empty())
    layer_input_pq.pop();
  /***************************************************************************/

  /************************** Accuracy Computation **************************/
  if (train == false) {
    compareOutputCorrect(correct_count, y);
    //     cudaFree(layer_input[num_layers - 1]);
    //     *scalar_loss = computeLoss(); // Loss Computation
    return;
  }
  /***************************************************************************/
  *scalar_loss = computeLoss();  // Loss Computation

  cudaMemGetInfo(&free_bytes, &total_bytes);
  int bef1 = free_bytes;
  cudaMalloc(&dlayer_input[num_layers],
             batch_size * num_classes * data_type_size);
  cudaMemGetInfo(&free_bytes, &total_bytes);
  int aft1 = free_bytes;
  std::cout << "Allocated to dlayer " << num_layers << ": " << (bef1 - aft1)
            << " free: " << free_bytes / (1024.0 * 1024.0 * 1024.0) << "\n";

  if (layer_type[num_layers - 1] == SOFTMAX) {
    if (data_type == CUDNN_DATA_FLOAT) {
      cudaMemset(dlayer_input[num_layers], 0,
                 batch_size * num_classes * sizeof(float));
      softmaxLossBackProp<float><<<ceil(1.0 * batch_size / BW), BW>>>(
          this->y, (float *)layer_input[num_layers],
          (float *)dlayer_input[num_layers], batch_size, num_classes,
          softmax_eps);
    } else if (data_type == CUDNN_DATA_DOUBLE) {
      cudaMemset(dlayer_input[num_layers], 0,
                 batch_size * num_classes * sizeof(double));
      softmaxLossBackProp<double><<<ceil(1.0 * batch_size / BW), BW>>>(
          this->y, (double *)layer_input[num_layers],
          (double *)dlayer_input[num_layers], batch_size, num_classes,
          softmax_eps);
    }
  }

  /************************ Backward Propagation starts ***********************/
  std::cout << "Backward Propagation starts: " << '\n';
  for (int i = num_layers - 1; i >= 0; i--) {
    if (i > 0) {
      if (layer_type[i] == ACTV || layer_type[i] == SOFTMAX)
        dlayer_input[i] = dlayer_input[i + 1];

      // Prefetching
      if (offloaded[i - 1]) {
        std::cout << "Prefetching layer " << i - 1 << "\n";
        cudaMalloc(&layer_input[i - 1],
                   layer_input_size[i - 1] * data_type_size);
        if (i - 1 != 0) {
          cudaMemcpyAsync(layer_input[i - 1], h_layer_input[i - 1],
                          layer_input_size[i - 1] * data_type_size,
                          cudaMemcpyHostToDevice, stream_memory);
        } else {
          cudaMemcpyAsync(layer_input[i - 1], X,
                          layer_input_size[i - 1] * data_type_size,
                          cudaMemcpyHostToDevice, stream_memory);
        }
      }

      cudaMemGetInfo(&free_bytes, &total_bytes);
      int bef2 = free_bytes;
      cudaMalloc(&dlayer_input[i], layer_input_size[i] * data_type_size);
      cudaMemGetInfo(&free_bytes, &total_bytes);
      int aft2 = free_bytes;
      std::cout << "Allocated to dlayer " << i << ": " << (bef2 - aft2)
                << " free: " << free_bytes / (1024.0 * 1024.0 * 1024.0) << "\n";
    }
    cudaMemGetInfo(&free_bytes, &total_bytes);
    std::cout << "BP Before Derivative of Layer " << i << ": "
              << free_bytes / (1024.0 * 1024.0 * 1024.0) << "\n";
    if (layer_type[i] == CONV) {
      ConvLayerParams *cur_params = (ConvLayerParams *)params[i];

      if (cur_params->activation_mode != ACTIVATION_NONE) {
        checkCUDNN(cudnnActivationBackward(
            cudnn_handle, cur_params->actv_desc, &alpha,
            cur_params->output_tensor, layer_input[i + 1],
            cur_params->output_tensor, dlayer_input[i + 1],
            cur_params->output_tensor, layer_input[i + 1], &beta,
            cur_params->output_tensor, dlayer_input[i + 1]));
      }

      cudaMemGetInfo(&free_bytes, &total_bytes);
      std::cout << "After cudnnActivationBackward " << i << ": "
                << free_bytes / (1024.0 * 1024.0 * 1024.0) << "\n";

      size_t temp_data_wksp;

      if (i == 0)
        temp_data_wksp = 0;
      else
        temp_data_wksp = cur_params->bwd_data_workspace_size;

      this->workspace_size =
          max(cur_params->bwd_filter_workspace_size, temp_data_wksp);

      if (i == 1) std::cout << this->workspace_size << "\n";

      std::cout << cudaMalloc(&(this->workspace), this->workspace_size) << "\n";

      checkCUDNN(cudnnConvolutionBackwardBias(
          cudnn_handle, &alpha, cur_params->output_tensor, dlayer_input[i + 1],
          &beta, cur_params->bias_desc, cur_params->db));

      cudaMemGetInfo(&free_bytes, &total_bytes);
      std::cout << "After cudnnConvolutionBackwardBias " << i << ": "
                << free_bytes / (1024.0 * 1024.0 * 1024.0) << "\n";
      if (this->workspace == NULL) std::cout << "workspace problem\n";

      checkCUDNN(cudnnConvolutionBackwardFilter(
          cudnn_handle, &alpha, cur_params->input_tensor, layer_input[i],
          cur_params->output_tensor, dlayer_input[i + 1], cur_params->conv_desc,
          cur_params->bwd_filter_algo, this->workspace, this->workspace_size,
          &beta, cur_params->filter_desc, cur_params->dW));

      cudaMemGetInfo(&free_bytes, &total_bytes);
      std::cout << "After cudnnConvolutionBackwardFilter " << i << ": "
                << free_bytes / (1024.0 * 1024.0 * 1024.0) << "\n";

      if (i > 0)
        checkCUDNN(cudnnConvolutionBackwardData(
            cudnn_handle, &alpha, cur_params->filter_desc, cur_params->W,
            cur_params->output_tensor, dlayer_input[i + 1],
            cur_params->conv_desc, cur_params->bwd_data_algo, this->workspace,
            workspace_size, &beta, cur_params->input_tensor, dlayer_input[i]));

      cudaMemGetInfo(&free_bytes, &total_bytes);
      std::cout << "After cudnnConvolutionBackwardData " << i << ": "
                << free_bytes / (1024.0 * 1024.0 * 1024.0) << "\n";

      cur_params->stepParams(cublas_handle, learning_rate);
    }

    else if (layer_type[i] == FULLY_CONNECTED) {
      FCLayerParams *cur_params = (FCLayerParams *)params[i];

      if (cur_params->activation_mode != ACTIVATION_NONE) {
        checkCUDNN(cudnnActivationBackward(
            cudnn_handle, cur_params->actv_desc, &alpha,
            cur_params->output_tensor, layer_input[i + 1],
            cur_params->output_tensor, dlayer_input[i + 1],
            cur_params->output_tensor, layer_input[i + 1], &beta,
            cur_params->output_tensor, dlayer_input[i + 1]));
      }

      if (data_type == CUDNN_DATA_FLOAT) {
        // Bias backward
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, cur_params->C_out,
                    1, batch_size, &Salpha, (float *)dlayer_input[i + 1],
                    cur_params->C_out, (float *)one_vec, batch_size, &Sbeta,
                    (float *)cur_params->db, cur_params->C_out);

        // Weight backward
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, cur_params->C_out,
                    cur_params->C_in, batch_size, &Salpha,
                    (float *)dlayer_input[i + 1], cur_params->C_out,
                    (float *)layer_input[i], cur_params->C_in, &Sbeta,
                    (float *)cur_params->dW, cur_params->C_out);

        // Data backward
        if (i > 0)
          cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, cur_params->C_in,
                      batch_size, cur_params->C_out, &Salpha,
                      (float *)cur_params->W, cur_params->C_out,
                      (float *)dlayer_input[i + 1], cur_params->C_out, &Sbeta,
                      (float *)dlayer_input[i], cur_params->C_in);
      }

      else if (data_type == CUDNN_DATA_DOUBLE) {
        // Bias backward
        cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, cur_params->C_out,
                    1, batch_size, &Dalpha, (double *)dlayer_input[i + 1],
                    cur_params->C_out, (double *)one_vec, batch_size, &Dbeta,
                    (double *)cur_params->db, cur_params->C_out);

        // Weight backward
        cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, cur_params->C_out,
                    cur_params->C_in, batch_size, &Dalpha,
                    (double *)dlayer_input[i + 1], cur_params->C_out,
                    (double *)layer_input[i], cur_params->C_in, &Dbeta,
                    (double *)cur_params->dW, cur_params->C_out);

        // Data backward
        if (i > 0)
          cublasDgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, cur_params->C_in,
                      batch_size, cur_params->C_out, &Dalpha,
                      (double *)cur_params->W, cur_params->C_out,
                      (double *)dlayer_input[i + 1], cur_params->C_out, &Dbeta,
                      (double *)dlayer_input[i], cur_params->C_in);
      }
      cur_params->stepParams(cublas_handle, learning_rate);
    }

    else if (layer_type[i] == POOLING) {
      PoolingLayerParams *cur_params = (PoolingLayerParams *)params[i];
      checkCUDNN(
          cudnnPoolingBackward(cudnn_handle, cur_params->pool_desc, &alpha,
                               cur_params->output_tensor, layer_input[i + 1],
                               cur_params->output_tensor, dlayer_input[i + 1],
                               cur_params->input_tensor, layer_input[i], &beta,
                               cur_params->input_tensor, dlayer_input[i]));
    }

    else if (layer_type[i] == ACTV) {
      ActivationLayerParams *cur_params = (ActivationLayerParams *)params[i];
      checkCUDNN(cudnnActivationBackward(
          cudnn_handle, cur_params->actv_desc, &alpha, cur_params->input_tensor,
          layer_input[i + 1], cur_params->input_tensor, dlayer_input[i + 1],
          cur_params->input_tensor, layer_input[i], &beta,
          cur_params->input_tensor, dlayer_input[i]));
      continue;
    }

    else if (layer_type[i] == SOFTMAX) {
      SoftmaxLayerParams *cur_params = (SoftmaxLayerParams *)params[i];
      checkCUDNN(cudnnSoftmaxBackward(
          cudnn_handle, cur_params->algo, cur_params->mode, &alpha,
          cur_params->input_tensor, layer_input[i + 1],
          cur_params->input_tensor, dlayer_input[i + 1], &beta,
          cur_params->input_tensor, dlayer_input[i]));
      continue;
    }

    cudaMemGetInfo(&free_bytes, &total_bytes);
    std::cout << "Before Synchronization " << i << ": "
              << free_bytes / (1024.0 * 1024.0 * 1024.0) << "\n";
    cudaStreamSynchronize(stream_compute);

    cudaMemGetInfo(&free_bytes, &total_bytes);
    std::cout << "After Synchronization " << i << ": "
              << free_bytes / (1024.0 * 1024.0 * 1024.0) << "\n";
    cudaStreamSynchronize(stream_memory);

    if (layer_type[i] == CONV) cudaFree(this->workspace);

    cudaMemGetInfo(&free_bytes, &total_bytes);
    std::cout << "BP After Derivative of Layer " << i << ": "
              << free_bytes / (1024.0 * 1024.0 * 1024.0) << "\n";

    cudaMemGetInfo(&free_bytes, &total_bytes);
    int bef3 = free_bytes;
    cudaFree(layer_input[i + 1]);
    cudaMemGetInfo(&free_bytes, &total_bytes);
    int aft3 = free_bytes;
    std::cout << "freed to layer " << i + 1 << ": " << (aft3 - bef3)
              << " free: " << free_bytes / (1024.0 * 1024.0 * 1024.0) << "\n";

    cudaMemGetInfo(&free_bytes, &total_bytes);
    int bef4 = free_bytes;
    cudaFree(dlayer_input[i + 1]);
    cudaMemGetInfo(&free_bytes, &total_bytes);
    int aft4 = free_bytes;
    std::cout << "freed to dlayer " << i + 1 << ": " << (aft4 - bef4)
              << " free: " << free_bytes / (1024.0 * 1024.0 * 1024.0) << "\n";

    if (i == 0) {
      cudaFree(layer_input[i]);
      cudaMemGetInfo(&free_bytes, &total_bytes);
      int aft5 = free_bytes;
      std::cout << "freed to layer " << i << ": " << (aft5 - aft4)
                << " free: " << free_bytes / (1024.0 * 1024.0 * 1024.0) << "\n";
      cudaFree(dlayer_input[i]);
      cudaMemGetInfo(&free_bytes, &total_bytes);
      int aft6 = free_bytes;
      std::cout << "freed to layer " << i << ": " << (aft6 - aft5)
                << " free: " << free_bytes / (1024.0 * 1024.0 * 1024.0) << "\n";
    }

    cudaMemGetInfo(&free_bytes, &total_bytes);
    std::cout << "freed up feature map and its derivative after layer " << i
              << " of BP: " << free_bytes / (1024.0 * 1024.0 * 1024.0) << '\n';
  }
  std::cout << "Backward Propagation ends: " << '\n';
  /************************ Backward Propagation ends ***********************/

  cudaMemGetInfo(&free_bytes, &total_bytes);
  std::cout << "free mem before final free: "
            << free_bytes / (1024.0 * 1024.0 * 1024.0) << '\n';
  for (int k = 0; k < num_layers; k++) {
    if (layer_input[k] != NULL) cudaFree(layer_input[k]);
    if (dlayer_input[k] != NULL) cudaFree(dlayer_input[k]);
  }
  cudaMemGetInfo(&free_bytes, &total_bytes);
  std::cout << "free mem after 1FP1BP: "
            << free_bytes / (1024.0 * 1024.0 * 1024.0) << '\n';

  // Make offloaded array to all false
  for (int c = 0; c < num_layers; c++) offloaded[c] = false;
}
