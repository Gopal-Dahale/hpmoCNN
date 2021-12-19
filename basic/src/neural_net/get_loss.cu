#include <cstdio>
#include <cublas_v2.h>
#include <cudnn.h>
#include <curand.h>
#include <string>
#include <time.h>

#include "neural_net.cuh"

template <typename T>
__global__ void softmaxLossBackProp(int *y, T *SO, T *dSO, int batch_size,
                                    int output_size, float eps)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= batch_size)
    return;
  int cur_class = static_cast<int>(y[i]);
  dSO[i * output_size + cur_class] =
      -1 / (SO[i * output_size + cur_class] * batch_size + eps);
}

void NeuralNet::getLoss(void *X, int *y, double learning_rate, bool train,
                        int *correct_count, float *loss, bool doo)
{
  std::vector<float> t1, t2;
  this->getLoss(X, y, learning_rate, t1, t2, train, correct_count, loss, doo);
}

void NeuralNet::getLoss(void *X, int *y, double learning_rate,
                        std::vector<float> &fwd_dnn_lag,
                        std::vector<float> &bwd_dnn_lag, bool train,
                        int *correct_count, float *scalar_loss, bool doo)
{
  cudaMalloc(&layer_input[0], layer_input_size[0] * data_type_size);
  cudaMemcpy(layer_input[0], X,
             batch_size * input_channels * input_h * input_w * data_type_size,
             cudaMemcpyHostToDevice);
  if (train == true)
    cudaMemcpy(this->y, y, batch_size * data_type_size, cudaMemcpyHostToDevice);

  float alpha = 1.0, beta = 0.0;
  float Salpha = 1.0, Sbeta = 0.0;
  double Dalpha = 1.0, Dbeta = 0.0;

  // Forward Propagation
//   std::cout << "Forward Propagation: "<< '\n';
  for (int i = 0; i < num_layers; i++)
  {
    std::vector<int> free_layer;
    if (train == false && i == num_layers - 1)
      break;
    cudaMalloc(&layer_input[i + 1], layer_input_size[i + 1] * data_type_size);
    if(i>0)
      layer_input_pq.push({layer_input_size[i], i});
    cudaMemGetInfo(&free_bytes, &total_bytes);
//     std::cout << "Before Offload and computation of current layer: " << free_bytes <<'\n';
    if(free_bytes - 1024 * 1024 * 1024 <= layer_input_size[i + 2] * data_type_size)
    {
      int temp_free_bytes = 0;
      while(temp_free_bytes - 1024 * 1024 * 1024 <= layer_input_size[i + 2] * data_type_size || layer_input_pq.empty()!=true)
      {
        int temp = layer_input_pq.top().first;
        free_layer.push_back(temp);
        temp_free_bytes += layer_input_pq.top().second * data_type_size;
        offloaded[temp] = true;
        cudaMemcpyAsync(h_layer_input[temp], layer_input[temp], layer_input_size[temp] * data_type_size, cudaMemcpyDeviceToHost, stream_memory);
        layer_input_pq.pop();
      }
    }
                            
       
//     if(i>1 && train == true && doo==true)
// //     {
// //       cudaMemGetInfo(&free_bytes, &total_bytes);
// //       std::cout << "Before Offload: " << free_bytes <<'\n'; 
// //       std::cout << "cudaMemPrefetchAsync: " <<
//     cudaMemPrefetchAsync(layer_input[i-1], layer_input_size[i-1]*data_type_size, cudaCpuDeviceId, stream_memory); //<< '\n';
// //     }
    

    if (layer_type[i] == CONV)
    {
      ConvLayerParams *cur_params = (ConvLayerParams *)params[i];

      // Computation
      checkCUDNN(cudnnConvolutionForward(
          cudnn_handle, &alpha, cur_params->input_tensor, layer_input[i],
          cur_params->filter_desc, cur_params->W, cur_params->conv_desc,
          cur_params->fwd_algo, this->workspace, this->workspace_size, &beta,
          cur_params->output_tensor, layer_input[i + 1]));
      checkCUDNN(cudnnAddTensor(cudnn_handle, &alpha, cur_params->bias_desc,
                                cur_params->b, &alpha,
                                cur_params->output_tensor, layer_input[i + 1]));

      // If activation required
      if (cur_params->activation_mode != ACTIVATION_NONE)
      {
        checkCUDNN(cudnnActivationForward(
            cudnn_handle, cur_params->actv_desc, &alpha,
            cur_params->output_tensor, layer_input[i + 1], &beta,
            cur_params->output_tensor, layer_input[i + 1]));
      }
    }

    else if (layer_type[i] == FULLY_CONNECTED)
    {
      FCLayerParams *cur_params = (FCLayerParams *)params[i];

      if (data_type == CUDNN_DATA_FLOAT)
      {
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, cur_params->C_out,
                    batch_size, cur_params->C_in, &Salpha,
                    (float *)cur_params->W, cur_params->C_out,
                    (float *)layer_input[i], cur_params->C_in, &Sbeta,
                    (float *)layer_input[i + 1], cur_params->C_out);
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, cur_params->C_out,
                    batch_size, 1, &Salpha, (float *)cur_params->b,
                    cur_params->C_out, (float *)one_vec, 1, &Salpha,
                    (float *)layer_input[i + 1], cur_params->C_out);
      }
      else if (data_type == CUDNN_DATA_DOUBLE)
      {
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
      if (cur_params->activation_mode != ACTIVATION_NONE)
      {
//         cudaMemGetInfo(&free_bytes, &total_bytes);
//         std::cout << "Before Offload: " << free_bytes <<'\n'; 
        checkCUDNN(cudnnActivationForward(
            cudnn_handle, cur_params->actv_desc, &alpha,
            cur_params->output_tensor, layer_input[i + 1], &beta,
            cur_params->output_tensor, layer_input[i + 1]));
      }
    }
    else if (layer_type[i] == POOLING)
    {
      PoolingLayerParams *cur_params = (PoolingLayerParams *)params[i];
      checkCUDNN(
          cudnnPoolingForward(cudnn_handle, cur_params->pool_desc, &alpha,
                              cur_params->input_tensor, layer_input[i], &beta,
                              cur_params->output_tensor, layer_input[i + 1]));
    }
    else if (layer_type[i] == ACTV)
    {
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
    if (layer_type[i + 1] == SOFTMAX)
    {
      i++;
      if (train == true)
      {
        layer_input[i + 1] = layer_input[i];
        SoftmaxLayerParams *cur_params = (SoftmaxLayerParams *)params[i];
        checkCUDNN(cudnnSoftmaxForward(
            cudnn_handle, cur_params->algo, cur_params->mode, &alpha,
            cur_params->input_tensor, layer_input[i], &beta,
            cur_params->input_tensor, layer_input[i + 1]));
      }
      // i--;
    }
    cudaStreamSynchronize(stream_compute);
    cudaStreamSynchronize(stream_memory);
    for(int c=0;c<free_layer.size();c++)
      cudaFree(layer_input[free_layer[c]]);
    cudaMemGetInfo(&free_bytes, &total_bytes);
//     std::cout << "After Offload and computation of current layer: " << free_bytes <<'\n';
  }

  // Accuracy Computation
  if (train == false)
  {
    compareOutputCorrect(correct_count, y);
//     cudaFree(layer_input[num_layers - 1]);
//     *scalar_loss = computeLoss(); // Loss Computation
    return;
  }
  *scalar_loss = computeLoss(); // Loss Computation
                           
  cudaMalloc(&dlayer_input[num_layers], batch_size * num_classes * data_type_size);                         

  // Backward Propagation
  if (layer_type[num_layers - 1] == SOFTMAX)
  {
    if (data_type == CUDNN_DATA_FLOAT)
    {
      cudaMemset(dlayer_input[num_layers], 0,
                 batch_size * num_classes * sizeof(float));
      softmaxLossBackProp<float><<<ceil(1.0 * batch_size / BW), BW>>>(
          this->y, (float *)layer_input[num_layers],
          (float *)dlayer_input[num_layers], batch_size, num_classes,
          softmax_eps);
    }
    else if (data_type == CUDNN_DATA_DOUBLE)
    {
      cudaMemset(dlayer_input[num_layers], 0,
                 batch_size * num_classes * sizeof(double));
      softmaxLossBackProp<double><<<ceil(1.0 * batch_size / BW), BW>>>(
          this->y, (double *)layer_input[num_layers],
          (double *)dlayer_input[num_layers], batch_size, num_classes,
          softmax_eps);
    }
  }

//   std::cout << "Backward Propagation: " << '\n';
  for (int i = num_layers - 1; i >= 0; i--)
  {
    if (i > 0)
    {
      if (layer_type[i] == ACTV or layer_type[i] == SOFTMAX)
      {
        dlayer_input[i] = dlayer_input[i + 1];
      }
      if(offloaded[i-1])
      {
        cudaMalloc(&layer_input[i-1], layer_input_size[i-1] * data_type_size);
        if (i-1 != 0)
	{
		cudaMemcpyAsync(layer_input[i-1], h_layer_input[i-1], layer_input_size[i-1] * data_type_size, cudaMemcpyHostToDevice, stream_memory);
	}
	else
	{
		cudaMemcpyAsync(layer_input[i-1], X, layer_input_size[i-1] * data_type_size, cudaMemcpyHostToDevice, stream_memory);
	}
      }
      cudaMalloc(&dlayer_input[i], layer_input_size[i] * data_type_size);
// //       else
// //       {
//             if(doo==true){
// //         cudaMemGetInfo(&free_bytes, &total_bytes);
// //         std::cout << "Before Prefetch: " << free_bytes <<'\n';
//         int device = -1;
//         cudaGetDevice(&device);
// //         std::cout << "cudaMemPrefetchAsync: " << 
//       cudaMemPrefetchAsync(layer_input[i-1],layer_input_size[i-1]*data_type_size,device,stream_memory);// <<'\n';
// // //         cudaMemGetInfo(&free_bytes, &total_bytes);
// // //         std::cout << "After Prefetch: "<< free_bytes <<'\n';
//             }
// //       }
    }
    if (layer_type[i] == CONV)
    {
      ConvLayerParams *cur_params = (ConvLayerParams *)params[i];

      if (cur_params->activation_mode != ACTIVATION_NONE)
      {
        checkCUDNN(cudnnActivationBackward(
            cudnn_handle, cur_params->actv_desc, &alpha,
            cur_params->output_tensor, layer_input[i + 1],
            cur_params->output_tensor, dlayer_input[i + 1],
            cur_params->output_tensor, layer_input[i + 1], &beta,
            cur_params->output_tensor, dlayer_input[i + 1]));
      }

      checkCUDNN(cudnnConvolutionBackwardBias(
          cudnn_handle, &alpha, cur_params->output_tensor, dlayer_input[i + 1],
          &beta, cur_params->bias_desc, cur_params->db));

      checkCUDNN(cudnnConvolutionBackwardFilter(
          cudnn_handle, &alpha, cur_params->input_tensor, layer_input[i],
          cur_params->output_tensor, dlayer_input[i + 1], cur_params->conv_desc,
          cur_params->bwd_filter_algo, this->workspace, this->workspace_size,
          &beta, cur_params->filter_desc, cur_params->dW));
      if (i > 0)
        checkCUDNN(cudnnConvolutionBackwardData(
            cudnn_handle, &alpha, cur_params->filter_desc, cur_params->W,
            cur_params->output_tensor, dlayer_input[i + 1],
            cur_params->conv_desc, cur_params->bwd_data_algo, this->workspace,
            workspace_size, &beta, cur_params->input_tensor, dlayer_input[i]));

      cur_params->stepParams(cublas_handle, learning_rate);
    }

    else if (layer_type[i] == FULLY_CONNECTED)
    {
      FCLayerParams *cur_params = (FCLayerParams *)params[i];

      if (cur_params->activation_mode != ACTIVATION_NONE)
      {
        checkCUDNN(cudnnActivationBackward(
            cudnn_handle, cur_params->actv_desc, &alpha,
            cur_params->output_tensor, layer_input[i + 1],
            cur_params->output_tensor, dlayer_input[i + 1],
            cur_params->output_tensor, layer_input[i + 1], &beta,
            cur_params->output_tensor, dlayer_input[i + 1]));
      }

      if (data_type == CUDNN_DATA_FLOAT)
      {
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

      else if (data_type == CUDNN_DATA_DOUBLE)
      {
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

    else if (layer_type[i] == POOLING)
    {
      PoolingLayerParams *cur_params = (PoolingLayerParams *)params[i];
      checkCUDNN(
          cudnnPoolingBackward(cudnn_handle, cur_params->pool_desc, &alpha,
                               cur_params->output_tensor, layer_input[i + 1],
                               cur_params->output_tensor, dlayer_input[i + 1],
                               cur_params->input_tensor, layer_input[i], &beta,
                               cur_params->input_tensor, dlayer_input[i]));
    }

    else if (layer_type[i] == ACTV)
    {
      ActivationLayerParams *cur_params = (ActivationLayerParams *)params[i];
      checkCUDNN(cudnnActivationBackward(
          cudnn_handle, cur_params->actv_desc, &alpha, cur_params->input_tensor,
          layer_input[i + 1], cur_params->input_tensor, dlayer_input[i + 1],
          cur_params->input_tensor, layer_input[i], &beta,
          cur_params->input_tensor, dlayer_input[i]));
      continue;
    }

    else if (layer_type[i] == SOFTMAX)
    {
      SoftmaxLayerParams *cur_params = (SoftmaxLayerParams *)params[i];
      checkCUDNN(cudnnSoftmaxBackward(
          cudnn_handle, cur_params->algo, cur_params->mode, &alpha,
          cur_params->input_tensor, layer_input[i + 1],
          cur_params->input_tensor, dlayer_input[i + 1], &beta,
          cur_params->input_tensor, dlayer_input[i]));
      continue;
    }
    cudaStreamSynchronize(stream_compute);

//     cudaMemGetInfo(&free_bytes, &total_bytes);
//     std::cout << "Before Offload (layer and dlayer): "<< free_bytes <<'\n';
//   if(doo==true){
//     std::cout << "cudaMemPrefetchAsync: " <<
//     cudaMemPrefetchAsync(layer_input[i+1], layer_input_size[i+1]*data_type_size, cudaCpuDeviceId, stream_memory); //<< '\n';
//     std::cout << "cudaMemPrefetchAsync: " <<
//     cudaMemPrefetchAsync(dlayer_input[i+1], layer_input_size[i+1]*data_type_size, cudaCpuDeviceId, stream_memory); //<< '\n';
//   }
    cudaStreamSynchronize(stream_memory);
    
    cudaFree(layer_input[i + 1]);
    cudaFree(dlayer_input[i + 1]);

    if(i==0)
      cudaFree(layer_input[i]);
    
//     cudaMemGetInfo(&free_bytes, &total_bytes);
//     std::cout << "freed up feature map and its derivative: " << free_bytes<<'\n';
  }
}
