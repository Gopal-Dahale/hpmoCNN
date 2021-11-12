#include <cstdio>
#include <cublas_v2.h>
#include <cudnn.h>
#include <curand.h>
#include <iostream>
#include <string>
#include <time.h>

#include "neural_net.cuh"

using namespace std;

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
                        int *correct_count, float *loss)
{
  std::vector<float> t1, t2;
  this->getLoss(X, y, learning_rate, t1, t2, train, correct_count, loss);
}

void NeuralNet::getLoss(void *X, int *y, double learning_rate,
                        std::vector<float> &fwd_dnn_lag,
                        std::vector<float> &bwd_dnn_lag, bool train,
                        int *correct_count, float *scalar_loss)
{
  cudaMemcpy(layer_input[0], X,
             batch_size * input_channels * input_h * input_w * data_type_size,
             cudaMemcpyHostToDevice);
  if (train == true)
    cudaMemcpy(this->y, y, batch_size * data_type_size, cudaMemcpyHostToDevice);

  float alpha = 1.0, beta = 0.0;
  float Salpha = 1.0, Sbeta = 0.0;
  double Dalpha = 1.0, Dbeta = 0.0;

  // Forward Propagation
  cout << "Forward Propagation: " << endl;
  cout << "Num Layers: " << num_layers << endl;
  cout << "Train: " << train << endl;
  for (int i = 0; i < num_layers; i++)
  {
    if (train == false && i == num_layers - 1)
      break;

    if (layer_type[i] == CONV)
    {
      cout << "CONV Layer\n";
      ConvLayerParams *cur_params = (ConvLayerParams *)params[i];

      // Computation
      checkCUDNN(cudnnConvolutionForward(
          cudnn_handle, &alpha, cur_params->input_tensor, layer_input[i],
          cur_params->filter_desc, cur_params->W, cur_params->conv_desc,
          cur_params->fwd_algo, workspace, workspace_size, &beta,
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
      cout << "FC Layer\n";
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
        checkCUDNN(cudnnActivationForward(
            cudnn_handle, cur_params->actv_desc, &alpha,
            cur_params->output_tensor, layer_input[i + 1], &beta,
            cur_params->output_tensor, layer_input[i + 1]));
      }
    }
    else if (layer_type[i] == POOLING)
    {
      cout << "Pooling Layer\n";
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
      cout << "Softmax Layer\n";
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
  }

  // Accuracy Computation
  if (train == false)
  {
    compareOutputCorrect(correct_count, y);
    return;
  }
  *scalar_loss = computeLoss(); // Loss Computation

  // Backward Propagation
  if (layer_type[num_layers - 1] == SOFTMAX)
  {
    cout << "Softmax Layer Backward\n";
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
  for (int i = num_layers - 1; i >= 0; i--)
  {
    if (layer_type[i] == CONV)
    {
      cout << "CONV Layer Backward\n";
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
          cur_params->bwd_filter_algo, workspace, workspace_size, &beta,
          cur_params->filter_desc, cur_params->dW));
      if (i > 0)
        checkCUDNN(cudnnConvolutionBackwardData(
            cudnn_handle, &alpha, cur_params->filter_desc, cur_params->W,
            cur_params->output_tensor, dlayer_input[i + 1],
            cur_params->conv_desc, cur_params->bwd_data_algo, workspace,
            workspace_size, &beta, cur_params->input_tensor, dlayer_input[i]));

      cur_params->stepParams(cublas_handle, learning_rate);
    }

    else if (layer_type[i] == FULLY_CONNECTED)
    {
      cout << "FC Layer Backward\n";
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
      cout << "Pooling Layer Backward\n";
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
      cout << "Actv Layer Backward\n";
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
      cout << "Softmax Layer Backward\n";
      SoftmaxLayerParams *cur_params = (SoftmaxLayerParams *)params[i];
      checkCUDNN(cudnnSoftmaxBackward(
          cudnn_handle, cur_params->algo, cur_params->mode, &alpha,
          cur_params->input_tensor, layer_input[i + 1],
          cur_params->input_tensor, dlayer_input[i + 1], &beta,
          cur_params->input_tensor, dlayer_input[i]));
      continue;
    }
    cudaStreamSynchronize(stream_compute);
  }
}