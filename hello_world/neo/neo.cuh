#pragma once
#include "./kernels/affine_transformer.cuh"

namespace affine
{
    void affine_transformer(int a, int *x, int *y, int *z, int n)
    {
        int *gpu_a = NULL;
        int *gpu_x = NULL;
        int *gpu_y = NULL;
        int *gpu_z = NULL;

        cudaMalloc(&gpu_a, sizeof(int));
        cudaMalloc(&gpu_x, sizeof(int) * n);
        cudaMalloc(&gpu_y, sizeof(int) * n);
        cudaMalloc(&gpu_z, sizeof(int) * n);

        cudaMemcpy(gpu_a, &a, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_x, x, sizeof(int) * n, cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_y, y, sizeof(int) * n, cudaMemcpyHostToDevice);

        affine_kernel<<<1, 32>>>(gpu_a, gpu_x, gpu_y, gpu_z, n);
        cudaDeviceSynchronize();

        cudaMemcpy(z, gpu_z, sizeof(int) * n, cudaMemcpyDeviceToHost);

        cudaFree(gpu_a);
        cudaFree(gpu_x);
        cudaFree(gpu_y);
        cudaFree(gpu_z);
    }
}