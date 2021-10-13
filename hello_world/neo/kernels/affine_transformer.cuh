#pragma once

#include "../device/affine_base.cuh"

namespace affine
{
    __global__ void affine_kernel(int *a, int *x, int *y, int *z, int n)
    {
        int i = threadIdx.x;
        int num = *a;
        if (i < n)
            z[i] = affine_sum(x[i], y[i], num);
    }
}