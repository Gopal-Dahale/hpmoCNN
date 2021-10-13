#pragma once

namespace affine
{
    __device__ int affine_sum(int x, int y, int num)
    {
        int res = num * x + y;
        return res;
    }
}