#include <iostream>

#include "test.h"

// Kernel definition
__global__ void VecAdd(float* A, float* B, float* C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

void performTest(float* a, float* b, float* res, int size){
    VecAdd<<<1, size>>>(a, b, res);
}