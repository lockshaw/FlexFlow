#ifndef _FLEXFLOW_LINEAR_UTILS_H
#define _FLEXFLOW_LINEAR_UTILS_H

__global__ void sigmoid_backward(float *grad_ptr, const float *output, int n);
__global__ void reluBackward(float* grad_ptr, const float* input, int n);

#endif // _FLEXFLOW_LINEAR_UTILS_H
