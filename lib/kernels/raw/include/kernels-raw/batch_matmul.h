#ifndef _FLEXFLOW_OPS_KERNELS_BATCH_MATMUL_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_BATCH_MATMUL_KERNELS_H

#include "kernels-raw/device.h"

#ifdef __cplusplus
extern "C" {
#endif

void ff_batch_matmul_forward_kernel(ffStream_t stream,
                    ffHandle_t dnn_handle,
                    ffblasHandle_t blas,
                    float *output_ptr,
                    float const *a_input_ptr,
                    float const *b_input_ptr,
                    int m,
                    int n,
                    int k,
                    int batch,
                    int seq_length,
                    int a_seq_length_dim,
                    int b_seq_length_dim);

void ff_batch_matmul_backward_kernel(ffStream_t stream,
                     ffHandle_t dnn_handle,
                     ffblasHandle_t blas,
                     float const *o_ptr,
                     float const *o_grad_ptr,
                     float const *a_ptr,
                     float *a_grad_ptr,
                     float const *b_ptr,
                     float *b_grad_ptr,
                     int m,
                     int n,
                     int k,
                     int batch);

#ifdef __cplusplus
}
#endif

#endif
