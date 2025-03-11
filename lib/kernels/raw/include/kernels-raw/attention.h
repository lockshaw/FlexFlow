#ifndef _FLEXFLOW_LIB_KERNELS_RAW_INCLUDE_KERNELS_RAW_ATTENTION_H
#define _FLEXFLOW_LIB_KERNELS_RAW_INCLUDE_KERNELS_RAW_ATTENTION_H

#include "kernels-raw/device.h"
#include "kernels-raw/ffi_helpers.h"

FLEXFLOW_KERNELS_FFI_BEGIN();

void ff_attention_forward_kernel(ffStream_t stream,
                                 ffHandle_t dnn,
                                 ffAttnDescriptor_t attnDesc,
                                 int *loWinIdx,
                                 int *hiWinIdx,
                                 int *devQoSeqArray,
                                 int *devKvSeqArray,
                                 float *query_ptr,
                                 float *key_ptr,
                                 float *value_ptr,
                                 float *weight_ptr,
                                 float *output_ptr);

void ff_attention_backward_kernel(ffStream_t stream,
                                  ffHandle_t dnn,
                                  ffAttnDescriptor_t attnDesc,
                                 int *loWinIdx,
                                 int *hiWinIdx,
                                 int *devQoSeqArray,
                                 int *devKvSeqArray,
                                 float *query_ptr,
                                 float *query_grad_ptr,
                                 float *key_ptr,
                                 float *key_grad_ptr,
                                 float *value_ptr,
                                 float *value_grad_ptr,
                                 float *weight_ptr,
                                 float *weight_grad_ptr,
                                 float *output_grad_ptr);

FLEXFLOW_KERNELS_FFI_END();

#endif
