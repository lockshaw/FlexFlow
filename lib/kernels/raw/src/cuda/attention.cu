#include "kernels-raw/attention.h"

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
                                 float *output_ptr) {

}

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
                                 float *output_grad_ptr) {

}

