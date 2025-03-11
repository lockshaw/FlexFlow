/* Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "kernels-raw/batch_matmul.h"
#include <cassert>

void forward_kernel(cudaStream_t stream,
                    ffHandle_t dnn,
                    ffblasHandle_t blas,
                    float *output_ptr,
                    float const *a_input_ptr,
                    float const *b_input_ptr,
                    int m,
                    int n,
                    int k,
                    int batch,
                    int a_seq_length_dim,
                    int b_seq_length_dim,
                    int seq_length) {
  checkCUBLAS(cublasSetStream(blas, stream));
  checkCUDNN(cudnnSetStream(dnn, stream));
  int lda = k;
  int ldb = m;
  int ldo = m;
  long long int strideA = (long long int)n * k;
  long long int strideB = (long long int)k * m;
  long long int strideO = (long long int)n * m;
  if ((a_seq_length_dim == 0) && (seq_length >= 0)) {
    assert(seq_length <= k);
    k = seq_length;
    assert(b_seq_length_dim == 1);
  } else if ((a_seq_length_dim == 1) && (seq_length >= 0)) {
    assert(seq_length <= n);
    n = seq_length;
  } else {
    // currently only support a_seq_length_dim = 0 or 1
    assert((a_seq_length_dim < 0) || (seq_length < 0));
  }
  if ((b_seq_length_dim == 0) && (seq_length >= 0)) {
    assert(seq_length <= m);
    m = seq_length;
  } else if ((b_seq_length_dim == 1) && (seq_length >= 0)) {
    assert(a_seq_length_dim == 0);
    assert(k == seq_length);
  } else {
    // currently only support a_seq_length_dim = 0 or 1
    assert((b_seq_length_dim < 0) || (seq_length < 0));
  }

  float alpha = 1.0f, beta = 0.0f;
  checkCUBLAS(cublasSgemmStridedBatched(blas,
                                        CUBLAS_OP_N,
                                        CUBLAS_OP_N,
                                        m,
                                        n,
                                        k,
                                        &alpha,
                                        b_input_ptr,
                                        ldb,
                                        strideB,
                                        a_input_ptr,
                                        lda,
                                        strideA,
                                        &beta,
                                        output_ptr,
                                        ldo,
                                        strideO,
                                        batch));
}

void backward_kernel(cudaStream_t stream,
                     ffHandle_t dnn,
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
                     int batch) {
  checkCUBLAS(cublasSetStream(blas, stream));
  checkCUDNN(cudnnSetStream(dnn, stream));

  int a_stride = n * k;
  int b_stride = m * k;
  int o_stride = n * m;
  float alpha = 1.0f;
  checkCUBLAS(cublasSgemmStridedBatched(blas,
                                        CUBLAS_OP_T,
                                        CUBLAS_OP_N,
                                        k,
                                        n,
                                        m,
                                        &alpha,
                                        b_ptr,
                                        m,
                                        b_stride,
                                        o_grad_ptr,
                                        m,
                                        o_stride,
                                        &alpha,
                                        a_grad_ptr,
                                        k,
                                        a_stride,
                                        batch));
  checkCUBLAS(cublasSgemmStridedBatched(blas,
                                        CUBLAS_OP_N,
                                        CUBLAS_OP_T,
                                        m,
                                        k,
                                        n,
                                        &alpha,
                                        o_grad_ptr,
                                        m,
                                        o_stride,
                                        a_ptr,
                                        k,
                                        a_stride,
                                        &alpha,
                                        b_grad_ptr,
                                        m,
                                        b_stride,
                                        batch));
}
