#include "gtest/gtest.h"
#include "sparse.h"
#include "cuda_helper.h"
#include "model.h"

using namespace Legion;

/* TEST(spmm, sparse_left) { */
/*   cusparseHandle_t handle = nullptr; */
/*   checkCUSPARSE(cusparseCreate(&handle)); */
/* } */

TEST(spmm, sparse_left) {
  /* cudaStream_t stream; */
  /* checkCUDA(get_legion_stream(&stream)); */

  constexpr int A_num_rows = 4,
                A_num_cols = 4,
                A_nnz = 9,
                B_num_rows = A_num_cols,
                B_num_cols = 3,
                C_num_rows = A_num_rows,
                C_num_cols = B_num_cols,
                ldb = B_num_rows,
                ldc = C_num_rows,
                B_size = B_num_rows * B_num_cols,
                C_size = C_num_rows * C_num_cols;

  int hA_csrOffsets[A_num_rows + 1] = { 0, 3, 4, 7, 9 };
  int hA_columns[A_nnz] = { 0, 2, 3, 1, 0, 2, 3, 1, 3 };
  float hA_values[A_nnz] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };

  std::array<float, B_size> hB = { 1.0, 2.0, 3.0, 4.0, 
                                   5.0, 6.0, 7.0, 8.0, 
                                   9.0, 10.0, 11.0, 12.0 };
  std::array<float, C_size> hC;
  std::fill(hC.begin(), hC.end(), 0.0);

  std::array<float, C_size> hC_result = { 19.0,  8.0,  51.0,  52.0, 
                                          43.0, 24.0, 123.0, 120.0,
                                          67.0, 40.0, 195.0, 188.0 };
  float alpha = 1.0;
  float beta = 1.0;

  int *dA_csrOffsets, *dA_columns;
  float *dA_values, *dB, *dC;

  checkCUDA(cudaMalloc((void**)&dA_csrOffsets,
                       (A_num_rows + 1) * sizeof(int)));
  checkCUDA(cudaMalloc((void**)&dA_columns, A_nnz * sizeof(int)));
  checkCUDA(cudaMalloc((void**)&dA_values, A_nnz * sizeof(float)));
  checkCUDA(cudaMalloc((void**)&dB, B_size * sizeof(float)));
  checkCUDA(cudaMalloc((void**)&dC, C_size * sizeof(float)));

  checkCUDA(cudaMemcpy(dA_csrOffsets, 
                       hA_csrOffsets,
                       (A_num_rows + 1) * sizeof(int),
                       cudaMemcpyHostToDevice));
  checkCUDA(cudaMemcpy(dA_columns,
                       hA_columns,
                       A_nnz * sizeof(int),
                       cudaMemcpyHostToDevice));
  checkCUDA(cudaMemcpy(dA_values,
                       hA_values,
                       A_nnz * sizeof(float),
                       cudaMemcpyHostToDevice));
  checkCUDA(cudaMemcpy(dB, 
                       hB.data(), 
                       B_size * sizeof(float),
                       cudaMemcpyHostToDevice));
  checkCUDA(cudaMemcpy(dC, 
                       hC.data(), 
                       C_size * sizeof(float),
                       cudaMemcpyHostToDevice));

  cusparseHandle_t handle = nullptr;
  checkCUSPARSE(cusparseCreate(&handle));
  /* checkCUSPARSE(cusparseSetStream(handle, stream)); */

  cusparseSpMatDescr_t matA;
  checkCUSPARSE(cusparseCreateCsr(&matA, 
                                  A_num_rows, A_num_cols, A_nnz,
                                  dA_csrOffsets, dA_columns, dA_values,
                                  CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                  CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

  cusparseDnMatDescr_t matB, matC;
  checkCUSPARSE(cusparseCreateDnMat(&matB,
                                    B_num_rows, B_num_cols, ldb,
                                    dB, CUDA_R_32F, CUSPARSE_ORDER_COL));
  checkCUSPARSE(cusparseCreateDnMat(&matC,
                                    C_num_rows, C_num_cols, ldc,
                                    dC, CUDA_R_32F, CUSPARSE_ORDER_COL));

  spmm(handle,
       CUSPARSE_OPERATION_NON_TRANSPOSE,
       CUSPARSE_OPERATION_NON_TRANSPOSE,
       alpha,
       matA, 
       matB,
       beta,
       matC);

  checkCUDA(cudaMemcpy(hC.data(),
                       dC,
                       C_size * sizeof(float),
                       cudaMemcpyDeviceToHost));

  checkCUSPARSE(cusparseDestroySpMat(matA));
  checkCUSPARSE(cusparseDestroyDnMat(matB));
  checkCUSPARSE(cusparseDestroyDnMat(matC));
  checkCUSPARSE(cusparseDestroy(handle));
  checkCUDA(cudaFree(dA_csrOffsets));
  checkCUDA(cudaFree(dA_columns));
  checkCUDA(cudaFree(dA_values));
  checkCUDA(cudaFree(dB));
  checkCUDA(cudaFree(dC));

  EXPECT_EQ(hC, hC_result);
}
