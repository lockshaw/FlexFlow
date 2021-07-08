#include "sparse.h"

void make_dense(const float *ptr, int num_rows, int num_cols, cusparseDnMatDescr_t &denseDesc)
{
  checkCUSPARSE(cusparseCreateDnMat(&denseDesc,
                                    num_rows/*num_rows*/,
                                    num_cols/*num_cols*/,
                                    num_rows/*leading dimension*/, // set to num_rows because we are column major
                                    (void*)ptr,
                                    CUDA_R_32F,
                                    CUSPARSE_ORDER_COL));
}

void make_dense_rm(const float *ptr, int num_rows, int num_cols, cusparseDnMatDescr_t &denseDesc) 
{
  checkCUSPARSE(cusparseCreateDnMat(&denseDesc,
                                    num_rows/*num_rows*/,
                                    num_cols/*num_cols*/,
                                    num_cols/*leading dimension*/, // set to num_cols because we are row major
                                    (void*)ptr,
                                    CUDA_R_32F,
                                    CUSPARSE_ORDER_ROW));
}

void free_all_dense(cusparseDnMatDescr_t &denseDesc) 
{
  checkCUSPARSE(cusparseDestroyDnMat(denseDesc));
}

void make_csr(cusparseHandle_t handle, const float *ptr, int num_rows, int num_cols, cusparseSpMatDescr_t &sparseDesc) 
{
  int *csr_offsets, *csr_columns;
  float *csr_values;
  // convert the dense weights passed in via ptr to a sparse CSR matrix
  cusparseDnMatDescr_t denseDesc;
  checkCUSPARSE(cusparseCreateDnMat(&denseDesc, 
                                    num_rows/*num_rows*/, 
                                    num_cols/*num_cols*/, 
                                    num_rows/*leading dimension*/, // set to num_rows because we are column major
                                    (void*)ptr,
                                    CUDA_R_32F,
                                    CUSPARSE_ORDER_COL));

  checkCUDA(cudaMalloc((void**)&csr_offsets, (num_rows + 1) * sizeof(int)));
  checkCUSPARSE(cusparseCreateCsr(&sparseDesc,  num_rows, num_cols, 0,
                                  csr_offsets, NULL, NULL,
                                  CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                  CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
  size_t workspaceSize = 0;
  checkCUSPARSE(cusparseDenseToSparse_bufferSize(handle, 
                                                 denseDesc,
                                                 sparseDesc,
                                                 CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                                 &workspaceSize));
  void *workspaceBuffer = NULL;
  checkCUDA(cudaMalloc(&workspaceBuffer, workspaceSize));
  checkCUSPARSE(cusparseDenseToSparse_analysis(handle, 
                                               denseDesc,
                                               sparseDesc,
                                               CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                               workspaceBuffer));
  int64_t num_rows_tmp, num_cols_tmp, nnz;
  checkCUSPARSE(cusparseSpMatGetSize(sparseDesc,
                                     &num_rows_tmp,
                                     &num_cols_tmp,
                                     &nnz));
  checkCUDA(cudaMalloc((void**)&csr_columns, nnz * sizeof(int)));
  checkCUDA(cudaMalloc((void**)&csr_values, nnz * sizeof(float)));

  checkCUSPARSE(cusparseCsrSetPointers(sparseDesc,
                                       csr_offsets,
                                       csr_columns,
                                       csr_values));

  checkCUSPARSE(cusparseDenseToSparse_convert(handle,
                                              denseDesc,
                                              sparseDesc,
                                              CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                              workspaceBuffer));

  checkCUSPARSE(cusparseDestroyDnMat(denseDesc));
  checkCUDA(cudaFree(workspaceBuffer));
}

void csr_to_dense(cusparseHandle_t handle, cusparseSpMatDescr_t const &csr, cusparseDnMatDescr_t &cm)
{
  size_t workspaceSize = 0;
  void *workspaceBuffer = nullptr;

  checkCUSPARSE(cusparseSparseToDense_bufferSize(handle,
                                                 csr,
                                                 cm,
                                                 CUSPARSE_SPARSETODENSE_ALG_DEFAULT,
                                                 &workspaceSize));
  checkCUDA(cudaMalloc(&workspaceBuffer, workspaceSize));
  checkCUSPARSE(cusparseSparseToDense(handle,
                                      csr,
                                      cm,
                                      CUSPARSE_SPARSETODENSE_ALG_DEFAULT,
                                      workspaceBuffer));

  checkCUDA(cudaFree(workspaceBuffer));
}

void free_all_csr(cusparseSpMatDescr_t &sparseDescr) 
{
  int64_t rows, cols, nnz;
  int *csr_offsets, *csr_columns;
  float *csr_values;
  cusparseIndexType_t csrRowOffsetsType, csrColIndType;
  cusparseIndexBase_t idxBase;
  cudaDataType_t valueType;
  checkCUSPARSE(cusparseCsrGet(sparseDescr,
                               &rows/*rows*/,
                               &cols/*cols*/,
                               &nnz/*nnz*/,
                               (void**)&csr_offsets/*csrRowOffsets*/,
                               (void**)&csr_columns/*csrColInd*/,
                               (void**)&csr_values/*csrValues*/,
                               &csrRowOffsetsType/*csrRowOffsetsType*/,
                               &csrColIndType/*csrColIndType*/,
                               &idxBase/*idxBase*/,
                               &valueType/*valueType*/));
  checkCUSPARSE(cusparseDestroySpMat(sparseDescr));
  checkCUDA(cudaFree(csr_offsets));
  checkCUDA(cudaFree(csr_columns));
  checkCUDA(cudaFree(csr_values));
}

void transpose_cm_to_rm(cusparseDnMatDescr_t const &cm, cusparseDnMatDescr_t &rm) 
{
  int64_t c_rows, c_cols, c_ld;
  float *c_values = nullptr;
  cudaDataType c_type;
  cusparseOrder_t c_order;
  checkCUSPARSE(cusparseDnMatGet(cm,
                                 &c_rows,
                                 &c_cols,
                                 &c_ld,
                                 (void**)&c_values,
                                 &c_type,
                                 &c_order));
  assert (c_ld == c_rows);
  assert (c_order == CUSPARSE_ORDER_COL);
  assert (c_type == CUDA_R_32F);

  make_dense_rm(c_values,
                c_cols,
                c_rows,
                rm);

}

void spmm(cusparseHandle_t handle,
          cusparseOperation_t opA,
          cusparseOperation_t opB,
          float alpha,
          cusparseSpMatDescr_t const &aDescr,
          cusparseDnMatDescr_t const &bDescr,
          float beta,
          cusparseDnMatDescr_t &cDescr) 
{
  void *workspaceBuffer = NULL;
  size_t workspaceSize = 0;

  checkCUSPARSE(cusparseSpMM_bufferSize(handle,
                                        opA,
                                        opB,
                                        &alpha, 
                                        aDescr,
                                        bDescr,
                                        &beta,
                                        cDescr,
                                        CUDA_R_32F,
                                        CUSPARSE_SPMM_ALG_DEFAULT, 
                                        &workspaceSize));
  checkCUDA(cudaMalloc(&workspaceBuffer, workspaceSize));

  checkCUSPARSE(cusparseSpMM(handle,
                             opA,
                             opB,
                             &alpha,
                             aDescr,
                             bDescr,
                             &beta,
                             cDescr,
                             CUDA_R_32F,
                             CUSPARSE_SPMM_ALG_DEFAULT,
                             workspaceBuffer));

  checkCUDA(cudaFree(workspaceBuffer));
}

void sddmm(cusparseHandle_t handle,
           cusparseOperation_t opA,
           cusparseOperation_t opB,
           float alpha,
           cusparseDnMatDescr_t matA,
           cusparseDnMatDescr_t matB,
           float beta,
           cusparseSpMatDescr_t matC)
{
  /* void *workspaceBuffer = NULL; */
  /* size_t workspaceSize = 0; */

  /* checkCUSPARSE(cusparseSDDMM_bufferSize(handle, */ 
  /*                                        opA, */
  /*                                        opB, */
  /*                                        &alpha, */
  /*                                        matA, */ 
  /*                                        matB, */
  /*                                        &beta, */
  /*                                        matC, */
  /*                                        CUDA_R_32F, */
  /*                                        CUSPARSE_SDDMM_ALG_DEFAULT, */
  /*                                        &workspaceSize)); */
  /* checkCUDA(cudaMalloc(&workspaceBuffer, workspaceSize)); */
  /* checkCUSPARSE(cusparseSDDMM_preprocess(handle, */
  /*                            opA, */
  /*                            opB, */
  /*                            &alpha, */
  /*                            matA, */
  /*                            matB, */
  /*                            &beta, */ 
  /*                            matC, */
  /*                            CUDA_R_32F, */
  /*                            CUSPARSE_SDDMM_ALG_DEFAULT, */
  /*                            workspaceBuffer)); */
  /* checkCUSPARSE(cusparseSDDMM(handle, */
  /*                            opA, */
  /*                            opB, */
  /*                            &alpha, */
  /*                            matA, */
  /*                            matB, */
  /*                            &beta, */
  /*                            matC, */
  /*                            CUDA_R_32F, */
  /*                            CUSPARSE_SDDMM_ALG_DEFAULT, */
  /*                            workspaceBuffer)); */
}

void spmm(cusparseHandle_t handle,
          cusparseOperation_t opA,
          cusparseOperation_t opB,
          float alpha,
          cusparseDnMatDescr_t const &aDescr,
          cusparseSpMatDescr_t const &bDescr,
          float beta,
          cusparseDnMatDescr_t &cDescr) 
{
  assert (opB == CUSPARSE_OPERATION_NON_TRANSPOSE);
  assert (opA != CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE);

  cusparseDnMatDescr_t aRM, cRM;
  transpose_cm_to_rm(aDescr, aRM);
  transpose_cm_to_rm(cDescr, cRM);

  cusparseOperation_t realOpA = (opA == CUSPARSE_OPERATION_NON_TRANSPOSE) 
                                  ? CUSPARSE_OPERATION_TRANSPOSE
                                  : CUSPARSE_OPERATION_NON_TRANSPOSE;
  
  spmm(handle,
       opB,
       realOpA,
       alpha,
       bDescr,
       aRM,
       beta,
       cRM);

  cusparseDestroyDnMat(aRM);
  cusparseDestroyDnMat(cRM);
}
