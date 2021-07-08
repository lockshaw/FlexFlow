#include "cuda_helper.h"
#include <cusparse.h>

void make_dense(const float *ptr, int num_rows, int num_cols, cusparseDnMatDescr_t &denseDesc);
void make_dense_rm(const float *ptr, int num_rows, int num_cols, cusparseDnMatDescr_t &denseDesc);
void make_csr(cusparseHandle_t handle, const float *ptr, int num_rows, int num_cols, cusparseSpMatDescr_t &sparseDescr);
void tranpose_cm_to_rm(cusparseDnMatDescr_t const &cm, cusparseDnMatDescr_t &rm);
void csr_to_dense(cusparseHandle_t handle, cusparseSpMatDescr_t const &csr, cusparseDnMatDescr_t &cm);
void free_all_csr(cusparseSpMatDescr_t &sparseDescr);
void free_all_dense(cusparseDnMatDescr_t &denseDesc);
void spmm(cusparseHandle_t handle,
          cusparseOperation_t opA,
          cusparseOperation_t opB,
          float alpha,
          cusparseSpMatDescr_t const &aDescr,
          cusparseDnMatDescr_t const &bDescr,
          float beta,
          cusparseDnMatDescr_t &cDescr);
void spmm(cusparseHandle_t handle,
          cusparseOperation_t opA,
          cusparseOperation_t opB,
          float alpha,
          cusparseDnMatDescr_t const &aDescr,
          cusparseSpMatDescr_t const &bDescr,
          float beta,
          cusparseDnMatDescr_t &cDescr);
void sddmm(cusparseHandle_t handle,
           cusparseOperation_t opA,
           cusparseOperation_t opB,
           float alpha,
           cusparseDnMatDescr_t matA,
           cusparseDnMatDescr_t matB,
           float beta,
           cusparseSpMatDescr_t matC);
