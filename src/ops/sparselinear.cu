#include "model.h"
#include "cuda_helper.h"
#include "sputnik/sputnik.h"
#include <cusparse.h>
#include "sparse.h"
#include "linear_utils.h"

Tensor FFModel::sparse_linear(const Tensor& input,
                              int outDim,
                              ActiMode activation,
                              bool use_bias,
                              const Op *shared_op,
                              Initializer *kernel_initializer,
                              Initializer *bias_initializer,
                              const char *name)
{
  if (kernel_initializer == NULL) {
    int seed = std::rand();
    kernel_initializer = new GlorotUniform(seed);
  }
  if (bias_initializer == NULL) {
    bias_initializer = new ZeroInitializer();
  }

  SparseLinear *sl = new SparseLinear(*this, input, outDim, activation, use_bias, shared_op, 
                                      kernel_initializer, bias_initializer, name);
  layers.push_back(sl);
  return sl->outputs[0];
}

SparseLinear::SparseLinear(FFModel &model, 
                           const Tensor& input,
                           int out_dim,
                           ActiMode activation,
                           bool use_bias,
                           const Op *shared_op, 
                           Initializer *kernel_initializer,
                           Initializer *bias_initializer,
                           const char *name) 
: Op(model, OP_SPARSELINEAR, shared_op, name, input),
  in_channels(input.adim[0]), 
  out_channels(out_dim),
  activation(activation),
  use_bias(use_bias),
  kernel_initializer(kernel_initializer),
  bias_initializer( bias_initializer)
{
  assert (input.numDim == 2);
 
  this->numInputs = 1;
  this->numOutputs = 1;
  this->outputs[0].numDim = 2;
  this->outputs[0].adim[0] = this->out_channels;
  this->outputs[0].adim[1] = input.adim[1];
  weights[0].numDim = 2;
  weights[0].adim[0] = in_channels;
  weights[0].adim[1] = out_channels;
  numWeights = 1;
  if (use_bias) {
    weights[1].numDim = 1;
    weights[1].adim[0] = out_channels;
    numWeights = 2;
  }
}

SparseLinearMeta::SparseLinearMeta(FFHandler handler, int batch_size)
  : OpMeta(handler)
{
  this->one_ptr = [&] {
    float *dram_one_ptr = (float *)malloc(batch_size * sizeof(float));
    float *fb_one_ptr = nullptr;
    std::fill_n(dram_one_ptr, batch_size, 1.0);
    cudaMalloc((void**)&fb_one_ptr, batch_size * sizeof(float));
    cudaMemcpy((void*)fb_one_ptr, (void*)dram_one_ptr, sizeof(float) * batch_size, cudaMemcpyHostToDevice);
    free(dram_one_ptr);
    return fb_one_ptr;
  }();

  checkCUDNN(cudnnCreateActivationDescriptor(&actiDesc));
  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
}

SparseLinearMeta::~SparseLinearMeta() 
{
  checkCUDA(cudaFree((void*)this->one_ptr));
}

/*
   regions[0](O): output
   regions[1](I): kernel
   regions[2](I): bias (optional)
*/
OpMeta* SparseLinear::init_task(const Task *task,
                                const std::vector<PhysicalRegion> &regions,
                                Context ctx, Runtime *runtime) 
{
  assert (regions.size() == task->regions.size());
  assert (regions.size() == 2 || regions.size() == 3);
  const SparseLinear* sparselinear = (SparseLinear*) task->args;
  FFHandler handle = *((const FFHandler*) task->local_args);

  TensorAccessorW<float, 2> acc_output(
      regions[0], task->regions[0], FID_DATA, ctx, runtime, false/*readOutput*/);
  TensorAccessorR<float, 2> acc_kernel(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);
  int in_dim = acc_kernel.rect.hi[0] - acc_kernel.rect.lo[0] + 1;
  int out_dim = acc_output.rect.hi[0] - acc_output.rect.lo[0] + 1;
  int batch_size = acc_output.rect.volume() / out_dim;
  SparseLinearMeta *m = new SparseLinearMeta(handle, batch_size);
  m->activation = sparselinear->activation;
  m->use_bias = sparselinear->use_bias;
  m->profiling = sparselinear->profiling;
  m->trainableInputs[0] = sparselinear->trainableInputs[0];
  std::strcpy(m->op_name, sparselinear->name);
  if (use_cudnn_activation(m->activation)) {
    cudnnActivationMode_t mode;
    switch (sparselinear->activation) {
      case AC_MODE_RELU:
        mode = CUDNN_ACTIVATION_RELU;
        break;
      case AC_MODE_SIGMOID:
        mode = CUDNN_ACTIVATION_SIGMOID;
        break;
      default:
        // Unsupported activation mode
        assert(false);
    }
    checkCUDNN(cudnnSetActivationDescriptor(m->actiDesc, mode,
                                            CUDNN_PROPAGATE_NAN, 0.0));
    checkCUDNN(cudnnSetTensor4dDescriptor(m->outputTensor,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          batch_size, out_dim, 1, 1));
  }
  return m;
}

void SparseLinear::init(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  Rect<2> rect = runtime->get_index_space_domain(ctx, task_is);
  ParallelConfig pc;
  std::string pcname = name;
  ff.config.find_parallel_config(2, pcname, pc);
  int idx = 0;
  for (PointInRectIterator<2> it(rect); it(); it++) {
    FFHandler handle = ff.handlers[pc.device_ids[idx++]];
#ifdef FF_USE_NCCL
    handle.ncclComm = pc.nccl_comms[idx-1];
#endif
    argmap.set_point(*it, TaskArgument(&handle, sizeof(FFHandler)));
  }
  IndexLauncher launcher(LINEAR_INIT_TASK_ID, task_is,
                         TaskArgument(this, sizeof(Linear)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));
  //launcher.add_region_requirement(
  //    RegionRequirement(input_lps[0], 0/*projection id*/,
  //                      READ_ONLY, EXCLUSIVE, inputs[0].region));
  //launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(outputs[0].part, 0/*projection id*/,
                        WRITE_ONLY, EXCLUSIVE, outputs[0].region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(weights[0].part, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, weights[0].region));
  launcher.add_field(1, FID_DATA);
  // launcher.add_region_requirement(
  //     RegionRequirement(weights[1].part, 0/*projection id*/,
  //                       READ_ONLY, EXCLUSIVE, weights[1].region));
  // launcher.add_field(3, FID_DATA);
  if (ff.config.computationMode == COMP_MODE_TRAINING) {
    // Add inputs[0].region_grad to avoid Legion warning
    //launcher.add_region_requirement(
    //    RegionRequirement(input_grad_lps[0], 0/*projection id*/,
    //        WRITE_ONLY, EXCLUSIVE, inputs[0].region_grad));
    //launcher.add_field(2, FID_DATA);
  }
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  idx = 0;
  for (PointInRectIterator<2> it(rect); it(); it++) {
    meta[idx++] = fm.get_result<OpMeta*>(*it);
  }
}

bool SparseLinear::use_cudnn_activation(ActiMode mode)
{
  switch (mode) {
    case AC_MODE_RELU:
    case AC_MODE_SIGMOID:
    case AC_MODE_TANH:
      return true;
  }
  return false;
}
                              
/*static*/
void SparseLinear::forward_kernel(const SparseLinearMeta* m, 
                                  const float *input_ptr,
                                  float *output_ptr,
                                  const float *kernel_ptr,
                                  const float *bias_ptr,
                                  int in_dim, int out_dim, int batch_size,
                                  cudaStream_t stream) 
{ 
  checkCUSPARSE(cusparseSetStream(m->handle.sparse, stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
  checkCUDA(cublasSetStream(m->handle.blas, stream));

  cusparseDnMatDescr_t outputDesc;
  {
    // create the cusparse descr for output tensor
    make_dense(output_ptr, out_dim/*num_rows*/, batch_size/*num_cols*/, outputDesc);

    // create the cusparse descr for input tensor
    cusparseDnMatDescr_t inputDesc;
    make_dense(input_ptr, in_dim/*num_rows*/, batch_size/*num_cols*/, inputDesc);

    // create the sparse cusparse descr for kernel tensor
    cusparseSpMatDescr_t sparseKernelDesc;
    make_csr(m->handle.sparse, kernel_ptr, out_dim, in_dim, sparseKernelDesc);
     
    spmm(m->handle.sparse,
         CUSPARSE_OPERATION_NON_TRANSPOSE,
         CUSPARSE_OPERATION_NON_TRANSPOSE,
         1.0, 
         sparseKernelDesc,
         inputDesc,
         0.0, 
         outputDesc);

    free_all_csr(sparseKernelDesc);
  }
  free_all_dense(outputDesc);

  if (bias_ptr != NULL) {
    float alpha = 1.0, beta = 1.0;
    checkCUDA(cublasSgemm(m->handle.blas, CUBLAS_OP_T, CUBLAS_OP_N,
                          out_dim, batch_size, 1,
                          &alpha, bias_ptr, 1,
                          m->one_ptr, 1, &beta,
                          output_ptr, out_dim));
  }

  if (use_cudnn_activation(m->activation)) {
    float alpha = 1.0f, beta = 0.0f;
    checkCUDNN(cudnnActivationForward(m->handle.dnn, m->actiDesc,
        &alpha, m->outputTensor, output_ptr,
        &beta, m->outputTensor, output_ptr));
  } else if (m->activation == AC_MODE_GELU) {
    size_t elements = (size_t)out_dim * (size_t) batch_size;
    constexpr float B = 0.7978845608028654f;   // sqrt(2.0/M_PI)
    constexpr float C = 0.035677408136300125f; // 0.044715 * sqrt(2.0/M_PI)
    gelu_forward_kernel<<<GET_BLOCKS(elements), CUDA_NUM_THREADS>>>(
        elements, B, C, output_ptr);
  } else if (m->activation == AC_MODE_NONE) {
    // Do nothing
  } else {
    assert(false && "Unsupported activation for Linear");
  }
}

void SparseLinear::forward_task(const Task *task, 
                                const std::vector<PhysicalRegion> &regions,
                                Context ctx, Runtime *runtime) 
{
  const SparseLinearMeta *m = *((SparseLinearMeta**)task->local_args);
  assert (regions.size() == (3 + int(m->use_bias)));
  assert (task->regions.size() == (3 + int(m->use_bias)));

  TensorAccessorR<float, 2> acc_input(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<float, 2> acc_output(
      regions[1], task->regions[1], FID_DATA, ctx, runtime,
      false/*readOutput*/);
  TensorAccessorR<float, 2> acc_kernel(
      regions[2], task->regions[2], FID_DATA, ctx, runtime);
  int in_dim = acc_input.rect.hi[0] - acc_input.rect.lo[0] + 1;
  int out_dim = acc_output.rect.hi[0] - acc_output.rect.lo[0] + 1;
  int batch_size = acc_output.rect.volume() / out_dim;
  assert(acc_output.rect.volume() == out_dim * batch_size);
  assert(acc_input.rect.volume() == in_dim * batch_size);
  assert(acc_kernel.rect.volume() == in_dim * out_dim);
  const float* acc_bias_ptr = NULL;
  if (m->use_bias) {
    TensorAccessorR<float, 1> acc_bias(
        regions[3], task->regions[3], FID_DATA, ctx, runtime);
    assert(acc_bias.rect.volume() == out_dim);
    acc_bias_ptr = acc_bias.ptr;
  }

  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }
  SparseLinear::forward_kernel(m, 
                               acc_input.ptr, 
                               acc_output.ptr,
                               acc_kernel.ptr,
                               acc_bias_ptr,
                               in_dim,
                               out_dim,
                               batch_size,
                               stream);

  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("%s [SparseLinear] forward time = %.2lfms\n", m->op_name, elapsed);
    //print_tensor<NDIM, float>(acc_input.ptr, acc_input.rect, "[Linear:forward:input]");
    //print_tensor<2, float>(acc_kernel.ptr, acc_kernel.rect, "[Linear:forward:kernel]");
    //print_tensor<1, float>(acc_bias.ptr, acc_bias.rect, "[Linear:forward:bias]");
    //print_tensor<NDIM, float>(acc_output.ptr, acc_output.rect, "[Linear:forward:output]");
  }
}

void SparseLinear::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  Rect<2> rect = runtime->get_index_space_domain(ctx, task_is);
  int idx = 0;
  for (PointInRectIterator<2> it(rect); it(); it++) {
    OpMeta* mp = meta[idx++];
    argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta*)));
  }
  IndexLauncher launcher(SPARSE_LINEAR_FWD_TASK_ID, task_is,
                         TaskArgument(NULL, 0), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));
  launcher.add_region_requirement(
      RegionRequirement(input_lps[0], 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, inputs[0].region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(outputs[0].part, 0/*projection id*/,
                        WRITE_ONLY, EXCLUSIVE, outputs[0].region));
  launcher.add_field(1, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(weights[0].part, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, weights[0].region));
  launcher.add_field(2, FID_DATA);
  if (use_bias) {
    launcher.add_region_requirement(
        RegionRequirement(weights[1].part, 0/*projection id*/,
                          READ_ONLY, EXCLUSIVE, weights[1].region));
    launcher.add_field(3, FID_DATA);
  }
  runtime->execute_index_space(ctx, launcher);

}

/*static*/
void SparseLinear::backward_kernel(const SparseLinearMeta *m,
                                   const float *input_ptr,
                                   float *input_grad_ptr,
                                   const float *output_ptr,
                                   float *output_grad_ptr,
                                   const float *kernel_ptr,
                                   float *kernel_grad_ptr,
                                   float *bias_grad_ptr,
                                   int in_dim, int out_dim, int batch_size,
                                   cudaStream_t stream) 
{ 
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
  checkCUSPARSE(cusparseSetStream(m->handle.sparse, stream));
  checkCUDA(cublasSetStream(m->handle.blas, stream));

  float alpha = 1.0f;
  int output_size = out_dim * batch_size;
  if (m->activation == AC_MODE_RELU) {
    reluBackward<<<GET_BLOCKS(output_size), CUDA_NUM_THREADS, 0, stream>>>(
        output_grad_ptr, output_ptr, output_size);
  } else if (m->activation == AC_MODE_SIGMOID) {
    sigmoid_backward<<<GET_BLOCKS(output_size), CUDA_NUM_THREADS, 0, stream>>>(
        output_grad_ptr, output_ptr, output_size);
  } else {
    // TODO: only support relu and sigmoid for now
    assert(m->activation == AC_MODE_NONE);
  }

  cusparseDnMatDescr_t inputDesc, 
                       outputGradDesc,
                       denseKernelGradDesc;
  cusparseSpMatDescr_t sparseKernelGradDesc;

  make_dense(input_ptr, in_dim/*num_rows*/, batch_size/*num_cols*/, inputDesc);
  make_dense(output_grad_ptr, out_dim/*num_rows*/, batch_size/*num_cols*/, outputGradDesc);
  make_dense(kernel_grad_ptr, out_dim/*num_rows*/, in_dim/*num_cols*/, denseKernelGradDesc);
  make_csr(m->handle.sparse, kernel_grad_ptr, out_dim/*num_rows*/, in_dim/*num_cols*/, sparseKernelGradDesc);

  sddmm(m->handle.sparse,
        CUSPARSE_OPERATION_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        1.0,
        inputDesc,
        outputGradDesc,
        1.0,
        sparseKernelGradDesc);
  csr_to_dense(m->handle.sparse, sparseKernelGradDesc, denseKernelGradDesc);

  if (bias_grad_ptr != NULL) {
    checkCUDA(cublasSgemv(m->handle.blas, CUBLAS_OP_N,
                          out_dim, batch_size,
                          &alpha, output_grad_ptr, out_dim,
                          m->one_ptr, 1,
                          &alpha, bias_grad_ptr, 1));
  }
  // Compute data gradiant
  // NOTE: we use alpha=1 for input_grad to accumulate gradients
  if (input_grad_ptr != NULL) {
    cusparseDnMatDescr_t inputGradDesc;
    cusparseSpMatDescr_t kernelDesc;
    make_dense(input_grad_ptr, in_dim/*num_rows*/, batch_size/*num_cols*/, inputGradDesc);
    make_csr(m->handle.sparse, kernel_ptr, out_dim/*num_rows*/, in_dim/*num_cols*/, kernelDesc);
    spmm(m->handle.sparse,
         CUSPARSE_OPERATION_NON_TRANSPOSE,
         CUSPARSE_OPERATION_TRANSPOSE,
         1.0,
         outputGradDesc,
         kernelDesc,
         1.0,
         inputGradDesc);
    free_all_dense(inputGradDesc);
    free_all_csr(kernelDesc);
  }

  free_all_dense(inputDesc);
  free_all_dense(outputGradDesc);
  free_all_dense(denseKernelGradDesc);
  free_all_csr(sparseKernelGradDesc);
}

/*
  regions[0](I): input
  regions[1](I/O): replica_grad or input_grad
  regions[2](I): output
  regions[3](I/O): output_grad
  regions[4](I): filter
  regions[5](I/O): filter_grad
  regions[6](I/O): bias_grad
*/
void SparseLinear::backward_task(const Task *task,
                                 const std::vector<PhysicalRegion> &regions,
                                 Context ctx, Runtime *runtime) 
{
  const SparseLinearMeta* m = *((SparseLinearMeta**) task->local_args);
  assert(regions.size() == (5 + int(m->trainableInputs[0]) + int(m->use_bias)));
  assert(task->regions.size() == (5 + int(m->trainableInputs[0]) + int(m->use_bias)));
  float* input_grad = NULL;
  size_t rid = 0;
  TensorAccessorR<float, 2> acc_input(
      regions[rid], task->regions[rid], FID_DATA, ctx, runtime);
  rid++;
  if (m->trainableInputs[0]) {
    Domain domain = runtime->get_index_space_domain(
        ctx, task->regions[rid].region.get_index_space());
    if (domain.get_dim() == 3) {
      assert(domain.get_volume() == acc_input.rect.volume());
      input_grad = helperGetTensorPointerWO<float>(
          regions[rid], task->regions[rid], FID_DATA, ctx, runtime);
    } else {
      TensorAccessorW<float, 2> acc_replica_grad(
          regions[rid], task->regions[rid], FID_DATA, ctx, runtime,
          true/*readOutput*/);
      assert(acc_replica_grad.rect.volume() == acc_input.rect.volume());
      input_grad = acc_replica_grad.ptr;
    }
    rid++;
  }
  TensorAccessorR<float, 2> acc_output(
      regions[rid], task->regions[rid], FID_DATA, ctx, runtime);
  rid++;
  TensorAccessorW<float, 2> acc_output_grad(
      regions[rid], task->regions[rid], FID_DATA, ctx, runtime,
      true/*readOutput*/);
  rid++;
  TensorAccessorR<float, 2> acc_kernel(
      regions[rid], task->regions[rid], FID_DATA, ctx, runtime);
  rid++;
  TensorAccessorW<float, 2> acc_kernel_grad(
      regions[rid], task->regions[rid], FID_DATA, ctx, runtime,
      true/*readOutput*/);
  rid++;
  // make sure the sizes match
  int in_dim = acc_input.rect.hi[0] - acc_input.rect.lo[0] + 1;
  int out_dim = acc_output.rect.hi[0] - acc_output.rect.lo[0] + 1;
  int batch_size = acc_output.rect.volume() / out_dim;
  assert(acc_output.rect.volume() == out_dim * batch_size);
  assert(acc_output_grad.rect.volume() == out_dim * batch_size);
  assert(acc_kernel.rect.volume() == in_dim * out_dim);
  assert(acc_kernel_grad.rect.volume() == in_dim * out_dim);
  float* acc_bias_grad_ptr = NULL;
  if (m->use_bias) {
    TensorAccessorW<float, 1> acc_bias_grad(
        regions[rid], task->regions[rid], FID_DATA, ctx, runtime,
        true/*readOutput*/);
    rid++;
    assert(acc_bias_grad.rect.volume() == out_dim);
    acc_bias_grad_ptr = static_cast<float*>(acc_bias_grad.ptr);
  }
  assert(rid == regions.size());

  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }
  SparseLinear::backward_kernel(m, acc_input.ptr, input_grad,
      acc_output.ptr, acc_output_grad.ptr,
      acc_kernel.ptr, acc_kernel_grad.ptr,
      acc_bias_grad_ptr, in_dim, out_dim, batch_size, stream);
  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("SparseLinear backward time = %.2lfms\n", elapsed);
    //print_tensor<NDIM, float>(acc_output_grad.ptr, acc_output_grad.rect, "[Linear:backward:output_grad]");
    //print_tensor<2, float>(acc_kernel_grad.ptr, acc_kernel_grad.rect, "[Linear:backward:kernel_grad]");
    //print_tensor<1, float>(acc_bias_grad.ptr, acc_bias_grad.rect, "[Linear:backward:bias_grad]");
    //print_tensor<2, float>(input_grad, acc_input.rect, "[Linear:backward:input_grad]");
  }
}

void SparseLinear::backward(const FFModel &ff)
{
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  {
    ArgumentMap argmap;
    Rect<2> rect = runtime->get_index_space_domain(ctx, task_is);
    int idx = 0;
    for (PointInRectIterator<2> it(rect); it(); it++) {
      OpMeta* mp = meta[idx++];
      argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta*)));
    }
    IndexLauncher launcher(LINEAR_BWD_TASK_ID, task_is,
                           TaskArgument(NULL, 0), argmap,
                           Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                           FFConfig::get_hash_id(std::string(name)));
    int rid = 0;
    // regions[0](I): input
    launcher.add_region_requirement(
        RegionRequirement(input_lps[0], 0/*projection id*/,
                          READ_ONLY, EXCLUSIVE, inputs[0].region));
    launcher.add_field(rid++, FID_DATA);
    // regions[1](I/O): replica_grad
    if (trainableInputs[0]) {
      if (replica.region_grad != LogicalRegion::NO_REGION) {
        launcher.add_region_requirement(
            RegionRequirement(replica.part_grad, 0/*projection id*/,
                              WRITE_ONLY, EXCLUSIVE, replica.region_grad));
        launcher.add_field(rid++, FID_DATA);
      } else {
        launcher.add_region_requirement(
            RegionRequirement(input_grad_lps[0], 0/*projection id*/,
                              READ_WRITE, EXCLUSIVE, inputs[0].region_grad));
        launcher.add_field(rid++, FID_DATA);
      }
    }
    // regions[2](I): output
    launcher.add_region_requirement(
        RegionRequirement(outputs[0].part, 0/*projection id*/,
                          READ_ONLY, EXCLUSIVE, outputs[0].region));
    launcher.add_field(rid++, FID_DATA);
    // regions[3](I/O): output_grad
    launcher.add_region_requirement(
        RegionRequirement(outputs[0].part_grad, 0/*projection id*/,
                          READ_WRITE, EXCLUSIVE, outputs[0].region_grad));
    launcher.add_field(rid++, FID_DATA);
    // regions[4](I): filter
    launcher.add_region_requirement(
        RegionRequirement(weights[0].part, 0/*projection id*/,
                          READ_ONLY, EXCLUSIVE, weights[0].region));
    launcher.add_field(rid++, FID_DATA);
    // regions[5](I/O): filter_grad
    launcher.add_region_requirement(
        RegionRequirement(weights[0].part_grad, 0/*projection id*/,
                          READ_WRITE, EXCLUSIVE, weights[0].region_grad));
    launcher.add_field(rid++, FID_DATA);
    if (use_bias) {
      // regions[6](I/O): bias_grad
      launcher.add_region_requirement(
          RegionRequirement(weights[1].part_grad, 0/*projection id*/,
                            READ_WRITE, EXCLUSIVE, weights[1].region_grad));
      launcher.add_field(rid++, FID_DATA);
    }
    runtime->execute_index_space(ctx, launcher);
  }
  if (replica.region_grad != LogicalRegion::NO_REGION && trainableInputs[0]) {
    // We aggregate parameters from replica tensor to input tensor
    // Note we use input's task_is to reduce extra data transfers
    ArgumentMap argmap;
    Rect<2> input_rect = runtime->get_index_partition_color_space(
      ctx, inputs[0].part_grad.get_index_partition());
    IndexSpaceT<2> input_task_is = IndexSpaceT<2>(ff.get_task_is(input_rect));
    // If we are the first layer, our input uses data parallel and does
    // not have an owner
    std::string input_pcname = "";
    if (inputs[0].owner_op != NULL)
      input_pcname = std::string(inputs[0].owner_op->name);
    IndexLauncher launcher(LINEAR_BWD2_TASK_ID, input_task_is,
                           TaskArgument(this, sizeof(Linear)), argmap,
                           Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                           FFConfig::get_hash_id(input_pcname));
    launcher.add_region_requirement(
        RegionRequirement(inputs[0].part_grad, 0/*projection id*/,
                          READ_WRITE, EXCLUSIVE, inputs[0].region_grad));
    launcher.add_field(0, FID_DATA);
    // Note that replica.part save's a partition of replica.region_grad
    launcher.add_region_requirement(
        RegionRequirement(replica.part, 0/*partition id*/,
                          READ_ONLY, EXCLUSIVE, replica.region_grad));
    launcher.add_field(1, FID_DATA);
    runtime->execute_index_space(ctx, launcher);
  }

}


void SparseLinear::print_layer(const FFModel &ff)
{
  assert(false && "SparseLinear::print_layer not implemented");
}

void SparseLinear::create_output_and_partition(FFModel &ff) 
{
  assert (false && "SparseLinear::create_output_and_partition not implemented");
}

void SparseLinear::create_weights(FFModel &ff) 
{
  assert (false && "SparseLinear::create_weights not implemented");
}

bool SparseLinear::measure_operator_cost(Simulator *sim,
                                         const ParallelConfig &pc,
                                         CostMetrics& cost_metrics) 
{
  assert (false && "SparseLinear::measure_operator_cost not implemented");
  return false;
}
