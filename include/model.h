/* Copyright 2020 Stanford
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

#ifndef _FLEXFLOW_MODEL_H_
#define _FLEXFLOW_MODEL_H_
#include "legion.h"
#include "config.h"
#include "tensor.h"
#include "initializer.h"
#include "simulator.h"
#include "optimizer.h"
#include "accessor.h"
#include "loss_functions.h"
#include "metrics_functions.h"
#include <cuda_runtime.h>
#include <curand.h>
#include <unistd.h>
#ifdef FF_ENABLE_NCCL
#include <mpi.h>
#endif

using namespace Legion;

#include "ffconst.h"

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  FF_INIT_TASK_ID,
  IMAGE_INIT_TASK_ID,
  LABEL_INIT_TASK_ID,
  LOAD_IMAGES_TASK_ID,
  NORMALIZE_IMAGES_TASK_ID,
  ELEMENTBINARY_INIT_TASK_ID,
  ELEMENTBINARY_FWD_TASK_ID,
  ELEMENTBINARY_BWD_TASK_ID,
  ELEMENTUNARY_INIT_TASK_ID,
  ELEMENTUNARY_FWD_TASK_ID,
  ELEMENTUNARY_BWD_TASK_ID,
  CONV2D_INIT_TASK_ID,
  CONV2D_INIT_PARA_TASK_ID,
  CONV2D_FWD_TASK_ID,
  CONV2D_BWD_TASK_ID,
  CONV2D_UPD_TASK_ID,
  DROPOUT_INIT_TASK_ID,
  DROPOUT_FWD_TASK_ID,
  DROPOUT_BWD_TASK_ID,
  EMBED_INIT_TASK_ID,
  EMBED_FWD_TASK_ID,
  EMBED_BWD_TASK_ID,
  POOL2D_INIT_TASK_ID,
  POOL2D_FWD_TASK_ID,
  POOL2D_BWD_TASK_ID,
  BATCHNORM_INIT_TASK_ID,
  BATCHNORM_INIT_PARA_TASK_ID,
  BATCHNORM_FWD_TASK_ID,
  BATCHNORM_BWD_TASK_ID,
  BATCHMATMUL_INIT_TASK_ID,
  BATCHMATMUL_FWD_TASK_ID,
  BATCHMATMUL_BWD_TASK_ID,
  LINEAR_INIT_TASK_ID,
  LINEAR_INIT_PARA_TASK_ID,
  LINEAR_FWD_TASK_ID,
  LINEAR_BWD_TASK_ID,
  LINEAR_BWD2_TASK_ID,
  LINEAR_UPD_TASK_ID,
  FLAT_INIT_TASK_ID,
  FLAT_FWD_TASK_ID,
  FLAT_BWD_TASK_ID,
  SOFTMAX_INIT_TASK_ID,
  SOFTMAX_FWD_TASK_ID,
  SOFTMAX_BWD_TASK_ID,
  CONCAT_INIT_TASK_ID,
  CONCAT_FWD_TASK_ID,
  CONCAT_BWD_TASK_ID,
  SPLIT_INIT_TASK_ID,
  SPLIT_FWD_TASK_ID,
  SPLIT_BWD_TASK_ID,
  RESHAPE_INIT_TASK_ID,
  RESHAPE_FWD_TASK_ID,
  RESHAPE_BWD_TASK_ID,
  REVERSE_INIT_TASK_ID,
  REVERSE_FWD_TASK_ID,
  REVERSE_BWD_TASK_ID,
  TRANSPOSE_INIT_TASK_ID,
  TRANSPOSE_FWD_TASK_ID,
  TRANSPOSE_BWD_TASK_ID,
  ATTENTION_INIT_TASK_ID,
  ATTENTION_FWD_TASK_ID,
  ATTENTION_BWD_TASK_ID,
  MSELOSS_BWD_TASK_ID,
  FUSEDOP_FWD_TASK_ID,
  FUSEDOP_BWD_TASK_ID,
  //Metrics tasks
  METRICS_COMP_TASK_ID,
  UPDATE_METRICS_TASK_ID,
  DUMMY_TASK_ID,
  // Loss
  LOSS_BWD_TASK_ID,
  // Optimizer with PS
  SGD_UPD_PS_TASK_ID,
  ADAM_UPD_PS_TASK_ID,
  // Optimizer with NCCL
  SGD_UPD_NCCL_TASK_ID,
  ADAM_UPD_NCCL_TASK_ID,
  // Initializer
  GLOROT_INIT_TASK_ID,
  ZERO_INIT_TASK_ID,
  CONSTANT_INIT_TASK_ID,
  UNIFORM_INIT_TASK_ID,
  NORMAL_INIT_TASK_ID,
  // Search
  STRATEGY_SEARCH_TASK_ID,
  // Python data loader
  PY_DL_FLOAT_LOAD_ENTIRE_CPU_TASK_ID,
  PY_DL_INT_LOAD_ENTIRE_CPU_TASK_ID,
  PY_DL_FLOAT_LOAD_BATCH_GPU_TASK_ID,
  PY_DL_INT_LOAD_BATCH_GPU_TASK_ID,
  // Custom tasks
  CUSTOM_GPU_TASK_ID_FIRST,
  CUSTOM_GPU_TASK_ID_1,
  CUSTOM_GPU_TASK_ID_2,
  CUSTOM_GPU_TASK_ID_3,
  CUSTOM_GPU_TASK_ID_4,
  CUSTOM_GPU_TASK_ID_5,
  CUSTOM_GPU_TASK_ID_6,
  CUSTOM_GPU_TASK_ID_7,
  CUSTOM_GPU_TASK_ID_LAST,
  CUSTOM_CPU_TASK_ID_FIRST,
  CUSTOM_CPU_TASK_ID_1,
  CUSTOM_CPU_TASK_ID_2,
  CUSTOM_CPU_TASK_ID_3,
  CUSTOM_CPU_TASK_ID_4,
  CUSTOM_CPU_TASK_ID_5,
  CUSTOM_CPU_TASK_ID_6,
  CUSTOM_CPU_TASK_ID_7,
  CUSTOM_CPU_TASK_ID_LAST,
  // Make sure PYTHON_TOP_LEVEL_TASK_ID is
  // consistent with python/main.cc
  PYTHON_TOP_LEVEL_TASK_ID = 11111,
};

enum ShardingID {
  DataParallelShardingID = 135,
};

enum FieldIDs {
  FID_DATA,
};

class FFModel;
class Op;
class DataLoader;

class OpMeta {
public:
  OpMeta(FFHandler _handle) : handle(_handle) {};
public:
  FFHandler handle;
};

class Op {
public:
  Op(FFModel& model, OperatorType type, const std::string& _name, const Tensor& input);
  Op(FFModel& model, OperatorType type, const std::string& _name, const Tensor& input1, const Tensor& input2);
  Op(FFModel& model, OperatorType type, const std::string& _name, const Tensor& input1, const Tensor& input2, const Tensor& input3);
  Op(FFModel& model, OperatorType type, const std::string& _name, int num, const Tensor* inputs);
  Op(FFModel& model, OperatorType type, const std::string& _name, int num);

  Op(FFModel& model, OperatorType type, const Op* shared_op, const std::string& _name, const Tensor& input);
  // Pure virtual functions that must be implemented
  virtual void init(const FFModel&) = 0;
  virtual void forward(const FFModel&) = 0;
  virtual void backward(const FFModel&) = 0;
  virtual void create_weights(FFModel& model) = 0;
  virtual void create_output_and_partition(FFModel& model) = 0;
  virtual void print_layer(const FFModel& model) = 0;
  virtual bool measure_compute_time(Simulator* sim,
      const ParallelConfig& pc, float& forward, float& backward) = 0;
  virtual Tensor init_inout(FFModel&, const Tensor&) = 0;
  // Other virtual functions that can be optionally overwritten
  virtual ParallelConfig get_random_parallel_config(const FFModel& ff) const;
  virtual ParallelConfig get_data_parallel_config(const FFModel& ff) const;
  virtual Domain get_input_tensor_shape(const ParallelConfig& pc, int input_idx, int part_idx);
  virtual Domain get_output_tensor_shape(const ParallelConfig& pc, int output_idx, int part_idx);
  virtual Domain get_weight_tensor_shape(const ParallelConfig& pc, int weight_idx, int part_idx);
  // Helper functions
  void prefetch(const FFModel&);
  void zero_grad(const FFModel&);
  Parameter* get_parameter(int index);
public:
  OperatorType op_type;
  char name[MAX_OPNAME];
  IndexSpace task_is;
  Tensor outputs[MAX_NUM_OUTPUTS];
  Tensor inputs[MAX_NUM_INPUTS];
  Parameter weights[MAX_NUM_WEIGHTS];
  //bool trainableInputs[MAX_NUM_INPUTS];
  //bool resetInputGrads[MAX_NUM_INPUTS];
  LogicalPartition input_lps[MAX_NUM_INPUTS], input_grad_lps[MAX_NUM_INPUTS];
  //Tensor locals[MAX_NUM_LOCALS];
  OpMeta* meta[MAX_NUM_WORKERS];
  int numInputs, numWeights, numOutputs;
};

class ElementBinary;
class ElementUnary;
class Conv2D;
class Pool2D;
class Flat;
class Linear;
class Embedding;

class FFModel {
public:
  FFModel(FFConfig &config);
  // C++ APIs for constructing models
  // Add an exp layer
  Tensor exp(const Tensor& x);
  // Add an add layer
  Tensor add(const Tensor& x,
             const Tensor& y);
  // Add a subtract layer
  Tensor subtract(const Tensor& x,
                  const Tensor& y);
  // Add a multiply layer
  Tensor multiply(const Tensor& x,
                  const Tensor& y);
  // Add a divide layer
  Tensor divide(const Tensor& x,
                const Tensor& y);
  // Add an activation layer
  Tensor relu(const Tensor& x);
  Tensor sigmoid(const Tensor& x);
  Tensor tanh(const Tensor& x);
  Tensor elu(const Tensor& x);
  // Add a 2D convolutional layer
  Tensor conv2d(const Tensor& input,
                int outChannels,
                int kernelH, int kernelW,
                int strideH, int strideW,
                int paddingH, int paddingW,
                int groups = 1,
                ActiMode activation = AC_MODE_NONE,
                bool use_bias = true,
                const Op* shared_op = NULL,
                Initializer* krenel_initializer = NULL,
                Initializer* bias_initializer = NULL);
  // Add a dropout layer
  Tensor dropout(const Tensor& input,
                 float rate,
                 unsigned long long seed = 0);
  // Add an embedding layer
  Tensor embedding(const Tensor& input,
                   int num_entires, int outDim,
                   AggrMode aggr,
                   const Op* shared_op = NULL,
                   Initializer* kernel_initializer = NULL);
  // Add a 2D pooling layer
  Tensor pool2d(const Tensor& input,
                int kernelH, int kernelW,
                int strideH, int strideW,
                int paddingH, int paddingW,
                PoolType type = POOL_MAX,
                ActiMode activation = AC_MODE_NONE);
  // Add a batch_norm layer
  Tensor batch_norm(const Tensor& input,
                    bool relu = true);
  // Add a batch_matmul layer
  Tensor batch_matmul(const Tensor& A,
                      const Tensor& B);
  // Add a dense layer
  Tensor dense(const Tensor& input,
               int outDim,
               ActiMode activation = AC_MODE_NONE,
               bool use_bias = true,
               const Op* shared_op = NULL,
               Initializer* kernel_initializer = NULL,
               Initializer* bias_initializer = NULL);
  // Add a concat layer
  Tensor concat(int n, const Tensor* tensors,
                int axis);
  // Add a split layer
  void split(const Tensor& input, Tensor* outputs,
             const std::vector<int>& split, int axis);
  // Add a flat layer
  Tensor flat(const Tensor& input);
  // Add a softmax layer
  Tensor softmax(const Tensor& input);
  // Create input tensors and constants
  Tensor transpose(const Tensor& input,
                   const std::vector<int>& perm);
  Tensor reshape(const Tensor& input,
                 const std::vector<int>& shape);
  Tensor reverse(const Tensor& input,
                 int axis);
  Tensor multihead_attention(const Tensor& query,
                             const Tensor& key,
                             const Tensor& value,
                             int embed_dim,
                             int num_heads,
                             int kdim = 0,
                             int vdim = 0,
                             float dropout = 0.0f,
                             bool bias = true,
                             bool add_bias_kv = false,
                             bool add_zero_attn = false,
                             Initializer* kernel_initializer = NULL);
  template<int NDIM>
  Tensor create_tensor(const int dims[],
                       DataType data_type,
                       const Op* owner_op = NULL,
                       bool create_grad = true);
  template<int NDIM>
  Tensor create_constant(const int dims[],
                         float value,
                         DataType date_type);
  // ========================================
  // Functional APIs for constructing models
  // ========================================
  ElementUnary* exp();
  ElementBinary* add();
  ElementBinary* subtract();
  ElementBinary* multiply();
  ElementBinary* divide();
  ElementUnary* relu();
  ElementUnary* sigmoid();
  ElementUnary* tanh();
  ElementUnary* elu();
  Conv2D* conv2d(int inChannels,
                 int outChannels,
                 int kernelH, int kernelW,
                 int strideH, int strideW,
                 int paddingH, int paddingW,
                 int groups,
                 ActiMode activation = AC_MODE_NONE,
                 bool use_bias = true,
                 Initializer* krenel_initializer = NULL,
                 Initializer* bias_initializer = NULL);
  Embedding* embedding(int num_entires, int outDim,
                       AggrMode aggr,
                       Initializer* kernel_initializer);
  Pool2D* pool2d(int kernelH, int kernelW,
                 int strideH, int strideW,
                 int paddingH, int paddingW,
                 PoolType type = POOL_MAX,
                 ActiMode activation = AC_MODE_NONE);
  Linear* dense(int inDim, int outDim,
                ActiMode activation = AC_MODE_NONE,
                bool use_bias = true,
                Initializer* kernel_initializer = NULL,
                Initializer* bias_initializer = NULL);
  Flat* flat();
  // ========================================
  // Internal APIs that should not be invoked from applications
  // ========================================
  template<int NDIM>
  void create_disjoint_partition(const Tensor& tensor,
                                 const IndexSpaceT<NDIM>& part_is,
                                 LogicalPartition& part_fwd,
                                 LogicalPartition& part_bwd);

  template<int NDIM, int TDIM>
  void create_data_parallel_partition_with_diff_dims(const Tensor& tensor,
                                                     const IndexSpaceT<TDIM>& task_is,
                                                     LogicalPartition& part_fwd,
                                                     LogicalPartition& part_bwd);
  // Deprecated API --- to be removed
  //template<int NDIM>
  //Tensor create_tensor(const int* dims,
  //                     const IndexSpaceT<NDIM>& part_is,
  //                     DataType data_type,
  //                     bool create_grad = true);
  template<int NDIM>
  Parameter create_conv_weight(Op* op,
      const int* dims,
      DataType data_type,
      Initializer* initializer,
      bool create_grad = true,
      Parameter::CommType comm_type = Parameter::PS);
  template<int NDIM, int TDIM>
  Parameter create_linear_weight(Op* op,
      const int* dims,
      DataType data_type,
      Initializer* initializer,
      bool create_grad = true,
      Parameter::CommType comm_type = Parameter::PS);
  template<int NDIM, int TDIM>
  Tensor create_linear_replica(const int* dims,
                               const IndexSpaceT<TDIM>& part_is,
                               DataType data_type);
  static PerfMetrics update_metrics_task(const Task *task,
                                         const std::vector<PhysicalRegion> &regions,
                                         Context ctx, Runtime *runtime);
  void reset_metrics();
  void init_layers();
  void prefetch();
  void forward();
  void compute_metrics();
  void backward();
  void update();
  bool apply_fusion(const std::vector<Op*>& layers, std::vector<Op*>& new_layers);
  void compile(LossType loss_type, const std::vector<MetricsType>& metrics);
  void compile(Optimizer* optimizer, LossType loss_type, const std::vector<MetricsType>& metrics);
  void optimize(Simulator* simulator,
                std::map<Op*, ParallelConfig>& best,
                size_t budget, float alpha) const;
  void rewrite(const std::map<Op*, ParallelConfig>& current,
               std::map<Op*, ParallelConfig>& next) const;
  void zero_gradients();
  void print_layers(int id);
  // Internal funcitons
  Tensor get_tensor_from_guid(int guid);
  IndexSpace get_or_create_task_is(ParallelConfig pc);
  IndexSpace get_or_create_task_is(const Domain& domain);
  IndexSpace get_or_create_task_is(int ndims, const std::string& pcname);
  IndexSpace get_task_is(const Domain& domain) const;
  IndexSpace get_task_is(ParallelConfig pc) const;
  IndexSpace get_task_is(int ndims, const std::string& pcname) const;
public:
  int op_global_guid;
  FFConfig config;
  Optimizer* optimizer;
  Loss* loss_op;
  Metrics* metrics_op;
  Tensor label_tensor;
  //std::vector<Tensor> input_tensors;

  std::vector<Op*> layers;
  std::vector<Parameter> parameters;
  FFHandler handlers[MAX_NUM_WORKERS];
  Future current_metrics;
  //DataLoader *dataLoader;
private:
  std::map<ParallelConfig, IndexSpace, ParaConfigCompare> taskIs;
};

class ElementBinaryMeta : public OpMeta {
public:
  ElementBinaryMeta(FFHandler handle);
  cudnnTensorDescriptor_t inputTensor, outputTensor;
  cudnnOpTensorDescriptor_t opDesc;
  OperatorType op_type;
};

class ElementBinary : public Op {
public:
  ElementBinary(FFModel& model,
                OperatorType type,
                const Tensor& x,
                const Tensor& y);
  ElementBinary(FFModel& model,
                OperatorType type);
  Tensor init_inout(FFModel& model, const Tensor& input);
  //void add_to_model(FFModel& model);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  void print_layer(const FFModel& model) {assert(0);}
  //Parameter* get_parameter(int index) {assert(0); return NULL;}
  void create_weights(FFModel& model);
  void create_output_and_partition(FFModel& model);

  static OpMeta* init_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, HighLevelRuntime *runtime);
  bool measure_compute_time(Simulator* sim,
                            const ParallelConfig& pc,
                            float& forward_time,
                            float& backward_time);
  static void forward_kernel(const ElementBinaryMeta* m,
                      const float* in1_ptr,
                      const float* in2_ptr,
                      float* out_ptr);
  static void backward_kernel(const ElementBinaryMeta* m,
                       const float* out_grad_ptr,
                       const float* in1_ptr,
                       const float* in2_ptr,
                       float* in1_grad_ptr,
                       float* in2_grad_ptr);
private:
  template<int NDIM>
  void create_output_and_partition_with_dim(FFModel& model);
public:
  //IndexSpace task_is;
  OperatorType op_type;
};

class ElementUnaryMeta : public OpMeta {
public:
  ElementUnaryMeta(FFHandler handle);
  cudnnTensorDescriptor_t inputTensor, outputTensor;
  cudnnActivationDescriptor_t actiDesc;
  OperatorType op_type;
};

class ElementUnary : public Op {
public:
  ElementUnary(FFModel& model,
               OperatorType type,
               const Tensor& x);
  ElementUnary(FFModel& model,
               OperatorType type);
  Tensor init_inout(FFModel& model, const Tensor& input);
  //void add_to_model(FFModel& model);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  void print_layer(const FFModel& model) {assert(0);}
  //Parameter* get_parameter(int index) {assert(0); return NULL;}
  void create_weights(FFModel& model);
  void create_output_and_partition(FFModel& model);
  static OpMeta* init_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, HighLevelRuntime *runtime);
  static void forward_kernel(const ElementUnaryMeta* m,
                      const float* in_ptr,
                      float* out_ptr,
                      size_t num_elements);
  static void backward_kernel(const ElementUnaryMeta* m,
                       const float* in_ptr,
                       float* in_grad_ptr,
                       const float* out_ptr,
                       const float* out_grad_ptr,
                       size_t num_elements);
  bool measure_compute_time(Simulator* sim,
                            const ParallelConfig& pc,
                            float& forward_time,
                            float& backward_time);
  static bool use_cudnn(OperatorType type);
private:
  template<int NDIM>
  void create_output_and_partition_with_dim(FFModel& model);
};

class Conv2DMeta : public OpMeta {
public:
  Conv2DMeta(FFHandler handler);
  cudnnTensorDescriptor_t inputTensor, biasTensor, outputTensor;
  cudnnFilterDescriptor_t filterDesc;
  cudnnActivationDescriptor_t actiDesc;
  cudnnConvolutionDescriptor_t convDesc;
  cudnnConvolutionFwdAlgo_t fwdAlgo;
  cudnnConvolutionBwdFilterAlgo_t bwdFilterAlgo;
  cudnnConvolutionBwdDataAlgo_t bwdDataAlgo;
  bool relu;
};

class Conv2D : public Op {
public:
  Conv2D(FFModel& model,
         const Tensor& input,
         int out_dim,
         int kernelH, int kernelW,
         int strideH, int strideW,
         int paddingH, int paddingW,
         int groups,
         ActiMode activation,
         bool use_bias,
         const Op* shared_op,
         Initializer* kernel_initializer,
         Initializer* bias_initializer);
  Conv2D(FFModel& model,
         int in_dim, int out_dim,
         int kernelH, int kernelW,
         int strideH, int strideW,
         int paddingH, int paddingW,
         int groups,
         ActiMode activation,
         bool use_bias,
         Initializer* kernel_initializer,
         Initializer* bias_initializer);
  Tensor init_inout(FFModel& model, const Tensor& input);
  //void add_to_model(FFModel& model);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  //void update(const FFModel&);
  void print_layer(const FFModel& model);
  //Parameter* get_parameter(int index);
  void create_weights(FFModel& model);
  void create_output_and_partition(FFModel& model);


  static OpMeta* init_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, HighLevelRuntime *runtime);
  static void forward_kernel(const Conv2DMeta* m,
                      const float* input_ptr,
                      float* output_ptr,
                      const float* filter_ptr,
                      const float* bias_ptr);
  static void backward_kernel(const Conv2DMeta* m,
                       const float* input_ptr,
                       float* input_grad_ptr,
                       const float* output_ptr,
                       float* output_grad_ptr,
                       const float* kernel_ptr,
                       float* kernel_grad_ptr,
                       float* bias_ptr);
  bool measure_compute_time(Simulator* sim,
                            const ParallelConfig& pc,
                            float& forward_time,
                            float& backward_time);
public:
  //IndexSpaceT<4> task_is;
  int in_channels, out_channels, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, groups;
  bool profiling, use_bias;
  ActiMode activation;
  Initializer *kernel_initializer;
  Initializer *bias_initializer;
};

class DropoutMeta : public OpMeta {
public:
  DropoutMeta(FFHandler handle);
  cudnnTensorDescriptor_t inputTensor, outputTensor;
  cudnnDropoutDescriptor_t dropoutDesc;
  void *reserveSpace, *dropoutStates;
  size_t reserveSpaceSize, dropoutStateSize;
};

class Dropout : public Op {
public:
  Dropout(FFModel& model,
          const Tensor& input,
          float rate,
          unsigned long long seed);
  Tensor init_inout(FFModel& model, const Tensor& input);
  //void add_to_model(FFModel& model);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  void print_layer(const FFModel& model) {assert(0);}
  //Parameter* get_parameter(int index) {assert(0); return NULL;}
  void create_weights(FFModel& model);
  void create_output_and_partition(FFModel& model);

  static OpMeta* init_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime);
  bool measure_compute_time(Simulator* sim,
                            const ParallelConfig& pc,
                            float& forward_time,
                            float& backward_time);
private:
  template<int NDIM>
  void create_output_and_partition_with_dim(FFModel& model);
public:
  //IndexSpaceT<4> task_is;
  float rate;
  unsigned long long seed;
  bool profiling;
};

class Pool2D : public Op {
public:
  Pool2D(FFModel& model,
         const Tensor& input,
         int kernelH, int kernelW,
         int strideH, int strideW,
         int paddingH, int paddingW,
         PoolType type, ActiMode _activation);
  Pool2D(FFModel& model,
         int kernelH, int kernelW,
         int strideH, int strideW,
         int paddingH, int paddingW,
         PoolType type, ActiMode _activation);
  Tensor init_inout(FFModel& model, const Tensor& input);
  //void add_to_model(FFModel& model);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  void update(const FFModel&);
  void print_layer(const FFModel& model) {assert(0);}
  //Parameter* get_parameter(int index) {assert(0); return NULL;}
  void create_weights(FFModel& model);
  void create_output_and_partition(FFModel& model);

  static OpMeta* init_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime);
  static void forward_kernel(const Pool2DMeta* m,
                             const float* input_ptr,
                             float* output_ptr);
  static void backward_kernel(const Pool2DMeta* m,
                              const float* input_ptr,
                              float* input_grad_ptr,
                              const float* output_ptr,
                              const float* output_grad_ptr);
  bool measure_compute_time(Simulator* sim,
                            const ParallelConfig& pc,
                            float& forward_time,
                            float& backward_time);
public:
  //IndexSpaceT<4> task_is;
  int kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w;
  PoolType pool_type;
  ActiMode activation;
  bool profiling;
};

class Pool2DMeta : public OpMeta {
public:
  Pool2DMeta(FFHandler handle);
  cudnnTensorDescriptor_t inputTensor, outputTensor;
  cudnnActivationDescriptor_t actiDesc;
  cudnnPoolingDescriptor_t poolDesc;
  bool relu;
};

class BatchNorm : public Op {
public:
  BatchNorm(FFModel& model, const Tensor& input, bool relu);

  Tensor init_inout(FFModel& model, const Tensor& input) { assert(0); return Tensor();}
  //void add_to_model(FFModel& model) {assert(0);}
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  void update(const FFModel&);
  void print_layer(const FFModel& model) {assert(0);}
  //Parameter* get_parameter(int index) {assert(0);return NULL;}
  void create_weights(FFModel& model);
  void create_output_and_partition(FFModel& model);

  static OpMeta* init_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void init_para_task(const Task *task,
                             const std::vector<PhysicalRegion> &regions,
                             Context ctx, Runtime *runtime);
  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime);
  bool measure_compute_time(Simulator* sim,
                            const ParallelConfig& pc,
                            float& forward_time,
                            float& backward_time);
public:
  //IndexSpaceT<4> task_is;
  bool relu, profiling;
  int num_replica;
  //Tensor locals[MAX_NUM_LOCALS];
};

class BatchNormMeta : public OpMeta {
public:
  BatchNormMeta(FFHandler handle) : OpMeta(handle) {};
  cudnnTensorDescriptor_t inputTensor, outputTensor, biasTensor;
  cudnnActivationDescriptor_t actiDesc;
  cudnnBatchNormMode_t mode;
  float *runningMean, *runningVar, *saveMean, *saveVar;
  bool relu;
};

class LinearMeta : public OpMeta {
public:
  LinearMeta(FFHandler handle, int batch_size);
  cudnnTensorDescriptor_t outputTensor;
  cudnnActivationDescriptor_t actiDesc;
  const float *one_ptr;
  ActiMode activation;
};

class Linear : public Op {
public:
  Linear(FFModel& model,
         const Tensor& input,
         int outChannels,
         ActiMode activation,
         bool use_bias,
         const Op* shared_op,
         Initializer* kernel_initializer,
         Initializer* bias_initializer);
  Linear(FFModel& model,
         int inChannels,
         int outChannels,
         ActiMode activation,
         bool use_bias,
         Initializer* kernel_initializer,
         Initializer* bias_initializer);
  Tensor init_inout(FFModel& model, const Tensor& input);
  //void add_to_model(FFModel& model);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  //void update(const FFModel&);
  void print_layer(const FFModel& model);
  //Parameter* get_parameter(int index);
  void create_weights(FFModel& model);
  void create_output_and_partition(FFModel& model);

  static OpMeta* init_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime);
  static void backward2_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime);
  static void forward_kernel(const LinearMeta* m,
                      const float* input_ptr,
                      float* output_ptr,
                      const float* filter_ptr,
                      const float* bias_ptr,
                      int in_dim, int out_dim, int batch_size);
  static void backward_kernel(const LinearMeta* m,
                       const float* input_ptr,
                       float* input_grad_ptr,
                       const float* output_ptr,
                       float* output_grad_ptr,
                       const float* kernel_ptr,
                       float* kernel_grad_ptr,
                       float* bias_ptr,
                       int in_dim, int out_dim, int batch_size);
  bool measure_compute_time(Simulator* sim,
                            const ParallelConfig& pc,
                            float& forward_time,
                            float& backward_time);
private:
  template<int NDIM>
  void create_output_and_partition_with_dim(FFModel& model);
  template<int NDIM>
  void create_weights_with_dim(FFModel& model);
  template<int NDIM>
  void init_with_dim(const FFModel& ff);
  template<int NDIM>
  void forward_with_dim(const FFModel& ff);
  template<int NDIM>
  void backward_with_dim(const FFModel& ff);
  template<int NDIM>
  static OpMeta* init_task_with_dim(const Task *task,
                                    const std::vector<PhysicalRegion> &regions,
                                    Context ctx, Runtime *runtime);
  template<int NDIM>
  static void forward_task_with_dim(const Task *task,
                                    const std::vector<PhysicalRegion> &regions,
                                    Context ctx, Runtime *runtime);
  template<int NDIM>
  static void backward_task_with_dim(const Task *task,
                                     const std::vector<PhysicalRegion> &regions,
                                     Context ctx, Runtime *runtime);
  template<int NDIM>
  static void backward2_task_with_dim(const Task *task,
                                      const std::vector<PhysicalRegion> &regions,
                                      Context ctx, Runtime *runtime);
public:
  int in_channels, out_channels;
  Tensor replica;
  bool profiling, use_bias;
  ActiMode activation;
  Initializer *kernel_initializer;
  Initializer *bias_initializer;
};

class BatchMatmulMeta : public OpMeta {
public:
  BatchMatmulMeta(FFHandler handler);
};

class BatchMatmul : public Op {
public:
  BatchMatmul(FFModel& model,
              const Tensor& A,
              const Tensor& B);
  Tensor init_inout(FFModel& model, const Tensor& input);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  void print_layer(const FFModel& model);
  void create_weights(FFModel& model);
  void create_output_and_partition(FFModel& model);
  static OpMeta* init_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime);
  static void forward_kernel(const BatchMatmulMeta* meta,
                      float* o_ptr,
                      const float* a_ptr,
                      const float* b_ptr,
                      const float* c_ptr,
                      int m, int n, int k, int batch);
  static void backward_kernel(const BatchMatmulMeta* meta,
                       const float* o_ptr,
                       const float* o_grad_ptr,
                       const float* a_ptr,
                       float* a_grad_ptr,
                       const float* b_ptr,
                       float* b_grad_ptr,
                       float* c_grad_ptr,
                       int m, int n, int k, int batch);
  bool measure_compute_time(Simulator* sim,
                            const ParallelConfig& pc,
                            float& forward_time,
                            float& backward_time);
private:
  template<int NDIM>
  void create_output_and_partition_with_dim(FFModel& model);
  template<int NDIM>
  void init_with_dim(const FFModel& ff);
  template<int NDIM>
  void forward_with_dim(const FFModel& ff);
  template<int NDIM>
  void backward_with_dim(const FFModel& ff);
public:
  bool profiling;
};

class Embedding : public Op {
public:
  Embedding(FFModel& model,
            const Tensor& input,
            int num_entries, int outDim,
            AggrMode _aggr,
            const Op* shared_op,
            Initializer* kernel_initializer);
  Embedding(FFModel& model,
            int num_entries, int outDim,
            AggrMode _aggr,
            Initializer* kernel_initializer);
  Tensor init_inout(FFModel& model, const Tensor& input);
  //void add_to_model(FFModel& model);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  //void update(const FFModel&);
  void print_layer(const FFModel& model) {assert(0);}
  //Parameter* get_parameter(int index);
  void create_weights(FFModel& model);
  void create_output_and_partition(FFModel& model);

  static OpMeta* init_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime);
  static void forward_task_cpu(const Task *task,
                               const std::vector<PhysicalRegion> &regions,
                               Context ctx, Runtime *runtime);
  static void backward_task_cpu(const Task *task,
                                const std::vector<PhysicalRegion> &regions,
                                Context ctx, Runtime *runtime);
  bool measure_compute_time(Simulator* sim,
                            const ParallelConfig& pc,
                            float& forward_time,
                            float& backward_time);
public:
  //IndexSpaceT<2> task_is;
  int num_entries, out_channels;
  AggrMode aggr;
  bool profiling;
  Initializer* kernel_initializer;
};


class Flat : public Op {
public:
  Flat(FFModel& model,
       const Tensor& input);
  Flat(FFModel& model);
  Tensor init_inout(FFModel& model, const Tensor& input);
  //void add_to_model(FFModel& model);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  //void update(const FFModel&);
  void print_layer(const FFModel& model) {assert(0);}
  //Parameter* get_parameter(int index) {return NULL;}
  void create_weights(FFModel& model);
  void create_output_and_partition(FFModel& model);

  static OpMeta* init_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime);
  static void forward_kernel(const float* input_ptr,
                             float* output_ptr,
                             size_t num_elements);
  static void backward_kernel(float* input_grad_ptr,
                              const float* output_grad_ptr,
                              size_t num_elements);
  bool measure_compute_time(Simulator* sim,
                            const ParallelConfig& pc,
                            float& forward_time,
                            float& backward_time);

  Domain get_input_tensor_shape(const ParallelConfig& pc, int input_idx, int part_idx);
public:
  //IndexSpaceT<2> task_is;
};

class FlatMeta : public OpMeta {
public:
  FlatMeta(FFHandler handle) : OpMeta(handle) {};
};

class MultiHeadAttentionMeta;
class MultiHeadAttention : public Op {
public:
  MultiHeadAttention(FFModel& model,
                     const Tensor& _query,
                     const Tensor& _key,
                     const Tensor& _value,
                     int _embed_dim, int _num_heads,
                     int _kdim, int _vdim,
                     float _dropout, bool _bias,
                     bool _add_bias_kv, bool _add_zero_attn,
                     Initializer* _kernel_initializer);
  Tensor init_inout(FFModel& model, const Tensor& input) { assert(0); return Tensor();}
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  void print_layer(const FFModel& model) {assert(0);}
  void create_weights(FFModel& model);
  void create_output_and_partition(FFModel& model);

  static OpMeta* init_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime);
  bool measure_compute_time(Simulator* sim,
                            const ParallelConfig& pc,
                            float& forward_time,
                            float& backward_time);
  void forward_kernel(const MultiHeadAttentionMeta* m,
                      const float* query_ptr,
                      const float* key_ptr,
                      const float* value_ptr,
                      const float* weight_ptr,
                      float* output_ptr) const;
  void backward_kernel(const MultiHeadAttentionMeta* m,
                       const float* query_ptr,
                       float* query_grad_ptr,
                       const float* key_ptr,
                       float* key_grad_ptr,
                       const float* value_ptr,
                       float* value_grad_ptr,
                       const float* weight_ptr,
                       float* weight_grad_ptr,
                       const float* output_grad_ptr) const;
public:
  int qSize, kSize, vSize, qProjSize, kProjSize, vProjSize, oProjSize;
  int qoSeqLength, kvSeqLength;
  Initializer* kernel_initializer;
  float dropout;
  bool profiling, bias, add_bias_kv, add_zero_attn;
};

class MultiHeadAttentionMeta : public OpMeta {
public:
  MultiHeadAttentionMeta(FFHandler handler,
                         const MultiHeadAttention* attn,
                         Memory gpu_mem,
                         int num_samples,
                         int num_heads);
  ~MultiHeadAttentionMeta(void);
public:
  Realm::RegionInstance reserveInst;
  size_t weightSize, reserveSpaceSize;
  cudnnAttnDescriptor_t attnDesc;
  cudnnSeqDataDescriptor_t qDesc, kDesc, vDesc, oDesc;
  int *devQoSeqArray, *devKvSeqArray, *loWinIdx, *hiWinIdx;
  void *reserveSpace;
};

class Softmax : public Op {
public:
  Softmax(FFModel& model,
          const Tensor& logit);
  Tensor init_inout(FFModel& model, const Tensor& input) {assert(0); return Tensor();}
  //void add_to_model(FFModel& model) {assert(0);}
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  //void update(const FFModel&);
  void print_layer(const FFModel& model) {assert(0);}
  //Parameter* get_parameter(int index) {assert(0); return NULL;}
  void create_weights(FFModel& model);
  void create_output_and_partition(FFModel& model);

  static OpMeta* init_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime);
  bool measure_compute_time(Simulator* sim,
                            const ParallelConfig& pc,
                            float& forward_time,
                            float& backward_time);
public:
  //IndexSpaceT<2> task_is;
  bool profiling;
};

class SoftmaxMeta : public OpMeta {
public:
  SoftmaxMeta(FFHandler handle) : OpMeta(handle) {};
#ifndef DISABLE_COMPUTATION
  cudnnTensorDescriptor_t inputTensor;
#endif
};

class TransposeMeta : public OpMeta {
public:
  TransposeMeta(FFHandler handler) : OpMeta(handler) {};
  int num_dim;
  int perm[MAX_TENSOR_DIM];
};

class Transpose : public Op {
public:
  Transpose(FFModel& model,
            const Tensor& input,
            const std::vector<int>& perm);
  Tensor init_inout(FFModel& model, const Tensor& input);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  void print_layer(const FFModel& model) {assert(0);}
  void create_weights(FFModel& model);
  void create_output_and_partition(FFModel& model);

  static OpMeta* init_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime);
  static void forward_kernel(const TransposeMeta* m,
                             const float* input_ptr,
                             float* output_ptr,
                             Domain in_domain,
                             Domain out_domain);
  static void backward_kernel(const TransposeMeta* m,
                              float* input_grad_ptr,
                              const float* output_grad_ptr,
                              Domain in_grad_domain,
                              Domain out_grad_domain);
  bool measure_compute_time(Simulator* sim,
                            const ParallelConfig& pc,
                            float& forward_time,
                            float& backward_time);
private:
  template<int NDIM>
  void create_output_and_partition_with_dim(FFModel& model);
public:
  int perm[MAX_TENSOR_DIM];
};

class Reverse : public Op {
public:
  Reverse(FFModel& model,
          const Tensor& input,
          int axis);
  Tensor init_inout(FFModel& model, const Tensor& input);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  void print_layer(const FFModel& model) {assert(0);}
  void create_weights(FFModel& model);
  void create_output_and_partition(FFModel& model);

  static OpMeta* init_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime);
  bool measure_compute_time(Simulator* sim,
                            const ParallelConfig& pc,
                            float& forward_time,
                            float& backward_time);
private:
  template<int NDIM>
  void create_output_and_partition_with_dim(FFModel& model);
public:
  int axis;
};

class Reshape : public Op {
public:
  Reshape(FFModel& model,
            const Tensor& input,
            const std::vector<int>& shape);
  Tensor init_inout(FFModel& model, const Tensor& input){assert(0); return Tensor();}
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  void print_layer(const FFModel& model) {assert(0);}
  void create_weights(FFModel& model);
  void create_output_and_partition(FFModel& model);

  static OpMeta* init_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime);
  static void forward_kernel(const float* input_ptr,
                             float* output_ptr,
                             size_t num_elements);
  static void backward_kernel(float* input_grad_ptr,
                              const float* output_grad_ptr,
                              size_t num_elements);
  bool measure_compute_time(Simulator* sim,
                            const ParallelConfig& pc,
                            float& forward_time,
                            float& backward_time);
private:
  template<int IDIM, int ODIM>
  void create_output_and_partition_with_dim(FFModel& model);
};

class ConcatMeta : public OpMeta {
public:
  ConcatMeta(FFHandler handle) : OpMeta(handle) {};
  int axis;
};

class Concat : public Op {
public:
  Concat(FFModel& model,
         int n, const Tensor* inputs, int axis);
  Tensor init_inout(FFModel& model, const Tensor& input) {assert(0); return Tensor();}
  //void add_to_model(FFModel& model) {assert(0);}
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  //void update(const FFModel&);
  void print_layer(const FFModel& model) {assert(0);}
  //Parameter* get_parameter(int index) {assert(0); return NULL;}
  void create_weights(FFModel& model);
  void create_output_and_partition(FFModel& model);

  static OpMeta* init_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime);
  static void forward_kernel(float* output,
                             const float** inputs,
                             int num_inputs,
                             int axis,
                             const Domain& out_domain,
                             const Domain* in_domain);
  static void backward_kernel(const float* output_grad,
                              float** input_grads,
                              int num_inputs,
                              int axis,
                              const Domain& out_grad_domain,
                              const Domain* in_grad_domain);
  bool measure_compute_time(Simulator* sim,
                            const ParallelConfig& pc,
                            float& forward_time,
                            float& backward_time);
public:
  int axis;
  //IndexSpace task_is;
  bool profiling;
};

class Split : public Op {
public:
  Split(FFModel& model,
        const Tensor& input,
        const std::vector<int>& split,
        int axis);
  Tensor init_inout(FFModel& model, const Tensor& input) {assert(0); return Tensor();}
  //void add_to_model(FFModel& model) {assert(0);}
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  //void update(const FFModel&);
  void print_layer(const FFModel& model) {assert(0);}
  //Parameter* get_parameter(int index) {assert(0); return NULL;}
  void create_weights(FFModel& model);
  void create_output_and_partition(FFModel& model);

  static OpMeta* init_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime);
  bool measure_compute_time(Simulator* sim,
                            const ParallelConfig& pc,
                            float& forward_time,
                            float& backward_time);
private:
  template<int NDIM>
  void create_output_and_partition_with_dim(FFModel& model);
public:
  int axis;
  //IndexSpace task_is;
  bool profiling;
};

class FusedOpMeta {
public:
  FusedOpMeta(void) {}
  OpMeta* meta[MAX_NUM_FUSED_OPERATORS];
  int numOperators;
};

class FusedOp : public Op {
public:
  enum SourceType {
    SOURCE_NONE,
    SOURCE_INPUT,
    SOURCE_WEIGHT,
    SOURCE_OUTPUT,
  };
  FusedOp(FFModel& model,
          Op* op);
  bool add_operator(FFModel& model, Op* op);
  Tensor init_inout(FFModel& model, const Tensor& input) {assert(0); return Tensor();}
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  void print_layer(const FFModel& model) {assert(0);}
  void create_weights(FFModel& model);
  void create_output_and_partition(FFModel& model);
  static OpMeta* init_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime);
  bool measure_compute_time(Simulator* sim,
                            const ParallelConfig& pc,
                            float& forward_time,
                            float& backward_time);
public:
  int op_num_inputs[MAX_NUM_FUSED_OPERATORS];
  int op_num_weights[MAX_NUM_FUSED_OPERATORS];
  int op_num_outputs[MAX_NUM_FUSED_OPERATORS];
  OperatorType op_op_type[MAX_NUM_FUSED_OPERATORS];
  SourceType op_input_source[MAX_NUM_FUSED_TENSORS];
  SourceType op_weight_source[MAX_NUM_FUSED_TENSORS];
  SourceType op_output_source[MAX_NUM_FUSED_TENSORS];
  int op_input_idx[MAX_NUM_FUSED_TENSORS];
  int op_weight_idx[MAX_NUM_FUSED_TENSORS];
  int op_output_idx[MAX_NUM_FUSED_TENSORS];
  Op* operators[MAX_NUM_FUSED_OPERATORS];
  FusedOpMeta fused_meta[MAX_NUM_WORKERS];
  int numOperators;
};

class UtilityTasks {
public:
  static FFHandler init_cuda_task(const Task *task,
                                  const std::vector<PhysicalRegion> &regions,
                                  Context ctx, Runtime *runtime);
  static void dummy_task(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx, Runtime *runtime);
  static void init_images_task(const Task *task,
                               const std::vector<PhysicalRegion> &regions,
                               Context ctx, Runtime *runtime);
  static void init_labels_task(const Task *task,
                               const std::vector<PhysicalRegion> &regions,
                               Context ctx, Runtime *runtime);
  static void load_images_task(const Task *task,
                               const std::vector<PhysicalRegion> &regions,
                               Context ctx, Runtime *runtime);
  static void normalize_images_task(const Task *task,
                                    const std::vector<PhysicalRegion> &regions,
                                    Context ctx, Runtime *runtime);
};

void top_level_task(const Task* task,
                    const std::vector<PhysicalRegion>& regions,
                    Context ctx, Runtime* runtime);

void data_load_task(const Task* task,
                    const std::vector<PhysicalRegion>& regions,
                    Context ctx, Runtime* runtime);

void register_custom_tasks();
void register_c_custom_tasks();
#endif//_FLEXFLOW_MODEL_H_
