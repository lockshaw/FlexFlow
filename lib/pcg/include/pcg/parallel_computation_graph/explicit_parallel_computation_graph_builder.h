#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_PARALLEL_COMPUTATION_GRAPH_EXPLICIT_PARALLEL_COMPUTATION_GRAPH_BUILDER_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_PARALLEL_COMPUTATION_GRAPH_EXPLICIT_PARALLEL_COMPUTATION_GRAPH_BUILDER_H

#include "op-attrs/activation.dtg.h"
#include "op-attrs/ff_dim_t.dtg.h"
#include "op-attrs/initializer_attrs.dtg.h"
#include "op-attrs/ops/element_unary_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/regularizer_attrs.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.dtg.h"
#include "pcg/parallel_computation_graph/parallel_tensor_guid_t.dtg.h"
#include "utils/positive_int/positive_int.h"

namespace FlexFlow {

struct ExplicitParallelComputationGraphBuilder {
public:
  ExplicitParallelComputationGraphBuilder();

  parallel_tensor_guid_t create_input_tensor(
      TensorShape const &shape,
      std::optional<std::string> const &name = std::nullopt);

  parallel_tensor_guid_t create_weight_tensor(
      TensorShape const &shape,
      InitializerAttrs const &initializer,
      std::optional<std::string> const &name = std::nullopt);

  parallel_tensor_guid_t
      add(parallel_tensor_guid_t const &lhs,
          parallel_tensor_guid_t const &rhs,
          std::optional<std::string> const &name = std::nullopt);

  parallel_tensor_guid_t conv2d(
      parallel_tensor_guid_t const &input,
      positive_int outChannels,
      positive_int kernelH,
      positive_int kernelW,
      positive_int strideH,
      positive_int strideW,
      nonnegative_int paddingH,
      nonnegative_int paddingW,
      parallel_tensor_guid_t const &kernel,
      std::optional<parallel_tensor_guid_t> const &bias,
      std::optional<Activation> const &activation = std::nullopt,
      positive_int groups = 1_p,
      std::optional<RegularizerAttrs> const &kernel_regularizer = std::nullopt,
      std::optional<std::string> const &name = std::nullopt);

  parallel_tensor_guid_t
      dense(parallel_tensor_guid_t const &input,
            positive_int outDim,
            parallel_tensor_guid_t const &projector,
            std::optional<parallel_tensor_guid_t> const &bias,
            std::optional<Activation> activation = std::nullopt,
            DataType data_type = DataType::FLOAT,
            std::optional<std::string> const &name = std::nullopt);

  parallel_tensor_guid_t
      relu(parallel_tensor_guid_t const &x,
           std::optional<std::string> const &name = std::nullopt);

  parallel_tensor_guid_t
      identity(parallel_tensor_guid_t const &x,
               std::optional<std::string> const &name = std::nullopt);

  parallel_tensor_guid_t
      gelu(parallel_tensor_guid_t const &x,
           std::optional<std::string> const &name = std::nullopt);

  parallel_tensor_guid_t
      sigmoid(parallel_tensor_guid_t const &x,
              std::optional<std::string> const &name = std::nullopt);

  parallel_tensor_guid_t
      tanh(parallel_tensor_guid_t const &x,
           std::optional<std::string> const &name = std::nullopt);

  parallel_tensor_guid_t
      elu(parallel_tensor_guid_t const &x,
          std::optional<std::string> const &name = std::nullopt);

  parallel_tensor_guid_t
      parallel_partition(parallel_tensor_guid_t const &input,
                         ff_dim_t dim,
                         int_ge_two degree,
                         std::optional<std::string> const &name = std::nullopt);

  parallel_tensor_guid_t
      parallel_combine(parallel_tensor_guid_t const &x,
                       ff_dim_t dim,
                       int_ge_two degree,
                       std::optional<std::string> const &name = std::nullopt);

  parallel_tensor_guid_t
      parallel_replicate(parallel_tensor_guid_t const &x,
                         int_ge_two degree,
                         std::optional<std::string> const &name = std::nullopt);

  parallel_tensor_guid_t
      parallel_reduce(parallel_tensor_guid_t const &x,
                      int_ge_two degree,
                      std::optional<std::string> const &name = std::nullopt);

  ParallelTensorShape get_shape(parallel_tensor_guid_t const &) const;

private:
  std::map<TensorSlotName, parallel_tensor_guid_t> add_layer(
      ParallelLayerAttrs const &layer,
      std::map<TensorSlotName, parallel_tensor_guid_t> const &inputs,
      std::map<TensorSlotName, parallel_tensor_guid_t> const &weights);

  parallel_tensor_guid_t
      element_unary(ElementUnaryAttrs const &element_unary_attrs,
                    parallel_tensor_guid_t const &input,
                    std::optional<std::string> const &name);

public:
  ParallelComputationGraph pcg;
};

} // namespace FlexFlow

#endif
