#include "pcg/parallel_computation_graph/explicit_parallel_computation_graph_builder.h"
#include "op-attrs/pcg_operator_attrs.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "utils/containers/require_only_key.h"

namespace FlexFlow {

static std::string get_default_name(OperatorType op_type) {
  return get_operator_type_name(op_type);
}

static std::string get_default_name(PCGOperatorAttrs const &attrs) {
  return get_default_name(pcg_op_attrs_get_op_type(attrs));
}

ExplicitParallelComputationGraphBuilder::
    ExplicitParallelComputationGraphBuilder()
    : pcg(empty_parallel_computation_graph()) {}

parallel_tensor_guid_t
    ExplicitParallelComputationGraphBuilder::create_input_tensor(
        TensorShape const &shape, std::optional<std::string> const &name) {

  ParallelLayerAttrs layer_attrs = ParallelLayerAttrs{
      PCGOperatorAttrs{InputAttrs{shape}},
      name,
  };

  return require_only_key(
      add_parallel_layer(this->pcg,
                         layer_attrs,
                         {},
                         {},
                         std::map<TensorSlotName, CreateGrad>{
                             {
                                 TensorSlotName::OUTPUT,
                                 CreateGrad::NO,
                             },
                         })
          .outputs,
      TensorSlotName::OUTPUT);
}

parallel_tensor_guid_t
    ExplicitParallelComputationGraphBuilder::create_weight_tensor(
        TensorShape const &shape,
        InitializerAttrs const &initializer_attrs,
        std::optional<std::string> const &name) {

  ParallelLayerAttrs layer_attrs = ParallelLayerAttrs{
      PCGOperatorAttrs{WeightAttrs{shape, initializer_attrs}},
      name,
  };

  return require_only_key(
      add_parallel_layer(this->pcg,
                         layer_attrs,
                         {},
                         {},
                         std::map<TensorSlotName, CreateGrad>{
                             {
                                 TensorSlotName::OUTPUT,
                                 CreateGrad::YES,
                             },
                         })
          .outputs,
      TensorSlotName::OUTPUT);
}

parallel_tensor_guid_t ExplicitParallelComputationGraphBuilder::add(
    parallel_tensor_guid_t const &lhs,
    parallel_tensor_guid_t const &rhs,
    std::optional<std::string> const &maybe_name) {

  ParallelTensorShape lhs_shape = this->get_shape(lhs);
  ParallelTensorShape rhs_shape = this->get_shape(rhs);

  DataType datatype = [&] {
    if (lhs_shape.data_type != rhs_shape.data_type) {
      throw mk_runtime_error(
          fmt::format("Datatypes do not match: {} (lhs) != {} (rhs)",
                      lhs_shape.data_type,
                      rhs_shape.data_type));
    } else {
      return lhs_shape.data_type;
    }
  }();

  ElementBinaryAttrs attrs = ElementBinaryAttrs{
      OperatorType::EW_ADD,
      datatype,
      false,
      false,
  };

  std::string name =
      maybe_name.value_or(get_default_name(PCGOperatorAttrs{attrs}));

  ParallelLayerAttrs layer = ParallelLayerAttrs{PCGOperatorAttrs{attrs}, name};

  return require_only_key(this->add_layer(layer,
                                          {
                                              {
                                                  TensorSlotName::LHS_INPUT,
                                                  lhs,
                                              },
                                              {
                                                  TensorSlotName::RHS_INPUT,
                                                  rhs,
                                              },
                                          },
                                          {}),
                          TensorSlotName::OUTPUT);
}

parallel_tensor_guid_t ExplicitParallelComputationGraphBuilder::conv2d(
    parallel_tensor_guid_t const &raw_input,
    positive_int outChannels,
    positive_int kernelH,
    positive_int kernelW,
    positive_int strideH,
    positive_int strideW,
    nonnegative_int paddingH,
    nonnegative_int paddingW,
    parallel_tensor_guid_t const &filter,
    std::optional<parallel_tensor_guid_t> const &bias,
    std::optional<Activation> const &activation,
    positive_int groups,
    std::optional<RegularizerAttrs> const &filter_regularizer,
    std::optional<std::string> const &maybe_name) {

  bool use_bias = bias.has_value();

  Conv2DAttrs attrs = Conv2DAttrs{
      /*out_channels=*/outChannels,
      /*kernel_h=*/kernelH,
      /*kernel_w=*/kernelW,
      /*stride_h=*/strideH,
      /*stride_w=*/strideW,
      /*padding_h=*/paddingH,
      /*padding_w=*/paddingW,
      /*groups=*/groups,
      /*activation=*/activation,
      /*use_bias=*/use_bias,
  };

  std::string name =
      maybe_name.value_or(get_default_name(PCGOperatorAttrs{attrs}));

  parallel_tensor_guid_t input = raw_input;

  ParallelLayerAttrs layer = ParallelLayerAttrs{PCGOperatorAttrs{attrs}, name};

  ParallelTensorShape input_shape = this->get_shape(input);

  std::map<TensorSlotName, parallel_tensor_guid_t> weights = {
      {
          TensorSlotName::FILTER,
          filter,
      },
  };

  if (use_bias) {
    weights.insert({TensorSlotName::BIAS, bias.value()});
  }

  return require_only_key(this->add_layer(layer,
                                          {
                                              {
                                                  TensorSlotName::INPUT,
                                                  input,
                                              },
                                          },
                                          weights),
                          TensorSlotName::OUTPUT);
}

parallel_tensor_guid_t ExplicitParallelComputationGraphBuilder::dense(
    parallel_tensor_guid_t const &input,
    positive_int outDim,
    parallel_tensor_guid_t const &projector,
    std::optional<parallel_tensor_guid_t> const &bias,
    std::optional<Activation> activation,
    DataType data_type,
    std::optional<std::string> const &maybe_name) {

  bool use_bias = bias.has_value();

  LinearAttrs attrs = LinearAttrs{
      /*out_channels=*/outDim,
      /*use_bias=*/use_bias,
      /*data_type=*/data_type,
      /*activation=*/activation,
      /*regularizer=*/std::nullopt,
  };

  std::string name =
      maybe_name.value_or(get_default_name(PCGOperatorAttrs{attrs}));

  ParallelLayerAttrs layer = ParallelLayerAttrs{PCGOperatorAttrs{attrs}, name};

  ParallelTensorShape input_shape = this->get_shape(input);

  std::map<TensorSlotName, parallel_tensor_guid_t> weights = {
      {
          TensorSlotName::WEIGHT,
          projector,
      },
  };

  if (use_bias) {
    weights.insert({TensorSlotName::BIAS, bias.value()});
  }

  return require_only_key(this->add_layer(layer,
                                          {
                                              {
                                                  TensorSlotName::INPUT,
                                                  input,
                                              },
                                          },
                                          weights),
                          TensorSlotName::OUTPUT);
}

parallel_tensor_guid_t
    ExplicitParallelComputationGraphBuilder::parallel_partition(
        parallel_tensor_guid_t const &input,
        ff_dim_t dim,
        int_ge_two degree,
        std::optional<std::string> const &maybe_name) {

  RepartitionAttrs attrs = RepartitionAttrs{
      /*repartition_dim=*/dim,
      /*repartition_degree=*/degree,
  };

  std::string name =
      maybe_name.value_or(get_default_name(PCGOperatorAttrs{attrs}));

  ParallelLayerAttrs layer = ParallelLayerAttrs{PCGOperatorAttrs{attrs}, name};

  return require_only_key(this->add_layer(layer,
                                          {
                                              {
                                                  TensorSlotName::INPUT,
                                                  input,
                                              },
                                          },
                                          {}),
                          TensorSlotName::OUTPUT);
}

parallel_tensor_guid_t
    ExplicitParallelComputationGraphBuilder::parallel_replicate(
        parallel_tensor_guid_t const &input,
        int_ge_two degree,
        std::optional<std::string> const &maybe_name) {

  ReplicateAttrs attrs = ReplicateAttrs{degree};

  std::string name =
      maybe_name.value_or(get_default_name(PCGOperatorAttrs{attrs}));

  ParallelLayerAttrs layer = ParallelLayerAttrs{PCGOperatorAttrs{attrs}, name};

  return require_only_key(this->add_layer(layer,
                                          {
                                              {
                                                  TensorSlotName::INPUT,
                                                  input,
                                              },
                                          },
                                          {}),
                          TensorSlotName::OUTPUT);
}

ParallelTensorShape ExplicitParallelComputationGraphBuilder::get_shape(
    parallel_tensor_guid_t const &t) const {
  return get_parallel_tensor_attrs(this->pcg, t).shape;
}

std::map<TensorSlotName, parallel_tensor_guid_t>
    ExplicitParallelComputationGraphBuilder::add_layer(
        ParallelLayerAttrs const &layer,
        std::map<TensorSlotName, parallel_tensor_guid_t> const &inputs,
        std::map<TensorSlotName, parallel_tensor_guid_t> const &weights) {

  ASSERT(are_disjoint(keys(inputs), keys(weights)));

  return add_parallel_layer(this->pcg, layer, inputs, weights, {}).outputs;
}

} // namespace FlexFlow
