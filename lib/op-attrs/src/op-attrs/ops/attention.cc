#include "op-attrs/ops/attention.h"
#include "op-attrs/ops/attention/multihead_attention_inputs.h"
#include "op-attrs/ops/attention/multihead_attention_parallel_inputs.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_dims.h"
#include "op-attrs/tensor_shape.h"
#include "utils/containers/extend.h"
#include "utils/expected.h"
#include "utils/integer_conversions.h"
#include <libassert/assert.hpp>
#include "utils/not_implemented.h"
#include "op-attrs/operator_task_space.h"
#include "op-attrs/parallel_tensor_space_to_parallel_tensor_space_biunique_mapping.dtg.h"
#include "op-attrs/parallel_tensor_dim_idx_t.h"
#include "op-attrs/parallel_tensor_space_to_parallel_tensor_space_biunique_mapping.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_biunique_mapping.h"
#include "op-attrs/standard_operator_task_group.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "utils/containers/transform.h"
#include "op-attrs/parallel_tensor_space_coordinate.h"
#include "utils/orthotope/bounded_component.h"
#include "utils/orthotope/orthotope_bounded_coord.h"
#include "op-attrs/task_space_coordinate.h"

namespace FlexFlow {

static void check_attrs(MultiHeadAttentionAttrs const &attrs) {
  ASSERT(!attrs.add_bias_kv,
         "add_bias_kv is not yet supported. If you need this "
         "functionality, please create an issue.");
}

std::map<TensorSlotName, IncomingTensorRole>
    get_attention_incoming_tensor_roles(MultiHeadAttentionAttrs const &attrs) {

  check_attrs(attrs);

  std::map<TensorSlotName, IncomingTensorRole> roles = {
      {TensorSlotName::QUERY, IncomingTensorRole::INPUT},
      {TensorSlotName::KEY, IncomingTensorRole::INPUT},
      {TensorSlotName::VALUE, IncomingTensorRole::INPUT},
      {TensorSlotName::WEIGHT, IncomingTensorRole::WEIGHT},
  };

  if (attrs.bias) {
    roles[TensorSlotName::INPUT_BIAS] = IncomingTensorRole::WEIGHT;
    roles[TensorSlotName::OUTPUT_BIAS] = IncomingTensorRole::WEIGHT;
  }

  return roles;
}

std::set<TensorSlotName> attention_get_slots(MultiHeadAttentionAttrs const &attrs) {
  std::set<TensorSlotName> result = {
      TensorSlotName::QUERY,
      TensorSlotName::KEY,
      TensorSlotName::VALUE,
      TensorSlotName::WEIGHT,
  };

  if (attrs.bias) {
    result.insert(TensorSlotName::INPUT_BIAS);
    result.insert(TensorSlotName::OUTPUT_BIAS);
  }

  return result;
}

TensorShape
    attention_get_output_shape(MultiHeadAttentionAttrs const &attrs,
                     TensorShape const &input_q,
                     TensorShape const &input_k,
                     TensorShape const &input_v) {
  check_attrs(attrs);

  MultiHeadAttentionInputs parsed =
      parse_attention_input_shape(input_q, input_k, input_v);

  return TensorShape{
      TensorDims{FFOrdered<positive_int>{
          parsed.batch_size,
          parsed.sequence_length,
          attrs.embed_dim,
      }},
      parsed.datatype,
  };
}

TensorShape
    attention_get_weights_shape(MultiHeadAttentionAttrs const &attrs,
                      TensorShape const &input_q,
                      TensorShape const &input_k,
                      TensorShape const &input_v) {
  check_attrs(attrs);

  MultiHeadAttentionInputs parsed =
      parse_attention_input_shape(input_q, input_k, input_v);

  // W^Q_i in "Attention Is All You Need" top of page 5
  positive_int qProjectWeightSize = parsed.query_size * attrs.kdim;

  // W^K_i in "Attention Is All You Need" top of page 5 (all i's put together)
  positive_int kProjectWeightSize = parsed.key_size * attrs.kdim;

  // W^V_i in "Attention Is All You Need" top of page 5 (all i's put together)
  positive_int vProjectWeightSize = parsed.value_size * attrs.vdim;

  // W^O in "Attention Is All You Need" top of page 5, with num_heads factored
  // out
  positive_int outWeightSize = attrs.vdim * attrs.embed_dim;

  return TensorShape{
      TensorDims{FFOrdered<positive_int>{
          (qProjectWeightSize + kProjectWeightSize + vProjectWeightSize +
           outWeightSize),
          attrs.num_heads,
      }},
      parsed.datatype,
  };
}

TensorShape
    attention_get_input_bias_shape(MultiHeadAttentionAttrs const &attrs,
                         TensorShape const &input_q,
                         TensorShape const &input_k,
                         TensorShape const &input_v) {
  check_attrs(attrs);

  MultiHeadAttentionInputs parsed =
        parse_attention_input_shape(input_q, input_k, input_v);

  return TensorShape{
      TensorDims{FFOrdered<positive_int>{
          attrs.kdim + attrs.kdim + attrs.vdim,
      }},
      parsed.datatype,
  };
}

TensorShape
    attention_get_output_bias_shape(MultiHeadAttentionAttrs const &attrs,
                          TensorShape const &input_q,
                          TensorShape const &input_k,
                          TensorShape const &input_v) {
  check_attrs(attrs);

  MultiHeadAttentionInputs parsed =
        parse_attention_input_shape(input_q, input_k, input_v);

  return TensorShape{
      TensorDims{FFOrdered<positive_int>{
          attrs.embed_dim,
      }},
      parsed.datatype,
  };
}

std::map<TensorSlotName, TensorShape>
    attention_get_weight_shapes(MultiHeadAttentionAttrs const &attrs,
                      TensorShape const &input_q,
                      TensorShape const &input_k,
                      TensorShape const &input_v) {

  std::map<TensorSlotName, TensorShape> weight_shapes = {
      {
          TensorSlotName::WEIGHT,
          attention_get_weights_shape(attrs, input_q, input_k, input_v),
      },
  };

  if (attrs.bias) {
    weight_shapes.insert({
        TensorSlotName::INPUT_BIAS,
        attention_get_input_bias_shape(attrs, input_q, input_k, input_v),
    });

    weight_shapes.insert({
        TensorSlotName::OUTPUT_BIAS,
        attention_get_output_bias_shape(attrs, input_q, input_k, input_v),
    });
  }

  return weight_shapes;
}

ParallelTensorShape
    attention_get_weights_parallel_shape(MultiHeadAttentionAttrs const &attrs,
                                         ParallelTensorShape const &input_q,
                                         ParallelTensorShape const &input_k,
                                         ParallelTensorShape const &input_v) {

  TensorShape unpar_shape =
      attention_get_weights_shape(attrs,
                        get_reduced_shape(input_q),
                        get_reduced_shape(input_k),
                        get_reduced_shape(input_v));

  ParallelTensorDimDegrees degrees =
      attention_get_weights_parallel_dim_degrees(
        attrs,
        get_parallel_degrees(input_q),
        get_parallel_degrees(input_k),
        get_parallel_degrees(input_v));

  return lift_shape_to_parallel_with_degrees(unpar_shape, degrees);
}

ParallelTensorShape
    attention_get_input_bias_parallel_shape(MultiHeadAttentionAttrs const &attrs,
                         ParallelTensorShape const &input_q,
                         ParallelTensorShape const &input_k,
                         ParallelTensorShape const &input_v)
{
  TensorShape unpar_shape =
        attention_get_input_bias_shape(attrs,
                             get_reduced_shape(input_q),
                             get_reduced_shape(input_k),
                             get_reduced_shape(input_v));

  ParallelTensorDimDegrees degrees =
      attention_get_input_bias_parallel_dim_degrees(
        attrs,
        get_parallel_degrees(input_q),
        get_parallel_degrees(input_k),
        get_parallel_degrees(input_v));

  return lift_shape_to_parallel_with_degrees(unpar_shape, degrees);
}

ParallelTensorShape
    attention_get_output_bias_parallel_shape(MultiHeadAttentionAttrs const &attrs,
                          ParallelTensorShape const &input_q,
                          ParallelTensorShape const &input_k,
                          ParallelTensorShape const &input_v) {
  TensorShape unpar_shape =
        attention_get_output_bias_shape(attrs,
                              get_reduced_shape(input_q),
                              get_reduced_shape(input_k),
                              get_reduced_shape(input_v));

  ParallelTensorDimDegrees degrees =
      attention_get_output_bias_parallel_dim_degrees(
        attrs,
        get_parallel_degrees(input_q),
        get_parallel_degrees(input_k),
        get_parallel_degrees(input_v));

  return lift_shape_to_parallel_with_degrees(unpar_shape, degrees);
}

ParallelTensorShape
    attention_get_output_parallel_shape(MultiHeadAttentionAttrs const &attrs,
                     ParallelTensorShape const &input_q,
                     ParallelTensorShape const &input_k,
                     ParallelTensorShape const &input_v) {

  TensorShape unpar_shape =
      attention_get_output_shape(attrs,
                       get_reduced_shape(input_q),
                       get_reduced_shape(input_k),
                       get_reduced_shape(input_v));

  ParallelTensorDimDegrees degrees =
    attention_get_output_parallel_dim_degrees(
      attrs,
      get_parallel_degrees(input_q),
      get_parallel_degrees(input_k),
      get_parallel_degrees(input_v));

  return lift_shape_to_parallel_with_degrees(unpar_shape, degrees);
}

positive_int attention_get_oSize(ParallelTensorShape const &) {
  NOT_IMPLEMENTED();
}

positive_int attention_get_oSize(TensorShape const &) {
  NOT_IMPLEMENTED();
}

ParallelTensorDimDegrees
    attention_get_weights_parallel_dim_degrees(
                      MultiHeadAttentionAttrs const &attrs,
                      ParallelTensorDimDegrees const &input_q,
                      ParallelTensorDimDegrees const &input_k,
                      ParallelTensorDimDegrees const &input_v)
{
  check_attrs(attrs);

  MultiHeadAttentionParallelInputs parsed =
      parse_attention_parallel_input_shape(input_q, input_k, input_v);

  positive_int joined_dim_degree = 1_p;
  positive_int head_dim_degree = parsed.discard_copy_degree.value;

  return ParallelTensorDimDegrees{
    /*sum_degree=*/SumDegree{1_p},
    /*discard_copy_degree=*/DiscardCopyDegree{parsed.batch_degree},
    /*shard_degrees=*/FFOrdered<positive_int>{joined_dim_degree, head_dim_degree},
  };
}

ParallelTensorDimDegrees
    attention_get_input_bias_parallel_dim_degrees(MultiHeadAttentionAttrs const &attrs,
                         ParallelTensorDimDegrees const &input_q,
                         ParallelTensorDimDegrees const &input_k,
                         ParallelTensorDimDegrees const &input_v)
{
  check_attrs(attrs);

  MultiHeadAttentionParallelInputs parsed =
        parse_attention_parallel_input_shape(input_q, input_k, input_v);

  SumDegree sum_degree = SumDegree{1_p};
  DiscardCopyDegree discard_copy_degree = DiscardCopyDegree{
      parsed.batch_degree * parsed.discard_copy_degree.value};
  FFOrdered<positive_int> shard_degrees = FFOrdered<positive_int>{1_p};

  return ParallelTensorDimDegrees{
    /*sum_degree=*/sum_degree,
    /*discard_copy_degree=*/discard_copy_degree,
    /*shard_degrees=*/shard_degrees,
  };
}

ParallelTensorDimDegrees
    attention_get_output_bias_parallel_dim_degrees(MultiHeadAttentionAttrs const &attrs,
                          ParallelTensorDimDegrees const &input_q,
                          ParallelTensorDimDegrees const &input_k,
                          ParallelTensorDimDegrees const &input_v)
{
  check_attrs(attrs);

  MultiHeadAttentionParallelInputs parsed =
        parse_attention_parallel_input_shape(input_q, input_k, input_v);

  SumDegree sum_degree = SumDegree{1_p};
  DiscardCopyDegree discard_copy_degree = DiscardCopyDegree{
      parsed.batch_degree * parsed.discard_copy_degree.value};
  FFOrdered<positive_int> shard_degrees = FFOrdered<positive_int>{1_p};

  return ParallelTensorDimDegrees{
    /*sum_degree=*/sum_degree,
    /*discard_copy_degree=*/discard_copy_degree,
    /*shard_degrees=*/shard_degrees,
  };
}

ParallelTensorDimDegrees
    attention_get_output_parallel_dim_degrees(MultiHeadAttentionAttrs const &attrs,
                     ParallelTensorDimDegrees const &input_q,
                     ParallelTensorDimDegrees const &input_k,
                     ParallelTensorDimDegrees const &input_v)
{
  check_attrs(attrs);

  MultiHeadAttentionParallelInputs parsed =
      parse_attention_parallel_input_shape(input_q, input_k, input_v);

  positive_int sum_degree = parsed.discard_copy_degree.value;
  positive_int discard_copy_degree = 1_p;
  positive_int batch_degree = parsed.batch_degree;
  positive_int seq_len_degree = 1_p;
  positive_int out_dim_degree = 1_p;

  return ParallelTensorDimDegrees{
    /*sum_degree=*/SumDegree{sum_degree},
    /*discard_copy_degree=*/DiscardCopyDegree{discard_copy_degree},
    /*shard_degrees=*/FFOrdered<positive_int>{
      batch_degree,
      seq_len_degree,
      out_dim_degree,
    },
  };
}

std::map<TensorSlotName, ParallelTensorDimDegrees>
    attention_get_weight_parallel_dim_degrees(MultiHeadAttentionAttrs const &attrs,
                      ParallelTensorDimDegrees const &input_q,
                      ParallelTensorDimDegrees const &input_k,
                      ParallelTensorDimDegrees const &input_v)
{
  std::map<TensorSlotName, ParallelTensorDimDegrees> weight_degrees = {
      {
          TensorSlotName::WEIGHT,
          attention_get_weights_parallel_dim_degrees(attrs, input_q, input_k, input_v),
      },
  };

  if (attrs.bias) {
    weight_degrees.insert({
        TensorSlotName::INPUT_BIAS,
        attention_get_input_bias_parallel_dim_degrees(attrs, input_q, input_k, input_v),
    });

    weight_degrees.insert({
        TensorSlotName::OUTPUT_BIAS,
        attention_get_output_bias_parallel_dim_degrees(attrs, input_q, input_k, input_v),
    });
  }

  return weight_degrees;
}

std::map<TensorSlotName, ParallelTensorShape>
    attention_get_weight_parallel_shapes(MultiHeadAttentionAttrs const &attrs,
                      ParallelTensorShape const &input_q,
                      ParallelTensorShape const &input_k,
                      ParallelTensorShape const &input_v) {

  std::map<TensorSlotName, ParallelTensorShape> weight_shapes = {
      {
          TensorSlotName::WEIGHT,
          attention_get_weights_parallel_shape(attrs, input_q, input_k, input_v),
      },
  };

  if (attrs.bias) {
    weight_shapes.insert({
        TensorSlotName::INPUT_BIAS,
        attention_get_input_bias_parallel_shape(attrs, input_q, input_k, input_v),
    });

    weight_shapes.insert({
        TensorSlotName::OUTPUT_BIAS,
        attention_get_output_bias_parallel_shape(attrs, input_q, input_k, input_v),
    });
  }

  return weight_shapes;
}

StandardOperatorTaskGroup attention_get_task_group(
    MultiHeadAttentionAttrs const &attrs,
    ParallelTensorDimDegrees const &input_q,
    ParallelTensorDimDegrees const &input_k,
    ParallelTensorDimDegrees const &input_v)
{
  ParallelTensorDimDegrees output_degrees =
    attention_get_output_parallel_dim_degrees(attrs, input_q, input_k, input_v);

  StandardOperatorTaskGroup task_group = StandardOperatorTaskGroup{
    transform(
      get_parallel_tensor_space_coordinates(output_degrees),
      [&](ParallelTensorSpaceCoordinate const &output_coord)
        -> AbstractedOperatorAtomicTaskShardBinding
      {
        parallel_tensor_dim_idx_t output_batch_dim =  
          shard_dim_idx(ff_dim_t{0_n});

        BoundedComponent data_parallelism_component = 
          bounded_component_for_ptensor_dim( 
            output_degrees,
            output_coord,
            output_batch_dim);

        ParallelTensorSpaceCoordinate query_coord = 
          parallel_tensor_space_coordinate_from_bounded_orthotope_components(
            /*sum_component=*/trivial_bounded_component(),
            /*discard_copy_component=*/trivial_bounded_component(),
            /*shard_components=*/make_3d_orthotope_bounded_coord(
              data_parallelism_component,
              trivial_bounded_component(),
              trivial_bounded_component()));

        ParallelTensorSpaceCoordinate key_coord = query_coord;
        ParallelTensorSpaceCoordinate value_coord = query_coord;

        ParallelTensorSpaceCoordinate weight_coord = 
          parallel_tensor_space_coordinate_from_bounded_orthotope_components(
            /*sum_component=*/trivial_bounded_component(),
            /*discard_copy_component=*/data_parallelism_component,
            /*shard_components=*/make_2d_orthotope_bounded_coord(
              trivial_bounded_component(),
              trivial_bounded_component()));

        ParallelTensorSpaceCoordinate input_bias_coord = 
          parallel_tensor_space_coordinate_from_bounded_orthotope_components(
            /*sum_component=*/trivial_bounded_component(),
            /*discard_copy_component=*/data_parallelism_component,
            /*shard_components=*/lift_bounded_component(
              trivial_bounded_component()));
          ;
        ParallelTensorSpaceCoordinate output_bias_coord = input_bias_coord;

        return AbstractedOperatorAtomicTaskShardBinding{
          /*tensor_corods=*/{
            {
              TensorSlotName::QUERY,
              query_coord,
            },
            {
              TensorSlotName::KEY,
              key_coord,
            },
            {
              TensorSlotName::VALUE,
              value_coord,
            },
            {
              TensorSlotName::WEIGHT,
              weight_coord,
            },
            {
              TensorSlotName::INPUT_BIAS,
              input_bias_coord,
            },
            {
              TensorSlotName::OUTPUT_BIAS,
              output_bias_coord,
            },
            {
              TensorSlotName::OUTPUT,
              output_coord,
            },
          },
          /*task_coord=*/task_coord_matching_parallel_tensor_space_coordinate(output_coord, output_degrees),
        };
      }),
  };

  return restrict_standard_operator_task_group_to_slots(
    task_group,
    attention_get_slots(attrs));
}

ShardSignatureInstance attention_get_shard_signature_instance(
    MultiHeadAttentionAttrs const &attrs,
    ParallelTensorDimDegrees const &input_q,
    ParallelTensorDimDegrees const &input_k,
    ParallelTensorDimDegrees const &input_v)
{
  StandardOperatorTaskGroup op_task_group =
    attention_get_task_group(attrs, input_q, input_k, input_v);

  return shard_signature_instance_from_standard_operator_task_group(op_task_group);
}

OperatorTaskSpace attention_get_operator_task_space(
    MultiHeadAttentionAttrs const &attrs,
    ParallelTensorDimDegrees const &input_q,
    ParallelTensorDimDegrees const &input_k,
    ParallelTensorDimDegrees const &input_v)
{
  StandardOperatorTaskGroup op_task_group =
    attention_get_task_group(attrs, input_q, input_k, input_v);

  return task_space_for_standard_operator_task_group(op_task_group);
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping
  attention_get_operator_to_query_input_mapping(
    MultiHeadAttentionAttrs const &attrs,
    ParallelTensorDimDegrees const &input_q,
    ParallelTensorDimDegrees const &input_k,
    ParallelTensorDimDegrees const &input_v)
{
  StandardOperatorTaskGroup op_task_group =
    attention_get_task_group(attrs, input_q, input_k, input_v);

  return standard_operator_task_group_get_operator_to_ptensor_mapping(op_task_group, TensorSlotName::QUERY);
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping
  attention_get_operator_to_key_input_mapping(
    MultiHeadAttentionAttrs const &attrs,
    ParallelTensorDimDegrees const &input_q,
    ParallelTensorDimDegrees const &input_k,
    ParallelTensorDimDegrees const &input_v)
{
  StandardOperatorTaskGroup op_task_group =
    attention_get_task_group(attrs, input_q, input_k, input_v);

  return standard_operator_task_group_get_operator_to_ptensor_mapping(op_task_group, TensorSlotName::KEY);
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping
  attention_get_operator_to_value_input_mapping(
    MultiHeadAttentionAttrs const &attrs,
    ParallelTensorDimDegrees const &input_q,
    ParallelTensorDimDegrees const &input_k,
    ParallelTensorDimDegrees const &input_v)
{
  StandardOperatorTaskGroup op_task_group =
    attention_get_task_group(attrs, input_q, input_k, input_v);

  return standard_operator_task_group_get_operator_to_ptensor_mapping(op_task_group, TensorSlotName::VALUE);
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping
  attention_get_operator_to_weights_mapping(
    MultiHeadAttentionAttrs const &attrs,
    ParallelTensorDimDegrees const &input_q,
    ParallelTensorDimDegrees const &input_k,
    ParallelTensorDimDegrees const &input_v)
{
  StandardOperatorTaskGroup op_task_group =
    attention_get_task_group(attrs, input_q, input_k, input_v);

  return standard_operator_task_group_get_operator_to_ptensor_mapping(op_task_group, TensorSlotName::WEIGHT);
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping
  attention_get_operator_to_input_bias_mapping(
    MultiHeadAttentionAttrs const &attrs,
    ParallelTensorDimDegrees const &input_q,
    ParallelTensorDimDegrees const &input_k,
    ParallelTensorDimDegrees const &input_v)
{
  StandardOperatorTaskGroup op_task_group =
    attention_get_task_group(attrs, input_q, input_k, input_v);

  return standard_operator_task_group_get_operator_to_ptensor_mapping(op_task_group, TensorSlotName::INPUT_BIAS);
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping
  attention_get_operator_to_output_bias_mapping(
    MultiHeadAttentionAttrs const &attrs,
    ParallelTensorDimDegrees const &input_q,
    ParallelTensorDimDegrees const &input_k,
    ParallelTensorDimDegrees const &input_v)
{
  StandardOperatorTaskGroup op_task_group =
    attention_get_task_group(attrs, input_q, input_k, input_v);

  return standard_operator_task_group_get_operator_to_ptensor_mapping(op_task_group, TensorSlotName::OUTPUT_BIAS);
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping
  attention_get_operator_to_output_mapping(
    MultiHeadAttentionAttrs const &attrs,
    ParallelTensorDimDegrees const &input_q,
    ParallelTensorDimDegrees const &input_k,
    ParallelTensorDimDegrees const &input_v)
{
  StandardOperatorTaskGroup op_task_group =
    attention_get_task_group(attrs, input_q, input_k, input_v);

  return standard_operator_task_group_get_operator_to_ptensor_mapping(op_task_group, TensorSlotName::OUTPUT);
}

std::map<TensorSlotName, OperatorSpaceToParallelTensorSpaceBiuniqueMapping>
  attention_get_operator_to_parallel_tensor_mappings(
    MultiHeadAttentionAttrs const &attrs,
    ParallelTensorDimDegrees const &input_q,
    ParallelTensorDimDegrees const &input_k,
    ParallelTensorDimDegrees const &input_v)
{
  std::map<TensorSlotName, OperatorSpaceToParallelTensorSpaceBiuniqueMapping> result = {
    {
      TensorSlotName::QUERY,
      attention_get_operator_to_query_input_mapping(attrs, input_q, input_k, input_v),
    },
    {
      TensorSlotName::KEY,
      attention_get_operator_to_key_input_mapping(attrs, input_q, input_k, input_v),
    },
    {
      TensorSlotName::VALUE,
      attention_get_operator_to_value_input_mapping(attrs, input_q, input_k, input_v),
    },
    {
      TensorSlotName::WEIGHT,
      attention_get_operator_to_weights_mapping(attrs, input_q, input_k, input_v),
    },
    {
      TensorSlotName::OUTPUT,
      attention_get_operator_to_output_mapping(attrs, input_q, input_k, input_v),
    },
  };

  if (attrs.bias) {
    result.insert({
      TensorSlotName::INPUT_BIAS,
      attention_get_operator_to_input_bias_mapping(attrs, input_q, input_k, input_v),
    });
    result.insert({
      TensorSlotName::OUTPUT_BIAS,
      attention_get_operator_to_output_bias_mapping(attrs, input_q, input_k, input_v),
    });
  }

  return result;
}

std::map<TensorSlotName, InitializerAttrs>
    attention_get_initializers(
        MultiHeadAttentionAttrs const &attrs,
        TensorShape const &input_q,
        TensorShape const &input_k,
        TensorShape const &input_v,
        std::optional<InitializerAttrs> const &maybe_weights_initializer,
        std::optional<InitializerAttrs> const &maybe_input_bias_initializer,
        std::optional<InitializerAttrs> const &maybe_output_bias_initializer) {
  check_attrs(attrs);

  ASSERT(
    attrs.bias || !maybe_input_bias_initializer.has_value(),
    fmt::format("Expected input_bias_initializer=std::nullopt since "
                "bias=false, but received input_bias_initializer: {}",
                maybe_input_bias_initializer.value())
  );

  ASSERT(
    attrs.bias || !maybe_output_bias_initializer.has_value(),
    fmt::format("Expected output_bias_initializer=std::nullopt since "
                "bias=false, but received output_bias_initializer: {}",
                maybe_output_bias_initializer.value())
  );

  InitializerAttrs default_weights_initializer = InitializerAttrs{
      GlorotUniformAttrs{
          /*seed=*/0,
      },
  };

  InitializerAttrs default_input_bias_initializer = InitializerAttrs{
      ZeroInitializerAttrs{},
  };

  InitializerAttrs default_output_bias_initializer = InitializerAttrs{
      ZeroInitializerAttrs{},
  };

  InitializerAttrs weights_initializer =
      maybe_weights_initializer.value_or(default_weights_initializer);
  InitializerAttrs input_bias_initializer =
      maybe_input_bias_initializer.value_or(default_input_bias_initializer);
  InitializerAttrs output_bias_initializer =
      maybe_output_bias_initializer.value_or(default_output_bias_initializer);

  if (attrs.bias) {
    return std::map<TensorSlotName, InitializerAttrs>{
        {TensorSlotName::WEIGHT, weights_initializer},
        {TensorSlotName::INPUT_BIAS, input_bias_initializer},
        {TensorSlotName::OUTPUT_BIAS, output_bias_initializer},
    };
  } else {
    return std::map<TensorSlotName, InitializerAttrs>{
        {TensorSlotName::WEIGHT, weights_initializer},
    };
  }
}

} // namespace FlexFlow
