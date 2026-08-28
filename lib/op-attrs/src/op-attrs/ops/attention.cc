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

namespace FlexFlow {

/* bool MultiHeadAttentionAttrs::is_valid(std::vector<ParallelTensorShape> const
 * &inputs) const { */
/*   return (inputs.size() == 3 && std::all_of(inputs.begin(), inputs.end(),
 * [](ParallelTensorShape const &s) { return s.is_valid(); })); */
/*   bool is_valid = true; */
/*   return is_valid; */
/* } */

positive_int attention_get_qProjSize(MultiHeadAttentionAttrs const &attrs) {
  return attrs.kdim;
}

positive_int attention_get_vProjSize(MultiHeadAttentionAttrs const &attrs) {
  return attrs.vdim;
}

positive_int attention_get_kProjSize(MultiHeadAttentionAttrs const &attrs) {
  return attrs.kdim;
}

positive_int attention_get_oProjSize(MultiHeadAttentionAttrs const &attrs) {
  return attrs.embed_dim;
}

positive_int attention_get_qSize(TensorShape const &query_shape) {
  return dim_at_idx(query_shape.dims, relative_ff_dim_t{0});
}

positive_int attention_get_kSize(TensorShape const &key_shape) {
  return dim_at_idx(key_shape.dims, relative_ff_dim_t{0});
}

positive_int attention_get_vSize(TensorShape const &value_shape) {
  return dim_at_idx(value_shape.dims, relative_ff_dim_t{0});
}

positive_int attention_get_qSize(MultiHeadAttentionParallelInputs const &inputs) {
  return inputs.query_dim.size;
}

positive_int attention_get_qSize(MultiHeadAttentionInputs const &inputs) {
  return inputs.query_size;
}

positive_int attention_get_kSize(MultiHeadAttentionParallelInputs const &inputs) {
  return inputs.key_dim.size;
}

positive_int attention_get_kSize(MultiHeadAttentionInputs const &inputs) {
  return inputs.key_size;
}

positive_int attention_get_vSize(MultiHeadAttentionParallelInputs const &inputs) {
  return inputs.value_dim.size;
}

positive_int attention_get_vSize(MultiHeadAttentionInputs const &inputs) {
  return inputs.value_size;
}

positive_int attention_get_kvSeqLength(MultiHeadAttentionParallelInputs const &inputs) {
  return inputs.sequence_dim.size;
}

positive_int attention_get_kvSeqLength(MultiHeadAttentionInputs const &inputs) {
  return inputs.sequence_length;
}

positive_int attention_get_qoSeqLength(MultiHeadAttentionParallelInputs const &inputs) {
  return inputs.sequence_dim.size; // FIXME -- assumes only prefill
}

positive_int attention_get_qoSeqLength(MultiHeadAttentionInputs const &inputs) {
  return inputs.sequence_length; // FIXME -- assumes only prefil
}

positive_int attention_get_num_samples(MultiHeadAttentionParallelInputs const &inputs) {
  return inputs.batch_dim.size;
}

positive_int attention_get_num_samples(MultiHeadAttentionInputs const &inputs) {
  return inputs.batch_size;
}

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
  MultiHeadAttentionParallelInputs parsed =
      parse_attention_parallel_input_shape(input_q, input_k, input_v);

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

  return lift_to_parallel_with_degrees(unpar_shape, degrees);
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

  return lift_to_parallel_with_degrees(unpar_shape, degrees);
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

  return lift_to_parallel_with_degrees(unpar_shape, degrees);
}

ParallelTensorShape
    attention_get_output_parallel_shape(MultiHeadAttentionAttrs const &attrs,
                     ParallelTensorShape const &input_q,
                     ParallelTensorShape const &input_k,
                     ParallelTensorShape const &input_v) {

  MultiHeadAttentionParallelInputs parsed =
      parse_attention_parallel_input_shape(input_q, input_k, input_v);

  TensorShape unpar_shape =
      attention_get_output_shape(attrs,
                       get_reduced_shape(input_q),
                       get_reduced_shape(input_k),
                       get_reduced_shape(input_v));

  positive_int sum_degree = parsed.discard_copy_degree.value;
  positive_int discard_copy_degree = 1_p;
  positive_int batch_degree = parsed.batch_dim.degree;
  positive_int seq_len_degree = 1_p;
  positive_int out_dim_degree = 1_p;

  ParallelTensorDimDegrees degrees =
    attention_get_output_parallel_dim_degrees(
      attrs,
      get_parallel_degrees(input_q),
      get_parallel_degrees(input_k),
      get_parallel_degrees(input_v));

  return lift_to_parallel_with_degrees(unpar_shape, degrees);
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

  TensorShape unpar_shape =
      attention_get_weights_shape(attrs,
                        get_reduced_shape(input_q),
                        get_reduced_shape(input_k),
                        get_reduced_shape(input_v));

  positive_int joined_dim_degree = 1_p;
  positive_int head_dim_degree = parsed.discard_copy_degree.value;

  return ParallelTensorDimDegrees{
    /*sum_degree=*/SumDegree{1_p},
    /*discard_copy_degree=*/DiscardCopyDegree{parsed.batch_dim.degree},
    /*shard_degrees=*/FFOrdered<positive_int>{joined_dim_degree, head_dim_degree},
  };
}

ParallelTensorDimDegrees
    attention_get_input_bias_parallel_dim_degrees(MultiHeadAttentionAttrs const &,
                         ParallelTensorDimDegrees const &input_q,
                         ParallelTensorDimDegrees const &input_k,
                         ParallelTensorDimDegrees const &input_v)
{
  check_attrs(attrs);

  MultiHeadAttentionParallelInputs parsed =
        parse_attention_parallel_input_shape(input_q, input_k, input_v);


  SumDegree sum_degree = SumDegree{1_p};
  DiscardCopyDegree discard_copy_degree = DiscardCopyDegree{
      parsed.batch_dim.degree * parsed.discard_copy_degree.value};
  FFOrdered<positive_int> shard_degrees = FFOrdered<positive_int>{1_p};

  return lift_to_parallel_with_degrees(
      unpar_shape, sum_degree, discard_copy_degree, shard_degrees);
}

ParallelTensorDimDegrees
    attention_get_output_bias_parallel_dim_degrees(MultiHeadAttentionAttrs const &,
                          ParallelTensorDimDegrees const &input_q,
                          ParallelTensorDimDegrees const &input_k,
                          ParallelTensorDimDegrees const &input_v)
{
  check_attrs(attrs);

  MultiHeadAttentionParallelInputs parsed =
        parse_attention_parallel_input_shape(input_q, input_k, input_v);

  SumDegree sum_degree = SumDegree{1_p};
  DiscardCopyDegree discard_copy_degree = DiscardCopyDegree{
      parsed.batch_dim.degree * parsed.discard_copy_degree.value};
  FFOrdered<positive_int> shard_degrees = FFOrdered<positive_int>{1_p};

  return ParallelTensorDimDegrees{
    /*sum_degree=*/sum_degree,
    /*discard_copy_degree=*/discard_copy_degree,
    /*shard_degrees=*/shard_degrees,
  };
}

ParallelTensorDimDegrees
    attention_get_output_parallel_dim_degrees(MultiHeadAttentionAttrs const &,
                     ParallelTensorDimDegrees const &input_q,
                     ParallelTensorDimDegrees const &input_k,
                     ParallelTensorDimDegrees const &input_v)
{
  check_attrs(attrs);

  MultiHeadAttentionParallelInputs parsed =
      parse_attention_parallel_input_shape(input_q, input_k, input_v);

  TensorShape unpar_shape =
      attention_get_output_shape(attrs,
                       get_reduced_shape(input_q),
                       get_reduced_shape(input_k),
                       get_reduced_shape(input_v));

  positive_int sum_degree = parsed.discard_copy_degree.value;
  positive_int discard_copy_degree = 1_p;
  positive_int batch_degree = parsed.batch_dim.degree;
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
    attention_get_weight_parallel_dim_degrees(MultiHeadAttentionAttrs const &,
                      ParallelTensorDimDegrees const &input_q,
                      ParallelTensorDimDegrees const &input_k,
                      ParallelTensorDimDegrees const &input_v)
{
  std::map<TensorSlotName, ParallelTensorShape> weight_degrees = {
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

OperatorTaskSpace attention_get_operator_task_space(
    MultiHeadAttentionAttrs const &attrs,
    ParallelTensorDimDegrees const &input_q,
    ParallelTensorDimDegrees const &input_k,
    ParallelTensorDimDegrees const &input_v)
{
  ParallelTensorDimDegrees output_degrees =
      attention_get_output_parallel_dim_degrees(attrs, input_q, input_k, input_v);

  return get_operator_task_space_matching_parallel_tensor_dim_degrees(
      output_degrees);
}

static ParallelTensorSpaceToParallelTensorSpaceBiuniqueMapping 
  attention_get_any_input_to_output_mapping(
    MultiHeadAttentionAttrs const &attrs,
    ParallelTensorDimDegrees const &input_q,
    ParallelTensorDimDegrees const &input_k,
    ParallelTensorDimDegrees const &input_v)
{
  EqProjection<parallel_tensor_dim_idx_t, parallel_tensor_dim_idx_t>
      inp_to_out = make_empty_eq_projection<
                      parallel_tensor_dim_idx_t,
                      parallel_tensor_dim_idx_t>();

  parallel_tensor_dim_idx_t batch_dim = 
    shard_dim_idx(ff_dim_t{0_n});

  project_dims(inp_to_out, discard_copy_degree(), sum_dim_idx());
  project_dims(inp_to_out, batch_dim, batch_dim);

  ParallelTensorDimDegrees output_degrees =
      attention_get_output_parallel_dim_degrees(attrs, input_q, input_k, input_v);

  return parallel_tensor_space_biunique_mapping_from_projection(
      DimProjection{inp_to_out}, input_q, output_degrees);
}

static ParallelTensorSpaceToParallelTensorSpaceBiuniqueMapping 
  attention_get_weights_to_output_mapping(
    MultiHeadAttentionAttrs const &attrs,
    ParallelTensorDimDegrees const &input_q,
    ParallelTensorDimDegrees const &input_k,
    ParallelTensorDimDegrees const &input_v)
{
  EqProjection<parallel_tensor_dim_idx_t, parallel_tensor_dim_idx_t>
      weights_to_out = make_empty_eq_projection<
                      parallel_tensor_dim_idx_t,
                      parallel_tensor_dim_idx_t>();

  parallel_tensor_dim_idx_t output_batch_dim = 
    shard_dim_idx(ff_dim_t{0_n});

  parallel_tensor_dim_idx_t weights_head_dim = 
    shard_dim_idx(ff_dim_t{0_n});

  project_dims(weights_to_out, weights_head_dim, sum_dim_idx());
  project_dims(weights_to_out, discard_copy_dim_idx(), output_batch_dim);

  ParallelTensorDimDegrees weights_degrees =
      attention_get_weights_parallel_dim_degrees(attrs, input_q, input_k, input_v);

  ParallelTensorDimDegrees output_degrees =
      attention_get_output_parallel_dim_degrees(attrs, input_q, input_k, input_v);

  return parallel_tensor_space_biunique_mapping_from_projection(
      DimProjection{weights_to_out}, weights_degrees, output_degrees);
}

static ParallelTensorSpaceToParallelTensorSpaceBiuniqueMapping 
  attention_get_input_bias_to_output_mapping(
    MultiHeadAttentionAttrs const &attrs,
    ParallelTensorDimDegrees const &input_q,
    ParallelTensorDimDegrees const &input_k,
    ParallelTensorDimDegrees const &input_v)
{
  EqProjection<parallel_tensor_dim_idx_t, parallel_tensor_dim_idx_t>
      input_bias_to_out = make_empty_eq_projection<
                      parallel_tensor_dim_idx_t,
                      parallel_tensor_dim_idx_t>();

  parallel_tensor_dim_idx_t output_batch_dim = 
    shard_dim_idx(ff_dim_t{0_n});

  project_dims(input_bias_to_out, discard_copy_dim_idx(), sum_dim_idx());
  project_dims(input_bias_to_out, discard_copy_dim_idx(), output_batch_dim);

  ParallelTensorDimDegrees input_bias_degrees =
      attention_get_input_bias_parallel_dim_degrees(attrs, input_q, input_k, input_v);

  ParallelTensorDimDegrees output_degrees =
      attention_get_output_parallel_dim_degrees(attrs, input_q, input_k, input_v);

  return parallel_tensor_space_biunique_mapping_from_projection(
      DimProjection{input_bias_to_out}, input_bias_degrees, output_degrees);
}

static ParallelTensorSpaceToParallelTensorSpaceBiuniqueMapping 
  attention_get_output_bias_to_output_mapping(
    MultiHeadAttentionAttrs const &attrs,
    ParallelTensorDimDegrees const &input_q,
    ParallelTensorDimDegrees const &input_k,
    ParallelTensorDimDegrees const &input_v)
{
  EqProjection<parallel_tensor_dim_idx_t, parallel_tensor_dim_idx_t>
      output_bias_to_out = make_empty_eq_projection<
                      parallel_tensor_dim_idx_t,
                      parallel_tensor_dim_idx_t>();

  parallel_tensor_dim_idx_t output_batch_dim = 
    shard_dim_idx(ff_dim_t{0_n});

  project_dims(output_bias_to_out, discard_copy_dim_idx(), sum_dim_idx());
  project_dims(output_bias_to_out, discard_copy_dim_idx(), output_batch_dim);

  ParallelTensorDimDegrees output_bias_degrees =
      attention_get_output_bias_parallel_dim_degrees(attrs, input_q, input_k, input_v);

  ParallelTensorDimDegrees output_degrees =
      attention_get_output_parallel_dim_degrees(attrs, input_q, input_k, input_v);

  return parallel_tensor_space_biunique_mapping_from_projection(
      DimProjection{output_bias_to_out}, output_bias_degrees, output_degrees);
}

static OperatorSpaceToParallelTensorSpaceBiuniqueMapping
  attention_get_operator_to_any_input_mapping(
    MultiHeadAttentionAttrs const &attrs,
    ParallelTensorDimDegrees const &input_q,
    ParallelTensorDimDegrees const &input_k,
    ParallelTensorDimDegrees const &input_v)
{
  ParallelTensorSpaceToParallelTensorSpaceBiuniqueMapping inp_to_out =
      attention_get_any_input_to_output_mapping(attrs, input_q, input_k, input_v);

  ParallelTensorSpaceToParallelTensorSpaceBiuniqueMapping out_to_inp =
      invert_parallel_tensor_space_biunique_mapping(inp_to_out);

  OperatorSpaceToParallelTensorSpaceBiuniqueMapping op_to_out =
      attention_get_operator_to_output_mapping(attrs, input_q, input_k, input_v);

  return operator_ptensor_space_biunique_mapping_from_composition(op_to_out, out_to_inp);
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping
  attention_get_operator_to_key_input_mapping(
    MultiHeadAttentionAttrs const &attrs,
    ParallelTensorDimDegrees const &input_q,
    ParallelTensorDimDegrees const &input_k,
    ParallelTensorDimDegrees const &input_v)
{
  return attention_get_operator_to_any_input_mapping(attrs, input_q, input_k, input_v);
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping
  attention_get_operator_to_key_input_mapping(
    MultiHeadAttentionAttrs const &attrs,
    ParallelTensorDimDegrees const &input_q,
    ParallelTensorDimDegrees const &input_k,
    ParallelTensorDimDegrees const &input_v)
{
  return attention_get_operator_to_any_input_mapping(attrs, input_q, input_k, input_v);
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping
  attention_get_operator_to_value_input_mapping(
    MultiHeadAttentionAttrs const &attrs,
    ParallelTensorDimDegrees const &input_q,
    ParallelTensorDimDegrees const &input_k,
    ParallelTensorDimDegrees const &input_v)
{
  return attention_get_operator_to_any_input_mapping(attrs, input_q, input_k, input_v);
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping
  attention_get_operator_to_weights_mapping(
    MultiHeadAttentionAttrs const &attrs,
    ParallelTensorDimDegrees const &input_q,
    ParallelTensorDimDegrees const &input_k,
    ParallelTensorDimDegrees const &input_v)
{
  ParallelTensorSpaceToParallelTensorSpaceBiuniqueMapping weights_to_out =
      attention_get_weights_to_output_mapping(attrs, input_q, input_k, input_v);

  ParallelTensorSpaceToParallelTensorSpaceBiuniqueMapping out_to_weights =
      invert_parallel_tensor_space_biunique_mapping(weights_to_out);

  OperatorSpaceToParallelTensorSpaceBiuniqueMapping op_to_out =
      attention_get_operator_to_output_mapping(attrs, input_q, input_k, input_v);

  return operator_ptensor_space_biunique_mapping_from_composition(op_to_out, out_to_weights);
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping
  attention_get_operator_to_input_bias_mapping(
    MultiHeadAttentionAttrs const &attrs,
    ParallelTensorDimDegrees const &input_q,
    ParallelTensorDimDegrees const &input_k,
    ParallelTensorDimDegrees const &input_v)
{
  ParallelTensorSpaceToParallelTensorSpaceBiuniqueMapping input_bias_to_out =
      attention_get_input_bias_to_output_mapping(attrs, input_q, input_k, input_v);

  ParallelTensorSpaceToParallelTensorSpaceBiuniqueMapping out_to_input_bias =
      invert_parallel_tensor_space_biunique_mapping(input_bias_to_out);

  OperatorSpaceToParallelTensorSpaceBiuniqueMapping op_to_out =
      attention_get_operator_to_output_mapping(attrs, input_q, input_k, input_v);

  return operator_ptensor_space_biunique_mapping_from_composition(op_to_out, out_to_input_bias);
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping
  attention_get_operator_to_output_bias_mapping(
    MultiHeadAttentionAttrs const &attrs,
    ParallelTensorDimDegrees const &input_q,
    ParallelTensorDimDegrees const &input_k,
    ParallelTensorDimDegrees const &input_v)
{
  ParallelTensorSpaceToParallelTensorSpaceBiuniqueMapping output_bias_to_out =
      attention_get_output_bias_to_output_mapping(attrs, input_q, input_k, input_v);

  ParallelTensorSpaceToParallelTensorSpaceBiuniqueMapping out_to_output_bias =
      invert_parallel_tensor_space_biunique_mapping(output_bias_to_out);

  OperatorSpaceToParallelTensorSpaceBiuniqueMapping op_to_out =
      attention_get_operator_to_output_mapping(attrs, input_q, input_k, input_v);

  return operator_ptensor_space_biunique_mapping_from_composition(op_to_out, out_to_output_bias);
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping
  attention_get_operator_to_output_mapping(
    MultiHeadAttentionAttrs const &attrs,
    ParallelTensorDimDegrees const &input_q,
    ParallelTensorDimDegrees const &input_k,
    ParallelTensorDimDegrees const &input_v)
{
  ParallelTensorDimDegrees output_degrees =
      attention_get_output_parallel_dim_degrees(attrs, input_q, input_k, input_v);

  return get_identity_biunique_mapping(
      attention_get_operator_task_space(attrs, input_q, input_k, input_v),
      output_degrees);
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
      attention_get_operator_to_query_input_mapping(attrs, input_q, input_k, input),
    },
    {
      TensorSlotName::KEY,
      attention_get_operator_to_key_input_mapping(attrs, input_q, input_k, input),
    },
    {
      TensorSlotName::VALUE,
      attention_get_operator_to_value_input_mapping(attrs, input_q, input_k, input),
    },
    {
      TensorSlotName::WEIGHTS,
      attention_get_operator_to_weights_mapping(attrs, input_q, input_k, input),
    },
    {
      TensorSlotName::OUTPUT,
      attention_get_operator_to_output_mapping(attrs, input_q, input_k, input),
    },
  };

  if (attrs.bias) {
    result.insert({
      TensorSlotName::INPUT_BIAS,
      attention_get_operator_to_input_bias_mapping(attrs, input_q, input_k, input),
    });
    result.insert({
      TensorSlotName::OUTPUT_BIAS,
      attention_get_operator_to_output_bias_mapping(attrs, input_q, input_k, input),
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
