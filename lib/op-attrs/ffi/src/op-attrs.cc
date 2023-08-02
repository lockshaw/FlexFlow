#include "flexflow/op-attrs.h"
#include "internal/op-attrs.h"
#include "internal/error.h"
#include "op-attrs/op.h"
#include "op-attrs/ops/embedding.h"
#include "utils/bidict.h"

flexflow_utils_exception_t make_opattrs_exception(flexflow_opattrs_error_code_t);

REGISTER_FFI_ENUM(
  flexflow_param_sync_t, 
  ParamSync, 
  FLEXFLOW_OPATTRS_ERROR_CODE_INVALID_PARAM_SYNC_VALUE, 
  {
    { FLEXFLOW_PARAM_SYNC_PARAMETER_SERVER, ParamSync::PS },
    { FLEXFLOW_PARAM_SYNC_NCCL, ParamSync::NCCL }
  });

REGISTER_FFI_ENUM(
  flexflow_datatype_t, 
  DataType, 
  FLEXFLOW_OPATTRS_ERROR_CODE_INVALID_DATATYPE_VALUE,
  {
    { FLEXFLOW_DATATYPE_BOOL, DataType::BOOL },
    { FLEXFLOW_DATATYPE_INT32, DataType::INT32 },
    { FLEXFLOW_DATATYPE_INT64, DataType::INT64 },
    { FLEXFLOW_DATATYPE_HALF, DataType::HALF },
    { FLEXFLOW_DATATYPE_FLOAT, DataType::FLOAT },
    { FLEXFLOW_DATATYPE_DOUBLE, DataType::DOUBLE }
  });

REGISTER_FFI_ENUM(
  flexflow_activation_t, 
  optional<Activation>, 
  FLEXFLOW_OPATTRS_ERROR_CODE_INVALID_ACTIVATION_VALUE,
  {
    { FLEXFLOW_ACTIVATION_RELU, Activation::RELU },
    { FLEXFLOW_ACTIVATION_SIGMOID, Activation::SIGMOID },
    { FLEXFLOW_ACTIVATION_TANH, Activation::TANH },
    { FLEXFLOW_ACTIVATION_GELU, Activation::GELU },
    { FLEXFLOW_ACTIVATION_NONE, nullopt }
  });

REGISTER_FFI_ENUM(
  flexflow_pool_op_t, 
  PoolOp, 
  FLEXFLOW_OPATTRS_ERROR_CODE_INVALID_POOL_OP_VALUE,
  {
    { FLEXFLOW_POOL_OP_MAX, PoolOp::MAX },
    { FLEXFLOW_POOL_OP_AVG, PoolOp::AVG }
  });

REGISTER_FFI_ENUM(
  flexflow_aggregate_op_t, 
  AggregateOp, 
  FLEXFLOW_OPATTRS_ERROR_CODE_INVALID_AGGREGATE_OP_VALUE,
  {
    { FLEXFLOW_AGGREGATE_OP_SUM, AggregateOp::SUM },
    { FLEXFLOW_AGGREGATE_OP_AVG, AggregateOp::AVG }
  });

REGISTER_FFI_ENUM(
  flexflow_op_type_t, 
  OperatorType,
  FLEXFLOW_OPATTRS_ERROR_CODE_INVALID_OP_TYPE_VALUE,
  {
    { FLEXFLOW_OP_TYPE_NOOP, Op::NOOP },
    { FLEXFLOW_OP_TYPE_INPUT, Op::INPUT },
    { FLEXFLOW_OP_TYPE_WEIGHT, Op::WEIGHT },
    { FLEXFLOW_OP_TYPE_CONV2D, Op::CONV2D },
    { FLEXFLOW_OP_TYPE_DROPOUT, Op::DROPOUT },
    { FLEXFLOW_OP_TYPE_LINEAR, Op::LINEAR },
    { FLEXFLOW_OP_TYPE_BATCHMATMUL, Op::BATCHMATMUL },
    { FLEXFLOW_OP_TYPE_POOL2D, Op::POOL2D },
    { FLEXFLOW_OP_TYPE_SCALAR_MULTIPLY, Op::SCALAR_MULTIPLY },
    { FLEXFLOW_OP_TYPE_SCALAR_ADD, Op::SCALAR_ADD },
    { FLEXFLOW_OP_TYPE_SCALAR_FLOOR_DIV, Op::SCALAR_FLOOR_DIV },
    { FLEXFLOW_OP_TYPE_SCALAR_TRUE_DIV, Op::SCALAR_TRUE_DIV },
    { FLEXFLOW_OP_TYPE_SCALAR_SUB, Op::SCALAR_SUB },
    { FLEXFLOW_OP_TYPE_RELU, Op::RELU },
    { FLEXFLOW_OP_TYPE_IDENTITY, Op::IDENTITY },
    { FLEXFLOW_OP_TYPE_SIGMOID, Op::SIGMOID },
    { FLEXFLOW_OP_TYPE_TANH, Op::TANH },
    { FLEXFLOW_OP_TYPE_ELU, Op::ELU },
    { FLEXFLOW_OP_TYPE_FLAT, Op::FLAT },
    { FLEXFLOW_OP_TYPE_SOFTMAX, Op::SOFTMAX },
    { FLEXFLOW_OP_TYPE_BATCHNORM, Op::BATCHNORM },
    { FLEXFLOW_OP_TYPE_CONCAT, Op::CONCAT },
    { FLEXFLOW_OP_TYPE_SPLIT, Op::SPLIT },
    { FLEXFLOW_OP_TYPE_EMBEDDING, Op::EMBEDDING },
    { FLEXFLOW_OP_TYPE_GROUP_BY, Op::GROUP_BY },
    { FLEXFLOW_OP_TYPE_CACHE, Op::CACHE },
    { FLEXFLOW_OP_TYPE_AGGREGATE, Op::AGGREGATE },
    { FLEXFLOW_OP_TYPE_AGG_SPEC, Op::AGG_SPEC },
    { FLEXFLOW_OP_TYPE_RESHAPE, Op::RESHAPE },
    { FLEXFLOW_OP_TYPE_REVERSE, Op::REVERSE },
    { FLEXFLOW_OP_TYPE_TRANSPOSE, Op::TRANSPOSE },
    { FLEXFLOW_OP_TYPE_EW_ADD, Op::EW_ADD },
    { FLEXFLOW_OP_TYPE_EW_MUL, Op::EW_MUL },
    { FLEXFLOW_OP_TYPE_MATMUL, Op::MATMUL },
    { FLEXFLOW_OP_TYPE_MUL, Op::MUL },
    { FLEXFLOW_OP_TYPE_ENLARGE, Op::ENLARGE },
    { FLEXFLOW_OP_TYPE_SQUEEZE, Op::SQUEEZE },
    { FLEXFLOW_OP_TYPE_UNSQUEEZE, Op::UNSQUEEZE },
    { FLEXFLOW_OP_TYPE_EW_SUB, Op::EW_SUB },
    { FLEXFLOW_OP_TYPE_EW_DIV, Op::EW_DIV },
    { FLEXFLOW_OP_TYPE_EW_EQUAL, Op::EW_EQUAL },
    { FLEXFLOW_OP_TYPE_EW_GREATER, Op::EW_GREATER },
    { FLEXFLOW_OP_TYPE_EW_LESS, Op::EW_LESS },
    { FLEXFLOW_OP_TYPE_EW_MAX, Op::EW_MAX },
    { FLEXFLOW_OP_TYPE_EW_MIN, Op::EW_MIN },
    { FLEXFLOW_OP_TYPE_REDUCE_ARGMAX, Op::REDUCE_ARGMAX },
    { FLEXFLOW_OP_TYPE_REDUCE_ARGMIN, Op::REDUCE_ARGMIN },
    { FLEXFLOW_OP_TYPE_REDUCE_MAX, Op::REDUCE_MAX },
    { FLEXFLOW_OP_TYPE_REDUCE_MEAN, Op::REDUCE_MEAN },
    { FLEXFLOW_OP_TYPE_REDUCE_MIN, Op::REDUCE_MIN },
    { FLEXFLOW_OP_TYPE_REDUCE_PROD, Op::REDUCE_PROD },
    { FLEXFLOW_OP_TYPE_REDUCE_SUM, Op::REDUCE_SUM },
    { FLEXFLOW_OP_TYPE_PAD, Op::PAD },
    { FLEXFLOW_OP_TYPE_SHAPE, Op::SHAPE },
    { FLEXFLOW_OP_TYPE_SIZE, Op::SIZE },
    { FLEXFLOW_OP_TYPE_TOPK, Op::TOPK },
    { FLEXFLOW_OP_TYPE_WHERE, Op::WHERE },
    { FLEXFLOW_OP_TYPE_CEIL, Op::CEIL },
    { FLEXFLOW_OP_TYPE_CAST, Op::CAST },
    { FLEXFLOW_OP_TYPE_EXP, Op::EXP },
    { FLEXFLOW_OP_TYPE_ROUND, Op::ROUND },
    { FLEXFLOW_OP_TYPE_LOG, Op::LOG },
    { FLEXFLOW_OP_TYPE_LOGICAL_NOT, Op::LOGICAL_NOT },
    { FLEXFLOW_OP_TYPE_SQRT, Op::SQRT },
    { FLEXFLOW_OP_TYPE_SIN, Op::SIN },
    { FLEXFLOW_OP_TYPE_COS, Op::COS },
    { FLEXFLOW_OP_TYPE_LEAKYRELU, Op::LEAKYRELU },
    { FLEXFLOW_OP_TYPE_SLICE, Op::SLICE },
    { FLEXFLOW_OP_TYPE_RESIZE, Op::RESIZE },
    { FLEXFLOW_OP_TYPE_PRELU, Op::PRELU },
    { FLEXFLOW_OP_TYPE_GELU, Op::GELU },
    { FLEXFLOW_OP_TYPE_MULTIHEAD_ATTENTION, Op::MULTIHEAD_ATTENTION },
    { FLEXFLOW_OP_TYPE_FUSED, Op::FUSED },
    { FLEXFLOW_OP_TYPE_RSQRT, Op::RSQRT },
    { FLEXFLOW_OP_TYPE_POW, Op::POW },
    { FLEXFLOW_OP_TYPE_MEAN, Op::MEAN },
    { FLEXFLOW_OP_TYPE_LAYERNORM, Op::LAYERNORM },
    { FLEXFLOW_OP_TYPE_GATHER, Op::GATHER },
    { FLEXFLOW_OP_TYPE_BROADCAST, Op::BROADCAST },
    { FLEXFLOW_OP_TYPE_REPARTITION, Op::REPARTITION },
    { FLEXFLOW_OP_TYPE_COMBINE, Op::COMBINE },
    { FLEXFLOW_OP_TYPE_REPLICATE, Op::REPLICATE },
    { FLEXFLOW_OP_TYPE_REDUCTION, Op::REDUCTION },
    { FLEXFLOW_OP_TYPE_BATCH, Op::BATCH },
    { FLEXFLOW_OP_TYPE_PIPELINE, Op::PIPELINE },
    { FLEXFLOW_OP_TYPE_FUSED_PARALLEL, Op::FUSED_PARALLEL },
  });

flexflow_error_t make_opattrs_error(flexflow_opattrs_error_code_t);

template <typename ExternalEnum>
external_to_internal_t<ExternalEnum> to_internal_impl(ExternalEnum e) {
  return enum_mapping<ExternalEnum>::mapping
    .maybe_at_l(e)
    .or_else([] { throw make_opattrs_error(enum_mapping<ExternalEnum>::err_code); })
    .value();
}

template <typename InternalEnum>
internal_to_external_t<InternalEnum> to_external_impl(InternalEnum i) {
  using Mapping = enum_mapping<internal_to_external_t<InternalEnum>>;

  return Mapping::mapping
    .maybe_at_r(i)
    .or_else([] { throw make_opattrs_error(Mapping::err_code); })
    .value();
}

ParamSync to_internal(flexflow_param_sync_t e) { return to_internal_impl(e); }
flexflow_param_sync_t to_external(ParamSync i) { return to_external_impl(i); }

DataType to_internal(flexflow_datatype_t e) { return to_internal_impl(e); }
flexflow_datatype_t to_external(DataType i) { return to_external_impl(i); }

optional<Activation> to_internal(flexflow_activation_t e) { return to_internal_impl(e); }
flexflow_activation_t to_external(optional<Activation> i) { return to_external_impl(i); }

PoolOp to_internal(flexflow_pool_op_t e) { return to_internal_impl(e); }
flexflow_pool_op_t to_external(PoolOp i) { return to_external_impl(i); }

AggregateOp to_internal(flexflow_aggregate_op_t e) { return to_internal_impl(e); }
flexflow_aggregate_op_t to_external(AggregateOp i) { return to_external_impl(i); }

OperatorType to_internal(flexflow_op_type_t e) { return to_internal_impl(e); }
flexflow_op_type_t to_external(OperatorType i) { return to_external_impl(i); }
