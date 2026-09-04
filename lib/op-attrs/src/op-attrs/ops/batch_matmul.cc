#include "op-attrs/ops/batch_matmul.h"
#include "op-attrs/ff_ordered/ff_ordered_concat.h"
#include "op-attrs/ff_ordered/ff_ordered_slice.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "op-attrs/parallel_tensor_dim_idx_t.h"
#include "op-attrs/parallel_tensor_dims.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_dims.h"
#include "utils/containers/require_same.h"
#include <libassert/assert.hpp>
#include "op-attrs/operator_task_space.h"
#include "op-attrs/operator_atomic_task_shard_binding.dtg.h"
#include "op-attrs/relative_ff_dim_t.h"
#include "op-attrs/parallel_tensor_space_coordinate.h"
#include "utils/containers/filtrans.h"
#include "utils/containers/binary_cartesian_product.h"
#include "utils/bidict/algorithms/bidict_from_pairs.h"
#include "utils/containers/transform.h"
#include "utils/containers/require_three_keys.h"
#include "op-attrs/parallel_tensor_space_to_parallel_tensor_space_biunique_mapping.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_biunique_mapping.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_biunique_mapping.dtg.h"

namespace FlexFlow {

TensorShape batch_matmul_get_output_shape(BatchMatmulAttrs const &,
                                          TensorShape const &lhs,
                                          TensorShape const &rhs) {
  positive_int lhs_row_dim = dim_at_idx(lhs.dims, relative_ff_dim_t{-2});
  positive_int lhs_col_dim = dim_at_idx(lhs.dims, relative_ff_dim_t{-1});

  positive_int rhs_row_dim = dim_at_idx(rhs.dims, relative_ff_dim_t{-2});
  positive_int rhs_col_dim = dim_at_idx(rhs.dims, relative_ff_dim_t{-1});

  ASSERT(lhs_col_dim == rhs_row_dim);

  auto get_leading_dims = [](TensorShape const &s) -> FFOrdered<positive_int> {
    return ff_ordered_slice(
        s.dims.ff_ordered, relative_ff_dim_t{0}, relative_ff_dim_t{-2});
  };

  FFOrdered<positive_int> leading_dims =
      require_same(get_leading_dims(lhs), get_leading_dims(rhs));

  return TensorShape{
      TensorDims{
          ff_ordered_concat(leading_dims,
                            FFOrdered<positive_int>{
                                lhs_row_dim,
                                rhs_col_dim,
                            }),
      },
      /*data_type=*/require_same(lhs.data_type, rhs.data_type),
  };
}

StandardOperatorTaskGroup batch_matmul_get_task_group(
    LinearAttrs const &attrs,
    ParallelTensorDimDegrees const &lhs_input_degrees,
    ParallelTensorDimDegrees const &rhs_input_degrees)
{
  num_ptensor_shard_dims_t num_input_shard_dims =
    require_same(
      get_ptensor_dim_degrees_num_shard_dims(lhs_input_degrees),
      get_ptensor_dim_degrees_num_shard_dims(rhs_input_degrees));

  return StandardOperatorTaskGroup{
    filtrans(
      binary_cartesian_product(
        get_parallel_tensor_space_coordinates(lhs_input_degrees),
        get_parallel_tensor_space_coordinates(rhs_input_degrees)),
      [&](std::pair<ParallelTensorSpaceCoordinate, ParallelTensorSpaceCoordinate> const &coords)
        -> std::optional<AbstractedOperatorAtomicTaskShardBinding>
      {
        ParallelTensorSpaceCoordinate lhs_input_coord = coords.first;
        ParallelTensorSpaceCoordinate rhs_input_coord = coords.second;

        parallel_tensor_dim_idx_t sum_dim = sum_dim_idx();
        parallel_tensor_dim_idx_t discard_copy_dim = discard_copy_dim_idx();

        std::set<parallel_tensor_dim_idx_t> leading_dims =
          shard_dim_idxs_for_interval(0, -2, input_num_shard_dims);

        parallel_tensor_dim_idx_t row_dim =
          shard_dim_idx_for_relative(-2, input_num_shard_dims);

        parallel_tensor_dim_idx_t col_dim =
          shard_dim_idx_for_relative(-1, input_num_shard_dims);

        // c
        OrthotopeBoundedCoord data_parallelism_coord =
          require_same(
            orthotope_bounded_coord_for_ptensor_dims(
              lhs_input_degrees,
              lhs_input_coord,
              leading_dims),
            orthotope_bounded_coord_for_ptensor_dims(
              rhs_input_degrees,
              rhs_input_coord,
              leading_dims));

        // h
        OrthotopeBoundedCoord output_column_parallelism_coord =
          require_same(
            orthotope_bounded_coord_for_ptensor_dims(
              rhs_input_degrees,
              rhs_input_coord,
              std::set{col_dim});

        // d
        OrthotopeBoundedCoord output_row_parallelism_coord =
            orthotope_bounded_coord_for_ptensor_dims(
              lhs_input_degrees,
              lhs_input_coord,
              std::set{row_dim});

        // e
        OrthotopeBoundedCoord reduction_parallelism =
          require_same(
            orthotope_bounded_coord_for_ptensor_dims(
              lhs_input_degrees,
              lhs_input_coord,
              std::set{col_dim}),
            orthotope_bounded_coord_for_ptensor_dims(
              rhs_input_degrees,
              rhs_input_coord,
              std::set{row_dim}));

        // a
        OrthotopeBoundedCoord lhs_preexisting_sum_parallelism_coord =
            orthotope_bounded_coord_for_ptensor_dims(
              lhs_input_degrees,
              lhs_input_coord,
              std::set{sum_dim});

        // f
        OrthotopeBoundedCoord rhs_preexisting_sum_parallelism_coord =
            orthotope_bounded_coord_for_ptensor_dims(
              rhs_input_degrees,
              rhs_input_coord,
              std::set{sum_dim});

        return AbstractedOperatorAtomicTaskShardBinding{
          /*tensor_corods=*/{
            {
              TensorSlotName::LHS_INPUT,
              lhs_input_coord,
            },
            {
              TensorSlotName::RHS_INPUT,
              rhs_input_coord,
            {
              TensorSlotName::OUTPUT,
              parallel_tensor_space_coordinate_from_ff_ordered(
                /*sum_degree=*/reduction_parallelism_component,
                /*discard_copy_degree=*/0_n,
                /*shard_coords=*/std::vector<nonnegative_int>{
                  data_parallelism_component,
                  output_channel_parallelism_component,
                }),
            },
          },
          /*task_coord=*/make_task_space_coordinate({
            data_parallelism_component,
            reduction_parallelism_component,
            output_channel_parallelism_component,
          }),
        };
      }),
  };
}

ParallelTensorDimDegrees batch_matmul_get_output_parallel_dim_degrees(
    BatchMatmulAttrs const &attrs,
    ParallelTensorDimDegrees const &lhs,
    ParallelTensorDimDegrees const &rhs) {
  ASSERT(get_ptensor_dim_degrees_num_shard_dims(lhs) ==
         num_ptensor_shard_dims_t{3_n});
  ASSERT(get_ptensor_dim_degrees_num_shard_dims(rhs) ==
         num_ptensor_shard_dims_t{3_n});

  positive_int batch_degree = require_same(
      get_degree_for_parallel_tensor_dim_idx(lhs, shard_dim_idx(ff_dim_t{0_n})),
      get_degree_for_parallel_tensor_dim_idx(rhs,
                                             shard_dim_idx(ff_dim_t{0_n})));

  positive_int reduction_parallelism_degree = require_same(
      get_degree_for_parallel_tensor_dim_idx(lhs, shard_dim_idx(ff_dim_t{2_n})),
      get_degree_for_parallel_tensor_dim_idx(rhs,
                                             shard_dim_idx(ff_dim_t{1_n})));

  positive_int lhs_row_degree =
      get_degree_for_parallel_tensor_dim_idx(lhs, shard_dim_idx(ff_dim_t{1_n}));

  positive_int rhs_column_degree =
      get_degree_for_parallel_tensor_dim_idx(rhs, shard_dim_idx(ff_dim_t{2_n}));

  ASSERT(lhs_row_degree * lhs.sum_degree.value ==
         rhs.discard_copy_degree.value);

  ASSERT(rhs_column_degree * rhs.sum_degree.value ==
         lhs.discard_copy_degree.value);

  return ParallelTensorDimDegrees{
      /*sum_degree=*/SumDegree{
          reduction_parallelism_degree * lhs.sum_degree.value *
              rhs.sum_degree.value,
      },
      /*discard_copy_degree=*/DiscardCopyDegree{1_p},
      /*shard_degrees=*/
      FFOrdered<positive_int>{
          batch_degree,
          lhs_row_degree,
          rhs_column_degree,
      },
  };
}

ParallelTensorShape
    batch_matmul_get_output_parallel_shape(BatchMatmulAttrs const &attrs,
                                           ParallelTensorShape const &lhs,
                                           ParallelTensorShape const &rhs) {
  TensorShape output_shape = batch_matmul_get_output_shape(
      attrs, get_reduced_shape(lhs), get_reduced_shape(rhs));

  ParallelTensorDimDegrees output_degrees =
      batch_matmul_get_output_parallel_dim_degrees(
          attrs, get_parallel_degrees(lhs), get_parallel_degrees(rhs));

  return lift_to_parallel_with_degrees(output_shape, output_degrees);
}

OperatorTaskSpace batch_matmul_get_operator_task_space(
    BatchMatmulAttrs const &attrs,
    ParallelTensorDimDegrees const &lhs,
    ParallelTensorDimDegrees const &rhs)
{
  ParallelTensorDimDegrees output_degrees =
      batch_matmul_get_output_parallel_dim_degrees(attrs, lhs, rhs);

  return get_operator_task_space_matching_parallel_tensor_dim_degrees(
      output_degrees);
}

std::set<OperatorAtomicTaskShardBinding>
    batch_matmul_get_parallel_task_signatures(BatchMatmulAttrs const &attrs,
                                ParallelTensorDimDegrees const &lhs_input_degrees,
                                ParallelTensorDimDegrees const &rhs_input_degrees) {

  num_tensor_dims_t num_input_dims =
    require_same(
      get_ptensor_dim_degrees_num_tensor_dims(lhs_input_degrees),
      get_ptensor_dim_degrees_num_tensor_dims(rhs_input_degrees));

  ASSERT(
    num_input_dims == num_tensor_dims_t{3_n},
    "Currently batch matmul is limited to 3d inputs. "
    "If you need support for other input shapes, please create an issue."
  );

  ParallelTensorDimDegrees output_degrees =
    batch_matmul_get_output_parallel_dim_degrees(attrs, lhs_input_degrees, rhs_input_degrees);

  std::set<ParallelTensorSpaceCoordinate> lhs_coords = get_parallel_tensor_space_coordinates(lhs_input_degrees);
  std::set<ParallelTensorSpaceCoordinate> rhs_coords = get_parallel_tensor_space_coordinates(rhs_input_degrees);

  auto get_shard_component = [&](ParallelTensorSpaceCoordinate const &coord, int idx) -> nonnegative_int {
    ff_dim_t dim = ff_dim_t_from_relative_ff_dim_t(
      relative_ff_dim_t{idx},
      num_input_dims);

    return ptensor_coord_component_for_ptensor_dim_idx(coord, shard_dim_idx(dim));
  };

  int r_dim_idx = -2;
  int c_dim_idx = -1;
  int leading_dim_idx = 0;

  std::set<OperatorAtomicTaskShardBinding> task_shard_sigs =
    filtrans(
      binary_cartesian_product(lhs_coords, rhs_coords),
      [&](std::pair<ParallelTensorSpaceCoordinate, ParallelTensorSpaceCoordinate> const &lhs_rhs_coord_pair)
        -> std::optional<OperatorAtomicTaskShardBinding>
      {
        ParallelTensorSpaceCoordinate lhs_coord = lhs_rhs_coord_pair.first;
        ParallelTensorSpaceCoordinate rhs_coord = lhs_rhs_coord_pair.second;

        positive_int lhs_sum_degree = lhs_input_degrees.sum_degree.value;

        positive_int rhs_sum_degree = rhs_input_degrees.sum_degree.value;

        nonnegative_int leading_dim_component =
          require_same(
            get_shard_component(lhs_coord, leading_dim_idx),
            get_shard_component(rhs_coord, leading_dim_idx));

        nonnegative_int lhs_row_component = get_shard_component(lhs_coord, r_dim_idx);
        nonnegative_int rhs_row_component = get_shard_component(rhs_coord, r_dim_idx);

        nonnegative_int lhs_col_component = get_shard_component(lhs_coord, c_dim_idx);
        nonnegative_int rhs_col_component = get_shard_component(rhs_coord, c_dim_idx);

        if (lhs_col_component != rhs_row_component) {
          return std::nullopt;
        }

        ParallelTensorSpaceCoordinate output_coord =
          ParallelTensorSpaceCoordinate{
            /*sum_component=*/
              lhs_col_component * lhs_sum_degree * rhs_sum_degree
              + lhs_coord.sum_component * rhs_sum_degree
              + rhs_coord.sum_component,
            /*discard_copy_component=*/0_n,
            /*shard_components=*/FFOrdered<nonnegative_int>{
              leading_dim_component,
              lhs_row_component,
              rhs_col_component,
            },
          };

        return OperatorAtomicTaskShardBinding{
          /*tensor_coords=*/{
            {
              TensorSlotName::LHS_INPUT,
              lhs_coord,
            },
            {
              TensorSlotName::RHS_INPUT,
              rhs_coord,
            },
            {
              TensorSlotName::OUTPUT,
              output_coord,
            },
          },
        };
      });

  positive_int total_parallel_degree =
    require_same(
      get_total_degree_of_ptensor_dim_degrees(lhs_input_degrees),
      get_total_degree_of_ptensor_dim_degrees(rhs_input_degrees),
      get_total_degree_of_ptensor_dim_degrees(output_degrees));

  ASSERT(task_shard_sigs.size() == total_parallel_degree);

  return task_shard_sigs;
}

static ParallelTensorSpaceToParallelTensorSpaceBiuniqueMapping
    batch_matmul_get_lhs_input_to_output_mapping(BatchMatmulAttrs const &attrs,
                                ParallelTensorDimDegrees const &lhs_input_degrees,
                                ParallelTensorDimDegrees const &rhs_input_degrees) {

  ParallelTensorDimDegrees output_degrees =
    batch_matmul_get_output_parallel_dim_degrees(attrs, lhs_input_degrees, rhs_input_degrees);

  std::set<OperatorAtomicTaskShardBinding> task_shard_sigs =
    batch_matmul_get_parallel_task_signatures(attrs, lhs_input_degrees, rhs_input_degrees);

  return parallel_tensor_space_biunique_mapping_from_coord_mapping(
      /*coord_mapping=*/
        bidict_from_pairs(
          transform(
            task_shard_sigs,
            [&](OperatorAtomicTaskShardBinding const &b)
              -> std::pair<ParallelTensorSpaceCoordinate, ParallelTensorSpaceCoordinate>
            {
              auto [lhs_coord, rhs_coord, out_coord] =
                require_three_keys(b.tensor_coords,
                                   TensorSlotName::LHS_INPUT,
                                   TensorSlotName::RHS_INPUT,
                                   TensorSlotName::OUTPUT);

              return {lhs_coord, out_coord};
            })),
      /*l_degrees=*/lhs_input_degrees,
      /*r_degrees=*/output_degrees);
}

static ParallelTensorSpaceToParallelTensorSpaceBiuniqueMapping
    batch_matmul_get_rhs_input_to_output_mapping(BatchMatmulAttrs const &attrs,
                                ParallelTensorDimDegrees const &lhs_input_degrees,
                                ParallelTensorDimDegrees const &rhs_input_degrees) {

  ParallelTensorDimDegrees output_degrees =
    batch_matmul_get_output_parallel_dim_degrees(attrs, lhs_input_degrees, rhs_input_degrees);

  std::set<OperatorAtomicTaskShardBinding> task_shard_sigs =
    batch_matmul_get_parallel_task_signatures(attrs, lhs_input_degrees, rhs_input_degrees);

  return parallel_tensor_space_biunique_mapping_from_coord_mapping(
      /*coord_mapping=*/
        bidict_from_pairs(
          transform(
            task_shard_sigs,
            [&](OperatorAtomicTaskShardBinding const &b)
              -> std::pair<ParallelTensorSpaceCoordinate, ParallelTensorSpaceCoordinate>
            {
              auto [lhs_coord, rhs_coord, out_coord] =
                require_three_keys(b.tensor_coords,
                                   TensorSlotName::LHS_INPUT,
                                   TensorSlotName::RHS_INPUT,
                                   TensorSlotName::OUTPUT);

              return {rhs_coord, out_coord};
            })),
      /*l_degrees=*/rhs_input_degrees,
      /*r_degrees=*/output_degrees);
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping batch_matmul_get_operator_to_lhs_input_mapping(
    BatchMatmulAttrs const &attrs,
    ParallelTensorDimDegrees const &lhs,
    ParallelTensorDimDegrees const &rhs)
{
  ParallelTensorSpaceToParallelTensorSpaceBiuniqueMapping inp_to_out =
      batch_matmul_get_lhs_input_to_output_mapping(attrs, lhs, rhs);

  ParallelTensorSpaceToParallelTensorSpaceBiuniqueMapping out_to_inp =
      invert_parallel_tensor_space_biunique_mapping(inp_to_out);

  OperatorSpaceToParallelTensorSpaceBiuniqueMapping op_to_out =
      batch_matmul_get_operator_to_output_mapping(attrs, lhs, rhs);

  return operator_ptensor_space_biunique_mapping_from_composition(op_to_out, out_to_inp);
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping batch_matmul_get_operator_to_rhs_input_mapping(
    BatchMatmulAttrs const &attrs,
    ParallelTensorDimDegrees const &lhs,
    ParallelTensorDimDegrees const &rhs)
{
  ParallelTensorSpaceToParallelTensorSpaceBiuniqueMapping inp_to_out =
      batch_matmul_get_rhs_input_to_output_mapping(attrs, lhs, rhs);

  ParallelTensorSpaceToParallelTensorSpaceBiuniqueMapping out_to_inp =
      invert_parallel_tensor_space_biunique_mapping(inp_to_out);

  OperatorSpaceToParallelTensorSpaceBiuniqueMapping op_to_out =
      batch_matmul_get_operator_to_output_mapping(attrs, lhs, rhs);

  return operator_ptensor_space_biunique_mapping_from_composition(op_to_out, out_to_inp);
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping batch_matmul_get_operator_to_output_mapping(
    BatchMatmulAttrs const &attrs,
    ParallelTensorDimDegrees const &lhs,
    ParallelTensorDimDegrees const &rhs)
{
  ParallelTensorDimDegrees output_degrees =
      batch_matmul_get_output_parallel_dim_degrees(attrs, lhs, rhs);

  return get_identity_biunique_mapping(
      batch_matmul_get_operator_task_space(attrs, lhs, rhs),
      output_degrees);
}

} // namespace FlexFlow
