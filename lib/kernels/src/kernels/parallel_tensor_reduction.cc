#include "kernels/parallel_tensor_reduction.h"

namespace FlexFlow {

static std::map<TensorSlotName, GenericTensorAccessorR>
  get_inputs_for_binding(
    std::map<TensorSlotName, EmulatedParallelTensor> const &inputs,
    OperatorAtomicTaskShardBinding const &binding)
{
  std::set<TensorSlotName> input_slots = require_same(
    keys(inputs),
    keys(binding.tensor_coords));

  return generate_map(
    input_slots,
    [&](TensorSlotName slot_name) -> GenericTensorAccessorR {
      ParallelTensorSpaceCoordinate ptensor_coord = binding.tensor_coords.at(slot_name);
      EmulatedParallelTensor t = inputs.at(slot_name);

      return t.shards.at(ptensor_coord);
    });
}

static std::map<TensorSlotName, GenericTensorAccessorR>
  apply_tensor_operation_to_binding(
    std::map<TensorSlotName, EmulatedParallelTensor> const &parallel_inputs,
    OperatorAtomicTaskShardBinding const &binding,
    std::function<std::map<TensorSlotName, GenericTensorAccessorR>(std::map<TensorSlotName, GenericTensorAccessorR) const &> const &operation)
{
  std::map<TensorSlotName, GenericTensorAccessorR> inputs =
    get_inputs_for_binding(inputs, binding);

  return operation(inputs);
}

static EmulatedParallelTensor
  reconstruct_parallel_tensor_for_slot(
    std::map<OperatorAtomicTaskShardBinding, std::map<TensorSlotName, GenericTensorAccessorR>> const &
      raw_outputs_for_bindings,
    TensorSlotName slot_name)
{
  return EmulatedParallelTensor{
    /*shards=*/
      map_keys_and_values(
        [&](OperatorAtomicTaskShardBinding const &b) -> ParallelTensorSpaceCoordinate {
          return b.tensor_coords.at(slot_name);
        },
        [&](std::map<TensorSlotName, GenericTensorAccessorR> const &raw_outputs)
          -> GenericTensorAccessorR
        {
          return raw_outputs.at(slot_name);
        }),
  };
}

static std::map<TensorSlotName, EmulatedParallelTensor>
  reconstruct_parallel_tensors(
    std::map<OperatorAtomicTaskShardBinding, std::map<TensorSlotName, GenericTensorAccessorR>> const &
      raw_outputs_for_bindings)
{
  std::set<TensorSlotName> output_slots =
    require_all_same1(
      transform(
        values(raw_outputs_for_bindings),
        [&](std::map<TensorSlotName, GenericTensorAccessorR> const &raw_outputs)
          -> std::set<TensorSlotName>
        {
          return keys(raw_outputs);
        }));

  return
    generate_map(
      output_slots,
      [&](TensorSlotName slot_name) -> EmulatedParallelTensor {
        return reconstruct_parallel_tensor_for_slot(raw_outputs_for_bindings, slot_name);
      });
}

std::map<TensorSlotName, EmulatedParallelTensor>
  parallelize_tensor_operation(
    std::map<TensorSlotName, EmulatedParallelTensor> const &inputs,
    std::set<OperatorAtomicTaskShardBinding> const &bindings,
    std::function<std::map<TensorSlotName, GenericTensorAccessorR>(std::map<TensorSlotName, GenericTensorAccessorR) const &> const &operation)
{
  std::map<
    OperatorAtomicTaskShardBinding,
    std::map<TensorSlotName, GenericTensorAccessorR>
  > raw_outputs_for_bindings =
    generate_map(
      bindings,
      [&](OperatorAtomicTaskShardBinding const &b)
        -> std::map<TensorSlotName, GenericTensorAccessorR>
      {
        return apply_tensor_operation_to_binding(inputs, binding, operation);
      });

  return reconstruct_parallel_tensors(raw_outputs_for_bindings);
}

static
  EmulatedParallelTensor
    fold_parallel_tensor_dimension(EmulatedParallelTensor const &input,
                                   parallel_tensor_dim_idx_t const &dim_idx,
                                   std::function<GenericTensorAccessorR(GenericTensorAccessorR const &lhs, GenericTensorAccessorR const &rhs)> const &f)
{
  std::set<ParallelTensorSpaceCoordinate>
    input_coords = keys(input.shards);

  std::map<ParallelTensorSpaceCoordinate, ParallelTensorSpaceCoordinate>
    input_coord_to_output_coord_map =
      generate_map(
        [&](ParallelTensorSpaceCoordinate const &input_coord)
          -> ParallelTensorSpaceCoordinate
        {
          ParallelTensorSpaceCoordinate output_coord = input_coord;
          ptensor_coord_component_for_ptensor_dim_idx(output_coord, dim_idx) = 0_n;
          return output_coord;
        });

  ManyToOne<
    ParallelTensorSpaceCoordinate,
    ParallelTensorSpaceCoordinate
  > input_coords_to_output_coord =
    many_to_one_from_map(input_coord_to_output_coord_map);

  nonnegative_int input_space_size =
    num_elements(input_coords_to_output_coord.left_entries());

  nonnegative_int output_space_size =
    num_elements(input_coords_to_output_coord.left_entries());

  ASSERT(input_space_size % output_space_size == 0);

  positive_int degree = positive_int{
    input_space_size / output_space_size,
  };
  if (degree == 1) {
    return input;
  }

  return EmulatedParallelTensor{
    transform_values(
      input_coords_to_output_coord.r_to_l(),
      [&](nonempty_set<ParallelTensorSpaceCoordinate> const &for_single_output_coord)
        -> GenericTensorAccessorR
      {
        std::vector<ParallelTensorSpaceCoordinate>
          ordered_input_coords =
            sorted_by(
              for_single_output_coord,
              [&](ParallelTensorSpaceCoordinate const &input_coord)
                -> nonnegative_int
              {
                return ptensor_coord_component_for_ptensor_dim_idx(input_coord, dim_idx);
              });

        std::set<GenericTensorAccessorR> input_shards =
          transform(
            ordered_input_coords,
            [&](ParallelTensorSpaceCoordinate const &c)
              -> GenericTensorAccessorR
            {
              return input.shards.at(c);
            });

        return foldl1(input_shards, f);
      }
  };
}

EmulatedParallelTensor
  perform_parallel_tensor_reduction(EmulatedParallelTensor const &input,
                                    Allocator &allocator)
{
  return fold_parallel_tensor_dimension(
    ptensor,
    sum_dim_idx(),
    [&](GenericTensorAccessorR const &lhs, GenericTensorAccessorR const &rhs)
      -> GenericTensorAccessorR
    {
      return read_only_accessor_from_write_accessor(
        tensor_accessor_add_to(lhs, rhs, allocator));
    });
}

EmulatedParallelTensor
  perform_parallel_tensor_discard_copy(EmulatedParallelTensor const &)
{
  return fold_parallel_tensor_dimension(
    ptensor,
    discard_copy_dim_idx(),
    [&](GenericTensorAccessorR const &lhs, GenericTensorAccessorR const &rhs)
      -> GenericTensorAccessorR
    {
      ASSERT(accessors_are_equal(lhs, rhs));
      return lhs;
    });
}

GenericTensorAccessorR
  perform_parallel_tensor_combination(EmulatedParallelTensor const &ptensor,
                                      ff_dim_t dim_idx,
                                      Allocator &allocator)
{
  return fold_parallel_tensor_dimension(
    ptensor,
    shard_dim_idx(dim_idx),
    [&](GenericTensorAccessorR const &lhs, GenericTensorAccessorR const &rhs)
      -> GenericTensorAccessorR
    {
      return tensor_accessor_binary_concat(lhs, rhs, dim_idx, allocator);
    });
}

GenericTensorAccessorR
  unparallelize_parallel_tensor_in_dimension(EmulatedParallelTensor const &ptensor,
                                             parallel_tensor_dim_idx_t dim_idx,
                                             Allocator &)
{
  if (dim_idx == sum_dim_idx()) {
    return perform_parallel_tensor_reduction(ptensor);
  } else if (dim_idx == discard_copy_dim_idx()) {
    return perform_parallel_tensor_discard_copy(ptensor);
  } else {
    return perform_parallel_tensor_combination(ptensor, dim_idx.require_shard_dim(), allocator);
  }
}

GenericTensorAccessorR
  unparallelize_parallel_tensor(EmulatedParallelTensor const &ptensor)
{
  std::set<ParallelTensorSpaceCoordinate> ptensor_shard_coords =
    keys(ptensor.shards);

  std::set<parallel_tensor_dim_idx_t> parallel_dim_idxs =
    require_all_same1(
      transform(
        ptensor_shard_coords,
        [&](ParallelTensorSpaceCoordinate const &c) -> std::set<parallel_tensor_dim_idx_t> {
          return get_dim_idxs_in_ptensor_space_coord(c);
        }));

  EmulatedParallelTensor result = ptensor;
  for (parallel_tensor_dim_idx_t parallel_dim_idx : parallel_dim_idxs) {
    result = unparallelize_parallel_tensor_in_dimension(result, parallel_dim_idx, allocator);
  }

  return get_only(result.shards);
}


} // namespace FlexFlow
