#include "compiler/machine_mapping/machine_view.h"
#include "compiler/machine_mapping/machine_view_dimension.dtg.h"
#include "compiler/machine_mapping/start_invariant_machine_view.h"
#include "compiler/machine_mapping/stride_t.dtg.h"
#include "op-attrs/get_operator_space_to_parallel_tensor_space_mappings.h"
#include "op-attrs/get_operator_task_space.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_mapping.h"
#include "op-attrs/operator_task_space.dtg.h"
#include "op-attrs/operator_task_space.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "op-attrs/task_space_coordinate.h"
#include "op-attrs/tensor_role.dtg.h"
#include "pcg/machine_compute_resource_slice.h"
#include "pcg/machine_compute_specification.h"
#include "pcg/machine_space_coordinate.dtg.h"
#include "pcg/machine_space_offset.dtg.h"
#include "pcg/machine_specification.dtg.h"
#include "pcg/machine_specification_dimension.dtg.h"
#include "pcg/unresolved_machine_space_offset.h"
#include "utils/bidict/generate_bidict.h"
#include "utils/containers/contains.h"
#include "utils/containers/count.h"
#include "utils/containers/filter.h"
#include "utils/containers/get_only.h"
#include "utils/containers/scanl.h"
#include "utils/containers/sum.h"
#include "utils/containers/transform.h"
#include "utils/containers/zip3_strict.h"
#include "utils/containers/zip_with_strict.h"
#include "utils/nonnegative_int/nonnegative_range.h"
#include "utils/nonnegative_int/num_elements.h"
#include "op-attrs/shape_inference.h"
#include "utils/containers/merge_disjoint_maps.h"

namespace FlexFlow {

nonnegative_int mv_get_expected_task_space_num_dims(MachineView const &mv) {
  return num_elements(mv_get_strides(mv));
}

std::vector<stride_t> mv_get_strides(MachineView const &mv) {
  return start_invariant_mv_get_strides(mv.start_invariant);
}

MachineView machine_view_2d_from_strides_and_machine_spec_dimensions(
    MachineSpaceCoordinate const &start,
    std::vector<stride_t> const &strides,
    std::vector<MachineSpecificationDimension> const &dims) {
  ASSERT(strides.size() == dims.size());
  std::vector<MachineViewDimension> dimensions = zip_with_strict(
      strides, dims, [](stride_t s, MachineSpecificationDimension d) {
        return MachineViewDimension{s, d};
      });
  return MachineView{
      start,
      StartInvariantMachineView{
          MachineView2dProjection{
              dimensions,
          },
      },
  };
}

MachineSpaceCoordinate get_machine_space_coordinate(
    OperatorTaskSpace const &task_space,
    MachineView const &machine_view,
    MachineComputeResourceSlice const &machine_space,
    TaskSpaceCoordinate const &coord) {

  ASSERT(mv_get_expected_task_space_num_dims(machine_view) ==
             op_task_space_num_dims(task_space),
         "Dimension of MachineView must match dimension of OperatorTaskSpace",
         machine_view,
         task_space);
  ASSERT(op_task_space_num_dims(task_space) ==
         task_space_coord_num_dims(coord));
  ASSERT(operator_task_space_contains_coord(task_space, coord));

  UnresolvedMachineSpaceOffset offset =
      get_machine_space_offset(task_space, machine_view.start_invariant, coord);

  return offset_machine_space_coordinate_by_unresolved(
      machine_space, machine_view.start, offset);
}

TaskSpaceCoordinate mv_task_space_coord_for_machine_space_coord(
    MachineComputeResourceSlice const &machine_space,
    MachineView const &machine_view,
    OperatorTaskSpace const &operator_task_space,
    MachineSpaceCoordinate const &machine_space_coord) {
  OperatorSpaceToMachineSpaceMapping mapping =
      get_coordinate_mapping_for_machine_view(
          operator_task_space, machine_space, machine_view);

  return mapping.raw_mapping.at_r(machine_space_coord);
}

OperatorSpaceToMachineSpaceMapping get_coordinate_mapping_for_machine_view(
    OperatorTaskSpace const &operator_task_space,
    MachineComputeResourceSlice const &machine_space,
    MachineView const &machine_view) {

  return OperatorSpaceToMachineSpaceMapping{
      /*raw_mapping=*/generate_bidict(
          get_task_space_coordinates(operator_task_space),
          [&](TaskSpaceCoordinate const &task_space_coord) {
            return get_machine_space_coordinate(
                /*operator_task_space=*/operator_task_space,
                /*machine_view=*/machine_view,
                /*machine_space=*/machine_space,
                /*task_space_coordinate=*/task_space_coord);
          }),
      /*operator_task_space=*/operator_task_space,
  };
}

std::set<MachineSpaceCoordinate> get_machine_space_coordinates(
    OperatorTaskSpace const &task_space,
    MachineComputeResourceSlice const &machine_space,
    MachineView const &machine_view) {

  ASSERT(op_task_space_num_dims(task_space) ==
         mv_get_expected_task_space_num_dims(machine_view));

  return transform(get_task_space_coordinates(task_space),
                   [&](TaskSpaceCoordinate const &coord) {
                     return get_machine_space_coordinate(
                         task_space, machine_view, machine_space, coord);
                   });
}

MachineView make_1d_to_2d_machine_view(MachineSpaceCoordinate const &start,
                                       MachineSpecificationDimension const &dim,
                                       stride_t stride) {

  return machine_view_2d_from_strides_and_machine_spec_dimensions(
      start, {stride}, {dim});
}

MachineView
    make_single_device_machine_view(MachineSpaceCoordinate const &coord) {
  return machine_view_2d_from_strides_and_machine_spec_dimensions(
      coord, {}, {});
}

static OperatorAtomicTaskShardBinding
    operator_atomic_task_shard_binding_from_machine_view(
        PCGOperatorAttrs const &op_attrs,
        std::map<TensorSlotName, ParallelTensorDimDegrees> const
            &inputs_dim_degrees,
        MachineView const &machine_view,
        MachineComputeResourceSlice const &machine_space,
        MachineSpaceCoordinate const &machine_space_coord) {
  OperatorTaskSpace op_task_space =
      get_operator_task_space(op_attrs, inputs_dim_degrees);

  TaskSpaceCoordinate task_space_coord =
      mv_task_space_coord_for_machine_space_coord(
          machine_space, machine_view, op_task_space, machine_space_coord);

  std::map<TensorSlotName, OperatorSpaceToParallelTensorSpaceMapping> mappings =
      get_operator_to_ptensor_mappings(op_attrs, inputs_dim_degrees);

  std::map<TensorSlotName, ParallelTensorDimDegrees> weights_dim_degrees = 
    infer_weight_degrees(op_attrs, inputs_dim_degrees);

  std::map<TensorSlotName, ParallelTensorDimDegrees> outputs_dim_degrees = 
    infer_output_degrees(op_attrs, inputs_dim_degrees);

  auto compute_ptensor_coords = [&](std::map<TensorSlotName, ParallelTensorDimDegrees> const &dim_degrees) 
    -> std::map<TensorSlotName, ParallelTensorSpaceCoordinate>
  {
    return generate_map(
        keys(dim_degrees),
        [&](TensorSlotName const &slot_name)
            -> ParallelTensorSpaceCoordinate {
          num_ptensor_shard_dims_t num_shard_dims =
              get_ptensor_dim_degrees_num_shard_dims(
                  dim_degrees.at(slot_name));

          return ptensor_coord_for_task_space_coord(
              mappings.at(slot_name), task_space_coord, num_shard_dims);
        });
  };

  return OperatorAtomicTaskShardBinding{
    /*tensor_coords=*/merge_disjoint_maps(std::vector{
      compute_ptensor_coords(inputs_dim_degrees), 
      compute_ptensor_coords(weights_dim_degrees),
      compute_ptensor_coords(outputs_dim_degrees),
    }),
  };
}

MappedOperatorTaskGroup mapped_operator_task_group_from_machine_view(
    PCGOperatorAttrs const &op_attrs,
    std::map<TensorSlotName, ParallelTensorDimDegrees> const
        &inputs_dim_degrees,
    MachineComputeResourceSlice const &machine_space,
    MachineView const &machine_view) {

  OperatorTaskSpace op_task_space =
      get_operator_task_space(op_attrs, inputs_dim_degrees);

  return MappedOperatorTaskGroup{
      generate_bidict(
          get_machine_space_coordinates(
              op_task_space, machine_space, machine_view),
          [&](MachineSpaceCoordinate const &machine_space_coord)
            -> OperatorAtomicTaskShardBinding
          {
            return operator_atomic_task_shard_binding_from_machine_view(
                op_attrs,
                inputs_dim_degrees,
                machine_view,
                machine_space,
                machine_space_coord);
          }),
  };
}

bidict<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate>
    get_tensor_shard_to_device_coord_mapping(ComputationGraphOpAttrs const &,
                                             MachineView const &) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
