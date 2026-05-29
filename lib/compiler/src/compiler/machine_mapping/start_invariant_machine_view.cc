#include "compiler/machine_mapping/start_invariant_machine_view.h"
#include "compiler/machine_mapping/machine_view.h"
#include "compiler/machine_mapping/machine_view_1d_projection.h"
#include "compiler/machine_mapping/machine_view_2d_projection.h"
#include "op-attrs/operator_task_space.h"
#include "op-attrs/task_space_coordinate.h"
#include "pcg/flat_machine_space_offset.h"
#include "pcg/machine_space_offset.h"
#include "utils/containers/count.h"
#include "utils/containers/filter.h"
#include "utils/containers/scanl.h"
#include "utils/containers/transform.h"
#include "utils/containers/zip.h"
#include "utils/containers/zip3.h"
#include "utils/containers/zip3_strict.h"
#include "utils/nonnegative_int/num_elements.h"
#include "utils/overload.h"

namespace FlexFlow {

MachineView machine_view_from_start_invariant(
    StartInvariantMachineView const &start_inv_mv,
    MachineSpaceCoordinate const &start) {
  return MachineView{
      start,
      start_inv_mv,
  };
}

StartInvariantMachineView
    start_invariant_from_machine_view(MachineView const &mv) {
  return StartInvariantMachineView{mv.start_invariant};
}

nonnegative_int get_expected_task_space_num_dims(
    StartInvariantMachineView const &start_inv_mv) {
  return start_inv_mv.visit<nonnegative_int>(overload{
      [](MachineView1dProjection const &p) -> nonnegative_int {
        return num_elements(p.strides);
      },
      [](MachineView2dProjection const &p) -> nonnegative_int {
        return num_elements(p.dimensions);
      },
  });
}

std::vector<stride_t> start_invariant_mv_get_strides(
    StartInvariantMachineView const &start_inv_mv) {

  return start_inv_mv.visit<std::vector<stride_t>>(overload{
      [](MachineView1dProjection const &p) -> std::vector<stride_t> {
        return p.strides;
      },
      [](MachineView2dProjection const &p) -> std::vector<stride_t> {
        return transform(p.dimensions, [](MachineViewDimension const &dim) {
          return dim.stride;
        });
      },
  });
}

StartInvariantMachineView
    start_invariant_machine_view_from_strides_and_machine_spec_dimensions(
        std::vector<stride_t> const &strides,
        std::vector<MachineSpecificationDimension> const &dims) {
  std::vector<MachineViewDimension> dimensions =
      transform(zip(strides, dims), [&](auto const &p) {
        return MachineViewDimension{p.first, p.second};
      });
  return StartInvariantMachineView{
      MachineView2dProjection{dimensions},
  };
}

UnresolvedMachineSpaceOffset get_machine_space_offset(
    OperatorTaskSpace const &task_space,
    StartInvariantMachineView const &start_inv_machine_view,
    TaskSpaceCoordinate const &coord) {

  ASSERT(get_expected_task_space_num_dims(start_inv_machine_view) ==
             op_task_space_num_dims(task_space),
         "Dimension of StartInvariantMachineView must match dimension of "
         "OperatorTaskSpace",
         start_inv_machine_view,
         task_space);
  ASSERT(op_task_space_num_dims(task_space) ==
         task_space_coord_num_dims(coord));
  ASSERT(operator_task_space_contains_coord(task_space, coord));

  return start_inv_machine_view.visit<UnresolvedMachineSpaceOffset>(overload{
      [&](MachineView1dProjection const &p) -> UnresolvedMachineSpaceOffset {
        FlatMachineSpaceOffset flat_offset =
            projection_1d_get_flat_machine_space_offset(task_space, p, coord);

        return UnresolvedMachineSpaceOffset{flat_offset};
      },
      [&](MachineView2dProjection const &p) -> UnresolvedMachineSpaceOffset {
        return UnresolvedMachineSpaceOffset{
            projection_2d_get_machine_space_offset(task_space, p, coord),
        };
      },
  });
}

std::unordered_set<UnresolvedMachineSpaceOffset> get_machine_space_offsets(
    OperatorTaskSpace const &task,
    StartInvariantMachineView const &start_inv_machine_view) {
  return transform(
      get_task_space_coordinates(task), [&](TaskSpaceCoordinate const &coord) {
        return get_machine_space_offset(task, start_inv_machine_view, coord);
      });
}

} // namespace FlexFlow
