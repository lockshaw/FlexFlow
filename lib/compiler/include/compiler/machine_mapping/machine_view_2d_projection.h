#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_MACHINE_VIEW_2D_PROJECTION_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_MACHINE_VIEW_2D_PROJECTION_H

#include "compiler/machine_mapping/machine_view_2d_projection.dtg.h"
#include "op-attrs/operator_task_space.dtg.h"
#include "op-attrs/task_space_coordinate.dtg.h"
#include "pcg/machine_space_offset.dtg.h"

namespace FlexFlow {

std::vector<MachineSpecificationDimension>
  projection_2d_get_target_dimensions(MachineView2dProjection const &);

std::vector<stride_t> projection_2d_get_strides(MachineView2dProjection const &);

MachineSpaceOffset
    projection_2d_get_machine_space_offset(OperatorTaskSpace const &task_space,
                             MachineView2dProjection const &projection,
                             TaskSpaceCoordinate const &coord);

} // namespace FlexFlow

#endif
