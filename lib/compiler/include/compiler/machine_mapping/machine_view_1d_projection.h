#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_MACHINE_VIEW_1D_PROJECTION_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_MACHINE_VIEW_1D_PROJECTION_H

#include "compiler/machine_mapping/machine_view_1d_projection.dtg.h"
#include "op-attrs/task_space_coordinate.dtg.h"
#include "op-attrs/operator_task_space.dtg.h"
#include "pcg/flat_machine_space_offset.dtg.h"

namespace FlexFlow {

std::vector<stride_t> projection_1d_get_strides(MachineView1dProjection const &);

FlatMachineSpaceOffset
    projection_1d_get_flat_machine_space_offset(OperatorTaskSpace const &task,
                             MachineView1dProjection const &projection,
                             TaskSpaceCoordinate const &coordinates);

} // namespace FlexFlow

#endif
