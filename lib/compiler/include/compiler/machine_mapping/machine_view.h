#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_MACHINE_VIEW_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_MACHINE_VIEW_H

#include "compiler/machine_mapping/machine_view.dtg.h"
#include "op-attrs/computation_graph_op_attrs.dtg.h"
#include "op-attrs/operator_task_space.dtg.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "op-attrs/pcg_operator_attrs.dtg.h"
#include "op-attrs/task_space_coordinate.dtg.h"
#include "pcg/machine_compute_resource_slice.dtg.h"
#include "pcg/machine_compute_specification.dtg.h"
#include "pcg/mapped_parallel_computation_graph/mapped_operator_task_group.h"
#include "pcg/mapped_parallel_computation_graph/operator_atomic_task_shard_binding.dtg.h"
#include "pcg/operator_space_to_machine_space_mapping.dtg.h"
#include "utils/bidict/bidict.h"
#include <cstddef>
#include <optional>
#include <unordered_set>

namespace FlexFlow {

nonnegative_int mv_get_expected_task_space_num_dims(MachineView const &mv);

std::vector<stride_t> mv_get_strides(MachineView const &mv);

MachineView machine_view_2d_from_strides_and_machine_spec_dimensions(
    MachineSpaceCoordinate const &start,
    std::vector<stride_t> const &strides,
    std::vector<MachineSpecificationDimension> const &dims);

// TODO(@lockshaw)(#pr): Update docs for new design
/**
 * \brief Compute the device (i.e., \ref MachineSpaceCoordinate) where the given
 * subtask (represented by \ref TaskSpaceCoordinate) should be mapped according
 * to the given \ref MachineView.
 *
 * The primary source of complexity here is that we want these mappings to be
 * bijective, i.e., every subtask should map to a unique device so they can all
 * execute in parallel. A naive choice of a propection, such as the following,
 * could yield to a non-bijective mapping: given a \ref MachineMapping with
 * start coordinate \f$zf$ and strides \f$\vec{s}\f$ (we assume for
 * simplicity that the machine space is one-dimensional our start coordinate is
 * just a single integer and so we don't also have to specify the \ref
 * MachineSpecificationDimension), and a \ref TaskSpaceCoordinate
 * \f$\vec{c'}\f$, we could propose computing the \ref MachineSpaceCoordinate
 * \f$c\f$ as $c = z + \vec{c'} \cdot \vec{s}\f$. However,
 * for \f$\vec{s} = (1, 1)\f$, \f$\vec{c'} = (1, 0)\f$ and \f$\vec{c'} = (0,
 * 1)\f$ then both map to the same device!
 *
 * To fix the issue, we can instead multiply each stride by the sizes of all of
 * the prior dimensions. Unfortunately this requires that we are given not only
 * the \ref MachineSpaceCoordinate of the subtask, but also the dimensions of
 * the \ref OperatorTaskSpace it comes from. Letting \f$\vec{d}\f$ denote the
 * dimensions of the \ref OperatorTaskSpace, we get the actual definition used:
 *
 * \f[
 *  c = z + c'_n s_n + c'_{n-1} s_{n-1} d_n s_n + \cdots
 * \f]
 * or more concisely,
 * \f[
 *   \mu_i = s_i \prod_{j=i+1}^{n} d_j s_j
 *   c = z + \vec{c'} \cdot \vec{\mu}
 * \f]
 */
MachineSpaceCoordinate get_machine_space_coordinate(
    OperatorTaskSpace const &operator_task_space,
    MachineView const &machine_view,
    MachineComputeResourceSlice const &machine_space,
    TaskSpaceCoordinate const &task_space_coordinate);

TaskSpaceCoordinate mv_task_space_coord_for_machine_space_coord(
    MachineComputeResourceSlice const &machine_space,
    MachineView const &,
    OperatorTaskSpace const &,
    MachineSpaceCoordinate const &);

OperatorSpaceToMachineSpaceMapping get_coordinate_mapping_for_machine_view(
    OperatorTaskSpace const &operator_task_space,
    MachineComputeResourceSlice const &machine_space,
    MachineView const &machine_view);

std::unordered_set<MachineSpaceCoordinate> get_machine_space_coordinates(
    OperatorTaskSpace const &task,
    MachineComputeResourceSlice const &machine_space,
    MachineView const &mv);

MachineView make_1d_to_2d_machine_view(MachineSpaceCoordinate const &start,
                                       MachineSpecificationDimension const &dim,
                                       stride_t stride);

MachineView make_single_device_machine_view(MachineSpaceCoordinate const &);

OperatorAtomicTaskShardBinding
    operator_atomic_task_shard_binding_from_machine_view(
        ComputationGraphOpAttrs const &,
        std::vector<ParallelTensorDimDegrees> const &,
        MachineView const &,
        MachineSpaceCoordinate const &);

MappedOperatorTaskGroup mapped_operator_task_group_from_machine_view(
    PCGOperatorAttrs const &,
    std::unordered_map<TensorSlotName, ParallelTensorDimDegrees> const &,
    MachineComputeResourceSlice const &machine_space,
    MachineView const &);

bidict<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate>
    get_tensor_shard_to_device_coord_mapping(ComputationGraphOpAttrs const &,
                                             MachineView const &);

} // namespace FlexFlow

#endif
