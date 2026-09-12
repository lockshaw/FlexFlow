#include "op-attrs/operator_task_space.h"
#include "op-attrs/operator_task_space.dtg.h"
#include "op-attrs/operator_task_space_dim_idx_t.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "op-attrs/parallel_tensor_dim_idx_t.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/containers/cartesian_product.h"
#include "utils/containers/extend.h"
#include "utils/containers/maximum.h"
#include "utils/containers/product.h"
#include "utils/containers/range.h"
#include "utils/containers/set_of.h"
#include "utils/containers/transform.h"
#include "utils/containers/vector_of.h"
#include "utils/fmt/set.h"
#include "utils/nonnegative_int/nonnegative_range.h"
#include "utils/nonnegative_int/num_elements.h"
#include "utils/orthotope/dim_domain.h"
#include "utils/orthotope/dim_ordering.h"
#include "utils/orthotope/minimal_dim_domain.h"
#include "utils/orthotope/minimal_orthotope.h"
#include "utils/orthotope/orthotope.dtg.h"
#include "utils/orthotope/orthotope.h"
#include "op-attrs/task_space_coordinate.h"
#include "utils/orthotope/dim_coord.h"

namespace FlexFlow {

OperatorTaskSpace trivial_op_task_space() {
  return OperatorTaskSpace{MinimalOrthotope{{}}};
}

std::set<operator_task_space_dim_idx_t>
    operator_task_space_get_dim_idxs(OperatorTaskSpace const &op_task_space) {
  return get_minimal_domain_dims(
      minimal_dim_domain_from_operator_task_space(op_task_space));
}

std::set<TaskSpaceCoordinate>
    get_task_space_coordinates(OperatorTaskSpace const &task) {

  std::vector<std::vector<nonnegative_int>> coordinate_ranges =
      transform(task.degrees.dims, [&](int_ge_two num_points) {
        return nonnegative_range(num_points.nonnegative_int_from_int_ge_two());
      });

  std::set<std::vector<nonnegative_int>> raw_coordinates =
      set_of(cartesian_product(coordinate_ranges));
  std::set<TaskSpaceCoordinate> task_space_coordinates =
      transform(raw_coordinates, [](std::vector<nonnegative_int> const &point) {
        return TaskSpaceCoordinate{OrthotopeCoord{point}};
      });
  return task_space_coordinates;
}

bool operator_task_space_contains_coord(OperatorTaskSpace const &task_space,
                                        TaskSpaceCoordinate const &coord) {
  return contains(get_task_space_coordinates(task_space), coord);
}

TaskSpaceCoordinate
    get_task_space_maximum_coordinate(OperatorTaskSpace const &task) {
  return maximum(get_task_space_coordinates(task));
}

nonnegative_int op_task_space_num_dims(OperatorTaskSpace const &op_task_space) {
  return minimal_orthotope_get_num_dims(op_task_space.degrees);
}

positive_int num_tasks(OperatorTaskSpace const &op_task_space) {
  return minimal_orthotope_get_volume(op_task_space.degrees);
}

positive_int
    op_task_space_dim_size_for_idx(OperatorTaskSpace const &op_task_space,
                                   operator_task_space_dim_idx_t idx) {
  int_ge_two dim_size =
      op_task_space.degrees.dims.at(idx.raw_idx.int_from_nonnegative_int());

  return dim_size.positive_int_from_int_ge_two();
}

MinimalDimDomain<operator_task_space_dim_idx_t>
    minimal_dim_domain_from_operator_task_space(
        OperatorTaskSpace const &operator_task_space) {

  MinimalOrthotope minimal_orthotope = operator_task_space.degrees;

  return minimal_dim_domain_from_minimal_orthotope(
      minimal_orthotope,
      set_of(operator_task_space_dim_idx_range(
          minimal_orthotope_get_num_dims(minimal_orthotope))),
      get_operator_task_space_dim_ordering());
}

OperatorTaskSpace operator_task_space_from_minimal_dim_domain(
    MinimalDimDomain<operator_task_space_dim_idx_t> const &minimal_dim_domain) {

  return OperatorTaskSpace{
      minimal_orthotope_from_minimal_dim_domain(
          minimal_dim_domain, get_operator_task_space_dim_ordering()),
  };
}

DimOrdering<operator_task_space_dim_idx_t>
    get_operator_task_space_dim_ordering() {
  return make_default_dim_ordering<operator_task_space_dim_idx_t>();
}

OperatorTaskSpace get_operator_task_space_matching_parallel_tensor_dim_degrees(
    ParallelTensorDimDegrees const &dim_degrees) {
  return OperatorTaskSpace{
      minimal_orthotope_from_minimal_dim_domain(
          minimal_dim_domain_from_parallel_tensor_dim_degrees(dim_degrees),
          get_parallel_tensor_dim_ordering()),
  };
}

OperatorTaskSpace
  smallest_operator_task_space_for_coord_set(
     std::set<TaskSpaceCoordinate> const &coord_set)
{
  std::set<DimCoord<operator_task_space_dim_idx_t>>
    dim_coord_set = transform(coord_set,
                              [&](TaskSpaceCoordinate const &c)
                                -> DimCoord<operator_task_space_dim_idx_t>
                              {
                                return dim_coord_from_task_space_coordinate(c);
                              });

  DimDomain<operator_task_space_dim_idx_t>
    dim_domain = smallest_dim_domain_for_coord_set(dim_coord_set);

  return operator_task_space_from_minimal_dim_domain(
    require_dim_domain_is_minimal(dim_domain));
}

std::optional<OperatorTaskSpace>
  strict_operator_task_space_for_coord_set(
     std::set<TaskSpaceCoordinate> const &coord_set)
{
  std::set<DimCoord<operator_task_space_dim_idx_t>>
    dim_coord_set = transform(coord_set,
                              [&](TaskSpaceCoordinate const &c)
                                -> DimCoord<operator_task_space_dim_idx_t>
                              {
                                return dim_coord_from_task_space_coordinate(c);
                              });

  std::optional<DimDomain<operator_task_space_dim_idx_t>>
    dim_domain = strict_dim_domain_for_coord_set(dim_coord_set);

  if (dim_domain.has_value()) {
    return operator_task_space_from_minimal_dim_domain(
      require_dim_domain_is_minimal(dim_domain.value()));
  } else {
    return std::nullopt;
  }
}

} // namespace FlexFlow
