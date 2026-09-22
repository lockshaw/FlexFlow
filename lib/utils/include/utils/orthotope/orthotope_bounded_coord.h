#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_ORTHOTOPE_BOUNDED_COORD_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_ORTHOTOPE_BOUNDED_COORD_H

#include "utils/orthotope/orthotope_bounded_coord.dtg.h"
#include "utils/orthotope/bounded_component.dtg.h"
#include <optional>

namespace FlexFlow {

nonnegative_int orthotope_bounded_coord_num_dims(OrthotopeBoundedCoord const &);

std::vector<BoundedComponent> 
  components_of_orthotope_bounded_coord(OrthotopeBoundedCoord const &);

OrthotopeBoundedCoord lift_bounded_component(BoundedComponent const &);

OrthotopeBoundedCoord make_2d_orthotope_bounded_coord(
    BoundedComponent const &,
    BoundedComponent const &);

OrthotopeBoundedCoord make_3d_orthotope_bounded_coord(
    BoundedComponent const &,
    BoundedComponent const &,
    BoundedComponent const &);

OrthotopeBoundedCoord make_4d_orthotope_bounded_coord(
    BoundedComponent const &,
    BoundedComponent const &,
    BoundedComponent const &,
    BoundedComponent const &);

OrthotopeBoundedCoord make_orthotope_bounded_coord_from_head_and_tail(
    BoundedComponent const &head,
    OrthotopeBoundedCoord const &tail);

OrthotopeBoundedCoord make_orthotope_bounded_coord_from_components(
    std::vector<BoundedComponent> const &);

OrthotopeBoundedCoord orthotope_bounded_coord_product(OrthotopeBoundedCoord const &, OrthotopeBoundedCoord const &);

OrthotopeBoundedCoord orthotope_bounded_coord_product(
    OrthotopeBoundedCoord const &,
    OrthotopeBoundedCoord const &,
    OrthotopeBoundedCoord const &);

OrthotopeBoundedCoord orthotope_bounded_coord_product(
    std::vector<OrthotopeBoundedCoord> const &);

std::pair<BoundedComponent, OrthotopeBoundedCoord> orthotope_bounded_coord_slice_first(
    OrthotopeBoundedCoord const &);

std::pair<OrthotopeBoundedCoord, BoundedComponent> orthotope_bounded_coord_slice_last(
    OrthotopeBoundedCoord const &);

std::optional<BoundedComponent> flatten_orthotope_bounded_coord(OrthotopeBoundedCoord const &);

std::pair<BoundedComponent, BoundedComponent> orthotope_unflatten_bounded_component_2d(
    BoundedComponent const &input,
    positive_int output_tail_bound);

OrthotopeBoundedCoord orthotope_unflatten_bounded_component(
    BoundedComponent const &component,
    Orthotope const &output_tail_orthotope);

Orthotope orthotope_find_left_weighted_cor(
    positive_int dimension_size,
    Orthotope const &ground_domain);

OrthotopeBoundedCoord orthotope_opportunistically_unflatten_bounded_component_for_ground_domain(
    BoundedComponent const &component,
    Orthotope const &ground_domain);

OrthotopeCoord project_bounded_coordinate_to_orthotope(OrthotopeBoundedCoord const &input_coord,
                                                       Orthotope const &output);

} // namespace FlexFlow

#endif
