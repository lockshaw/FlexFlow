#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_ORTHOTOPE_BOUNDED_COORD_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_ORTHOTOPE_BOUNDED_COORD_H

namespace FlexFlow {

OrthotopeBoundedCoord orthotope_bounded_coord_product(OrthotopeBoundedCoord const &, OrthotopeBoundedCoord const &);

BoundedIndex flatten_orthotope_bounded_coord(OrthotopeBoundedCoord const &);

} // namespace FlexFlow

#endif
