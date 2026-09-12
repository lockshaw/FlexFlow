#include "utils/orthotope/orthotope_bounded_coord.h"
#include "utils/containers/transform.h"
#include "utils/containers/concat_vectors.h"
#include "utils/containers/get_only.h"
#include "utils/containers/require_same.h"
#include "utils/nonnegative_int/num_elements.h"
#include "utils/containers/zip_with_strict.h"
#include "utils/containers/slice.h"
#include "utils/containers/get_last.h"

namespace FlexFlow {

nonnegative_int orthotope_bounded_coord_num_dims(OrthotopeBoundedCoord const &c) 
{
  return require_same(  
    num_elements(c.coord.raw),
    num_elements(c.bounds.dims));
}

std::vector<BoundedComponent> 
  components_of_orthotope_bounded_coord(OrthotopeBoundedCoord const &c)
{
  return zip_with_strict(
    c.coord.raw, 
    c.bounds.dims,
    [&](nonnegative_int component, positive_int bound) 
      -> BoundedComponent
    {
      return BoundedComponent{
        /*component=*/component,
        /*bound=*/bound,
      };
    });
}

OrthotopeBoundedCoord lift_bounded_component(BoundedComponent const &c)
{
  return OrthotopeBoundedCoord{
    /*coord=*/OrthotopeCoord{
      /*raw=*/{
        c.component,
      },
    },
    /*bounds=*/Orthotope{
      /*dims=*/{
        c.bound,
      },
    },
  };
}

OrthotopeBoundedCoord make_2d_orthotope_bounded_coord(
    BoundedComponent const &c1,
    BoundedComponent const &c2)
{
  return OrthotopeBoundedCoord{
    /*coord=*/OrthotopeCoord{
      /*raw=*/{
        c1.component,
        c2.component,
      },
    },
    /*bounds=*/Orthotope{
      /*dims=*/{
        c1.bound,
        c2.bound,
      },
    },
  };
}

OrthotopeBoundedCoord make_3d_orthotope_bounded_coord(
    BoundedComponent const &c1,
    BoundedComponent const &c2,
    BoundedComponent const &c3)
{
  return OrthotopeBoundedCoord{
    /*coord=*/OrthotopeCoord{
      /*raw=*/{
        c1.component,
        c2.component,
        c3.component,
      },
    },
    /*bounds=*/Orthotope{
      /*dims=*/{
        c1.bound,
        c2.bound,
        c3.bound,
      },
    },
  };
}

OrthotopeBoundedCoord make_orthotope_bounded_coord_from_components(
    std::vector<BoundedComponent> const &components)
{
  return OrthotopeBoundedCoord{
    /*coord=*/OrthotopeCoord{
      /*raw=*/
        transform(components,
                  [](BoundedComponent const &c) -> nonnegative_int {
                    return c.component;
                  }),
    },
    /*bounds=*/Orthotope{
      /*dims=*/
        transform(components,
                  [](BoundedComponent const &c) -> positive_int {
                    return c.bound;
                  }),
    },
  };
}

OrthotopeBoundedCoord orthotope_bounded_coord_product(
    OrthotopeBoundedCoord const &c1,
    OrthotopeBoundedCoord const &c2)
{
  return OrthotopeBoundedCoord{
    /*coord=*/OrthotopeCoord{
      concat_vectors(c1.coord.raw, c2.coord.raw),
    },
    /*bounds=*/Orthotope{
      concat_vectors(c1.bounds.dims, c2.bounds.dims),
    },
  };
}

OrthotopeBoundedCoord orthotope_bounded_coord_product(
    OrthotopeBoundedCoord const &c1,
    OrthotopeBoundedCoord const &c2,
    OrthotopeBoundedCoord const &c3)
{
  return OrthotopeBoundedCoord{
    /*coord=*/OrthotopeCoord{
      concat_vectors(std::vector{c1.coord.raw, c2.coord.raw, c3.coord.raw}),
    },
    /*bounds=*/Orthotope{
      concat_vectors(std::vector{c1.bounds.dims, c2.bounds.dims, c3.bounds.dims}),
    },
  };
}

OrthotopeBoundedCoord orthotope_bounded_coord_product(
    std::vector<OrthotopeBoundedCoord> const &cs)
{
  return OrthotopeBoundedCoord{
    /*coord=*/OrthotopeCoord{
      concat_vectors(
        transform(cs,
                  [](OrthotopeBoundedCoord const &c) -> std::vector<nonnegative_int> {
                    return c.coord.raw;
                  })),
    },
    /*bounds=*/Orthotope{
      concat_vectors(
        transform(cs,
                  [](OrthotopeBoundedCoord const &c) -> std::vector<positive_int> {
                    return c.bounds.dims;
                  })),
    },
  };
}

std::pair<BoundedComponent, OrthotopeBoundedCoord> orthotope_bounded_coord_slice_first(
    OrthotopeBoundedCoord const &coord) 
{
  std::vector<BoundedComponent> components = components_of_orthotope_bounded_coord(coord);

  BoundedComponent head = components.at(0);
  std::vector<BoundedComponent> tail = slice(components, 1, std::nullopt);

  return std::pair{
    head,
    make_orthotope_bounded_coord_from_components(tail),
  };
}

std::pair<OrthotopeBoundedCoord, BoundedComponent> orthotope_bounded_coord_slice_last(
    OrthotopeBoundedCoord const &coord) 
{
  std::vector<BoundedComponent> components = components_of_orthotope_bounded_coord(coord);

  BoundedComponent last = get_last(components);
  std::vector<BoundedComponent> leading = slice(components, 0, -1);

  return std::pair{
    make_orthotope_bounded_coord_from_components(leading),
    last,
  };
}

std::optional<BoundedComponent> flatten_orthotope_bounded_coord(OrthotopeBoundedCoord const &input) {
  nonnegative_int input_num_dims = orthotope_bounded_coord_num_dims(input);

  if (input_num_dims == 0) {
    return std::nullopt;
  }

  if (input_num_dims == 1) {
    return get_only(components_of_orthotope_bounded_coord(input));
  }

  std::pair<OrthotopeBoundedCoord, BoundedComponent> r = orthotope_bounded_coord_slice_last(input);
  OrthotopeBoundedCoord leading_components = r.first;
  BoundedComponent last_component = r.second;

  BoundedComponent flattened_leading =
    assert_unwrap(flatten_orthotope_bounded_coord(leading_components));

  return BoundedComponent{
    /*component=*/flattened_leading.component * last_component.bound + last_component.component,
    /*bound=*/flattened_leading.bound * last_component.bound,
  };
}

} // namespace FlexFlow
