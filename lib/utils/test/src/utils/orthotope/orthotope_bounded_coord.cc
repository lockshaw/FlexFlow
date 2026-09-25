#include <doctest/doctest.h>
#include "utils/orthotope/orthotope_bounded_coord.h"
#include "utils/not_implemented.h"
#include "test/utils/doctest/fmt/optional.h"
#include "test/utils/doctest/fmt/pair.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("orthotope_bounded_coord_slice_first") {
    SUBCASE("input num dims is 0") {
      OrthotopeBoundedCoord input = OrthotopeBoundedCoord{
        /*coord=*/OrthotopeCoord{{}},
        /*bounds=*/Orthotope{{}},
      };

      CHECK_THROWS(orthotope_bounded_coord_slice_first(input));
    }

    SUBCASE("input num dims is 1") {
      OrthotopeBoundedCoord input = OrthotopeBoundedCoord{
        /*coord=*/OrthotopeCoord{
          {2_n},
        },
        /*bounds=*/Orthotope{
          {4_p},
        },
      };

      std::pair<BoundedComponent, OrthotopeBoundedCoord> result = orthotope_bounded_coord_slice_first(input);
      std::pair<BoundedComponent, OrthotopeBoundedCoord> correct = {
        BoundedComponent{2_n, 4_p},
        OrthotopeBoundedCoord{
          /*coord=*/OrthotopeCoord{{}},
          /*bounds=*/Orthotope{{}},
        },
      };

      CHECK(result == correct);
    }

    SUBCASE("input num dims is > 1") {
      OrthotopeBoundedCoord input = OrthotopeBoundedCoord{
        /*coord=*/OrthotopeCoord{
          {2_n, 1_n, 3_n},
        },
        /*bounds=*/Orthotope{
          {4_p, 3_p, 5_p},
        },
      };

      std::pair<BoundedComponent, OrthotopeBoundedCoord> result = orthotope_bounded_coord_slice_first(input);
      std::pair<BoundedComponent, OrthotopeBoundedCoord> correct = {
        BoundedComponent{2_n, 4_p},
        OrthotopeBoundedCoord{
          /*coord=*/OrthotopeCoord{{1_n, 3_n}},
          /*bounds=*/Orthotope{{3_p, 5_p}},
        },
      };

      CHECK(result == correct);
    }
  }

  TEST_CASE("orthotope_bounded_coord_slice_last") {
    SUBCASE("input num dims is 0") {
      OrthotopeBoundedCoord input = OrthotopeBoundedCoord{
        /*coord=*/OrthotopeCoord{{}},
        /*bounds=*/Orthotope{{}},
      };

      CHECK_THROWS(orthotope_bounded_coord_slice_last(input));
    }

    SUBCASE("input num dims is 1") {
      OrthotopeBoundedCoord input = OrthotopeBoundedCoord{
        /*coord=*/OrthotopeCoord{
          {2_n},
        },
        /*bounds=*/Orthotope{
          {4_p},
        },
      };

      std::pair<OrthotopeBoundedCoord, BoundedComponent> result = orthotope_bounded_coord_slice_last(input);
      std::pair<OrthotopeBoundedCoord, BoundedComponent> correct = {
        OrthotopeBoundedCoord{
          /*coord=*/OrthotopeCoord{{}},
          /*bounds=*/Orthotope{{}},
        },
        BoundedComponent{2_n, 4_p},
      };

      CHECK(result == correct);
    }

    SUBCASE("input num dims is > 1") {
      OrthotopeBoundedCoord input = OrthotopeBoundedCoord{
        /*coord=*/OrthotopeCoord{
          {2_n, 1_n, 3_n},
        },
        /*bounds=*/Orthotope{
          {4_p, 3_p, 5_p},
        },
      };

      std::pair<OrthotopeBoundedCoord, BoundedComponent> result = orthotope_bounded_coord_slice_last(input);
      std::pair<OrthotopeBoundedCoord, BoundedComponent> correct = {
        OrthotopeBoundedCoord{
          /*coord=*/OrthotopeCoord{{2_n, 1_n}},
          /*bounds=*/Orthotope{{4_p, 3_p}},
        },
        BoundedComponent{3_n, 5_p},
      };

      CHECK(result == correct);
    }
  }

  TEST_CASE("flatten_orthotope_bounded_coord") {
    SUBCASE("input num dims is 0") {
      // TODO(@lockshaw)(#pr):
      OrthotopeBoundedCoord input = OrthotopeBoundedCoord{
        /*coord=*/OrthotopeCoord{{}},
        /*bounds=*/Orthotope{{}},
      };

      std::optional<BoundedComponent> result = flatten_orthotope_bounded_coord(input);
      std::optional<BoundedComponent> correct = std::nullopt;

      CHECK(result == correct);
    }

    SUBCASE("input num dims is 1") {
      OrthotopeBoundedCoord input = OrthotopeBoundedCoord{
        /*coord=*/OrthotopeCoord{
          {2_n},
        },
        /*bounds=*/Orthotope{
          {4_p},
        },
      };

      std::optional<BoundedComponent> result = flatten_orthotope_bounded_coord(input);
      std::optional<BoundedComponent> correct = BoundedComponent{
        /*component=*/2_n,
        /*bound=*/4_p,
      };

      CHECK(result == correct);
    }

    SUBCASE("input num dims is > 1") {
      OrthotopeBoundedCoord input = OrthotopeBoundedCoord{
        /*coord=*/OrthotopeCoord{
          {2_n, 1_n, 3_n},
        },
        /*bounds=*/Orthotope{
          {4_p, 3_p, 5_p},
        },
      };

      std::optional<BoundedComponent> result = flatten_orthotope_bounded_coord(input);
      std::optional<BoundedComponent> correct = BoundedComponent{
        /*component=*/2_n * 3_p * 5_p + 1_n * 5_p + 3_n,
        /*bound=*/4_p * 3_p * 5_p,
      };

      CHECK(result == correct);
    }
  }
}
