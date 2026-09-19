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

  TEST_CASE("orthotope_unflatten_bounded_component_2d") {
    SUBCASE("input bound == output tail bound") {
      positive_int output_tail_bound =  6_p;
      
      BoundedComponent input = BoundedComponent{
        /*component=*/2_n,
        /*output_tail_bound=*/output_tail_bound,
      };

      std::pair<BoundedComponent, BoundedComponent> result =       
        orthotope_unflatten_bounded_component_2d(input, output_tail_bound);

      std::pair<BoundedComponent, BoundedComponent> correct = std::pair{
        BoundedComponent{
          /*component=*/0_n,
          /*output_tail_bound=*/1_p,
        },
        BoundedComponent{
          /*component=*/2_n,
          /*output_tail_bound=*/output_tail_bound,
        },
      };

      CHECK(result == correct);
    }

    SUBCASE("input bound is divisible by output tail bound") {
      positive_int output_tail_bound =  6_p;
      
      SUBCASE("input component is bigger than output tail bound") {
        BoundedComponent input = BoundedComponent{
          /*component=*/17_n,
          /*output_tail_bound=*/18_p,
        };

        std::pair<BoundedComponent, BoundedComponent> result =       
          orthotope_unflatten_bounded_component_2d(input, output_tail_bound);

        std::pair<BoundedComponent, BoundedComponent> correct = std::pair{
          BoundedComponent{
            /*component=*/2_n,
            /*output_tail_bound=*/3_p,
          },
          BoundedComponent{
            /*component=*/5_n,
            /*output_tail_bound=*/6_p,
          },
        };

        CHECK(result == correct);
      }

      SUBCASE("input component is less than output tail bound") {
        BoundedComponent input = BoundedComponent{
          /*component=*/1_n,
          /*output_tail_bound=*/18_p,
        };

        std::pair<BoundedComponent, BoundedComponent> result =       
          orthotope_unflatten_bounded_component_2d(input, output_tail_bound);

        std::pair<BoundedComponent, BoundedComponent> correct = std::pair{
          BoundedComponent{
            /*component=*/0_n,
            /*output_tail_bound=*/3_p,
          },
          BoundedComponent{
            /*component=*/1_n,
            /*output_tail_bound=*/6_p,
          },
        };

        CHECK(result == correct);
      }
    }

    SUBCASE("output tail bound is divisible by input tail bound") {
      positive_int output_tail_bound = 12_p;

      BoundedComponent input = BoundedComponent{
        /*component=*/1_n,
        /*output_tail_bound=*/4_p,
      };

      std::pair<BoundedComponent, BoundedComponent> result =       
        orthotope_unflatten_bounded_component_2d(input, output_tail_bound);

      std::pair<BoundedComponent, BoundedComponent> correct = std::pair{
        BoundedComponent{
          /*component=*/0_n,
          /*output_tail_bound=*/1_p,
        },
        input,
      };

      CHECK(result == correct);
    }

    SUBCASE("input bound and output tail bound are mutually non-divisible") {
      positive_int output_tail_bound = 11_p;

      BoundedComponent input = BoundedComponent{
        /*component=*/1_n,
        /*output_tail_bound=*/4_p,
      };

      std::pair<BoundedComponent, BoundedComponent> result =       
        orthotope_unflatten_bounded_component_2d(input, output_tail_bound);

      std::pair<BoundedComponent, BoundedComponent> correct = std::pair{
        BoundedComponent{
          /*component=*/0_n,
          /*output_tail_bound=*/1_p,
        },
        input,
      };

      CHECK(result == correct);
    }
  }
}
