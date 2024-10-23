#include "utils/orthotope/orthotope_bijective_projection.h"
#include "utils/containers/zip.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("project_into_1d") {
    SUBCASE("to 1d from 1d is identity") {
      OrthotopeCoordinate coord = OrthotopeCoordinate{{2}};
      Orthotope orthotope = Orthotope{{5}};

      int result = project_into_1d(orthotope, coord);
      int correct = 2;

      CHECK(result == correct);
    }

    SUBCASE("basic example") {
      OrthotopeCoordinate coord = OrthotopeCoordinate{{4, 1}};
      Orthotope orthotope = Orthotope{{5, 3}};

      int result = project_into_1d(orthotope, coord);
      int correct = 4 * 3 + 1;

      CHECK(result == correct);
    }

    SUBCASE("order matters") {
      OrthotopeCoordinate coord = OrthotopeCoordinate{{1, 4}};
      Orthotope orthotope = Orthotope{{3, 5}};

      int result = project_into_1d(orthotope, coord);
      int correct = 1 * 5 + 4;

      CHECK(result == correct);
    }

    SUBCASE("throws if coord is outside of orthotope") {
      OrthotopeCoordinate coord = OrthotopeCoordinate{
        {2, 3, 1},
      };

      Orthotope orthotope = Orthotope{
        {5, 3, 2},
      };

      CHECK_THROWS(project_into_1d(orthotope, coord));
    }

    SUBCASE("throws if coord does not have same dimension as orthotope") {
      OrthotopeCoordinate coord = OrthotopeCoordinate{
        {2, 3, 1},
      };

      Orthotope orthotope = Orthotope{
        {5, 3},
      };

      CHECK_THROWS(project_into_1d(orthotope, coord));
    }

    SUBCASE("returns 0 if orthotope is 0-dimensional") {
      OrthotopeCoordinate coord = OrthotopeCoordinate{{}};
      Orthotope orthotope = Orthotope{{}};

      int result = project_into_1d(orthotope, coord);
      int correct = 0;

      CHECK(result == correct);
    }
  }

  TEST_CASE("project_out_of_1d") {
    SUBCASE("from 1d to 1d is identity") {
      Orthotope orthotope = Orthotope{{5}};
      int coord = 2;

      OrthotopeCoordinate result = project_out_of_1d(coord, orthotope);
      OrthotopeCoordinate correct = OrthotopeCoordinate{{2}};

      CHECK(result == correct);
    }

    SUBCASE("basic example") {
      Orthotope orthotope = Orthotope{{5, 3}};
      int coord = 4 * 3 + 1;

      OrthotopeCoordinate result = project_out_of_1d(coord, orthotope);
      OrthotopeCoordinate correct = OrthotopeCoordinate{{4, 1}};

      CHECK(result == correct);
    }

    SUBCASE("orthotope dimension order matters") {
      Orthotope orthotope = Orthotope{{3, 5}};
      int coord = 1 * 5 + 4;

      OrthotopeCoordinate result = project_out_of_1d(coord, orthotope);
      OrthotopeCoordinate correct = OrthotopeCoordinate{{1, 4}};

      CHECK(result == correct);
    }

    SUBCASE("throws if coord would be projected outside of orthotope") {
      Orthotope orthotope = Orthotope{{5, 3}};

      SUBCASE("smallest coord outside of orthotope") {
        int coord = 15;

        CHECK_THROWS(project_out_of_1d(coord, orthotope));
      }

      SUBCASE("largest coord inside of orthotope") {
        int coord = 14;

        OrthotopeCoordinate result = project_out_of_1d(coord, orthotope);
        OrthotopeCoordinate correct = OrthotopeCoordinate{{4, 2}};

        CHECK(result == correct);
      }
    }

    SUBCASE("if dst orthotope is 0-dimensional") {
      Orthotope orthotope = Orthotope{{}};

      SUBCASE("returns 0-d coord if input coord is 0") {
        int input_coord = 0;

        OrthotopeCoordinate result = project_out_of_1d(input_coord, orthotope);
        OrthotopeCoordinate correct = OrthotopeCoordinate{{}};

        CHECK(result == correct);
      }

      SUBCASE("throws if input coord is anything other than zero") {
        int input_coord = 1;

        CHECK_THROWS(project_out_of_1d(input_coord, orthotope));
      }
    }
  }

  TEST_CASE("project_coordinate_through") {
    Orthotope src = Orthotope{
      {2, 3},
    };

    Orthotope dst = Orthotope{
      {6},
    };

    OrthotopeBijectiveProjection proj = OrthotopeBijectiveProjection{
      {orthotope_dim_idx_t{0}, orthotope_dim_idx_t{0}},
      /*reversed=*/false,
    };

    OrthotopeCoordinate src_coord = OrthotopeCoordinate{
      {1, 2},
    };
    OrthotopeCoordinate dst_coord = OrthotopeCoordinate{
      {1*3+2},
    };

    SUBCASE("forward") {
      OrthotopeCoordinate result = project_coordinate_through(proj, src, src_coord, dst);
      OrthotopeCoordinate correct = dst_coord;

      CHECK(result == correct);
    }

    SUBCASE("backward") {
      OrthotopeBijectiveProjection reversed = reverse_projection(proj);

      OrthotopeCoordinate result = project_coordinate_through(reversed, dst, dst_coord, src);
      OrthotopeCoordinate correct = src_coord;

      CHECK(result == correct);
    }
  }
}
