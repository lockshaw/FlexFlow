#include "utils/orthotope/orthotope.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("orthotope_contains_coord") {
    Orthotope orthotope = Orthotope{
      {3, 1},
    };

    SUBCASE("returns true if coord is in orthotope bounds") {
      SUBCASE("smallest allowed coord") {
        OrthotopeCoordinate coord = OrthotopeCoordinate{
          {0, 0},
        };

        bool result = orthotope_contains_coord(orthotope, coord);
        bool correct = true;

        CHECK(result == correct);
      }

      SUBCASE("largest allowed coord") {
        OrthotopeCoordinate coord = OrthotopeCoordinate{
          {2, 0},
        };

        bool result = orthotope_contains_coord(orthotope, coord);
        bool correct = true;

        CHECK(result == correct);
      }
    }

    SUBCASE("returns false if coord is out of orthotope bounds") {
      SUBCASE("too low") {
        // exhaustively check all dims because we can
        SUBCASE("dim 0") {
          OrthotopeCoordinate coord = OrthotopeCoordinate{
            {-1, 0},
          };

          bool result = orthotope_contains_coord(orthotope, coord);
          bool correct = false;

          CHECK(result == correct);
        }

        SUBCASE("dim 1") {
          OrthotopeCoordinate coord = OrthotopeCoordinate{
            {1, -1},
          };

          bool result = orthotope_contains_coord(orthotope, coord);
          bool correct = false;

          CHECK(result == correct);
        }
      }

      SUBCASE("too high") {
        // exhaustively check all dims because we can
        SUBCASE("dim 0") {
          OrthotopeCoordinate coord = OrthotopeCoordinate{
            {3, 0},
          };

          bool result = orthotope_contains_coord(orthotope, coord);
          bool correct = false;

          CHECK(result == correct);
        }

        SUBCASE("dim 1") {
          OrthotopeCoordinate coord = OrthotopeCoordinate{
            {1, 1},
          };

          bool result = orthotope_contains_coord(orthotope, coord);
          bool correct = false;

          CHECK(result == correct);
        }
      }
    }

    SUBCASE("throws if num dims of coord does not match num dims of the orthotope") {
      OrthotopeCoordinate coord = OrthotopeCoordinate{
        {0, 0, 0},
      };

      CHECK_THROWS(orthotope_contains_coord(orthotope, coord));
    }
  }
}
