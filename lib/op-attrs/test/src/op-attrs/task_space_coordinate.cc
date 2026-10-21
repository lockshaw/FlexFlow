#include "op-attrs/task_space_coordinate.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("task_space_coord_set_is_orthotopic") {
    SUBCASE("coord set is empty") {
      std::set<TaskSpaceCoordinate> coord_set = {};

      bool result = task_space_coord_set_is_orthotopic(coord_set);
      bool correct = false;

      CHECK(result == correct);
    }

    SUBCASE("coord set contains coords of different dimensionalities") {
      std::set<TaskSpaceCoordinate> coord_set = {
          TaskSpaceCoordinate{OrthotopeCoord{{1_n}}},
          TaskSpaceCoordinate{OrthotopeCoord{{2_n, 3_n}}},
      };

      CHECK_THROWS(task_space_coord_set_is_orthotopic(coord_set));
    }

    SUBCASE("coord set contains only 0-d coords") {
      std::set<TaskSpaceCoordinate> coord_set = {
          TaskSpaceCoordinate{OrthotopeCoord{{}}},
      };

      bool result = task_space_coord_set_is_orthotopic(coord_set);
      bool correct = true;

      CHECK(result == correct);
    }

    SUBCASE("coord set is not orthotopic") {
      std::set<TaskSpaceCoordinate> coord_set = {
          TaskSpaceCoordinate{OrthotopeCoord{{0_n, 1_n}}},
          TaskSpaceCoordinate{OrthotopeCoord{{2_n, 4_n}}},
          TaskSpaceCoordinate{OrthotopeCoord{{3_n, 0_n}}},
      };

      bool result = task_space_coord_set_is_orthotopic(coord_set);
      bool correct = false;

      CHECK(result == correct);
    }

    SUBCASE("coord set is orthotopic") {
      std::set<TaskSpaceCoordinate> coord_set = {
          TaskSpaceCoordinate{OrthotopeCoord{{0_n, 0_n}}},
          TaskSpaceCoordinate{OrthotopeCoord{{1_n, 0_n}}},
          TaskSpaceCoordinate{OrthotopeCoord{{2_n, 0_n}}},
          TaskSpaceCoordinate{OrthotopeCoord{{0_n, 1_n}}},
          TaskSpaceCoordinate{OrthotopeCoord{{1_n, 1_n}}},
          TaskSpaceCoordinate{OrthotopeCoord{{2_n, 1_n}}},
      };

      bool result = task_space_coord_set_is_orthotopic(coord_set);
      bool correct = true;

      CHECK(result == correct);
    }

    SUBCASE("coord set's space is not minimal") {
      std::set<TaskSpaceCoordinate> coord_set = {
          TaskSpaceCoordinate{OrthotopeCoord{{0_n, 0_n}}},
          TaskSpaceCoordinate{OrthotopeCoord{{1_n, 0_n}}},
          TaskSpaceCoordinate{OrthotopeCoord{{2_n, 0_n}}},
      };

      bool result = task_space_coord_set_is_orthotopic(coord_set);
      bool correct = true;

      CHECK(result == correct);
    }
  }
}
