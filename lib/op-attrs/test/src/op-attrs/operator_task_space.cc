#include "op-attrs/operator_task_space.h"
#include "test/utils/doctest/fmt/optional.h"
#include "utils/fmt/set.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_task_space_coordinates") {

    SUBCASE("OperatorTaskSpace has 0 dimensions") {
      OperatorTaskSpace task = OperatorTaskSpace{MinimalOrthotope{{}}};

      std::set<TaskSpaceCoordinate> correct = {
          TaskSpaceCoordinate{OrthotopeCoord{{}}}};
      std::set<TaskSpaceCoordinate> result = get_task_space_coordinates(task);
      CHECK(correct == result);
    }

    SUBCASE("OperatorTaskSpace has 2 dimensions") {

      OperatorTaskSpace task =
          OperatorTaskSpace{MinimalOrthotope{{2_ge2, 2_ge2}}};

      std::set<TaskSpaceCoordinate> correct = {{
          TaskSpaceCoordinate{OrthotopeCoord{{0_n, 0_n}}},
          TaskSpaceCoordinate{OrthotopeCoord{{0_n, 1_n}}},
          TaskSpaceCoordinate{OrthotopeCoord{{1_n, 0_n}}},
          TaskSpaceCoordinate{OrthotopeCoord{{1_n, 1_n}}},
      }};
      std::set<TaskSpaceCoordinate> result = get_task_space_coordinates(task);
      CHECK(correct == result);
    }

    SUBCASE("OperatorTaskSpace has 3 dimensions") {

      OperatorTaskSpace task =
          OperatorTaskSpace{MinimalOrthotope{{3_ge2, 2_ge2, 2_ge2}}};

      std::set<TaskSpaceCoordinate> correct = {{
          TaskSpaceCoordinate{OrthotopeCoord{{0_n, 0_n, 0_n}}},
          TaskSpaceCoordinate{OrthotopeCoord{{0_n, 0_n, 1_n}}},
          TaskSpaceCoordinate{OrthotopeCoord{{0_n, 1_n, 0_n}}},
          TaskSpaceCoordinate{OrthotopeCoord{{0_n, 1_n, 1_n}}},
          TaskSpaceCoordinate{OrthotopeCoord{{1_n, 0_n, 0_n}}},
          TaskSpaceCoordinate{OrthotopeCoord{{1_n, 0_n, 1_n}}},
          TaskSpaceCoordinate{OrthotopeCoord{{1_n, 1_n, 0_n}}},
          TaskSpaceCoordinate{OrthotopeCoord{{1_n, 1_n, 1_n}}},
          TaskSpaceCoordinate{OrthotopeCoord{{2_n, 0_n, 0_n}}},
          TaskSpaceCoordinate{OrthotopeCoord{{2_n, 0_n, 1_n}}},
          TaskSpaceCoordinate{OrthotopeCoord{{2_n, 1_n, 0_n}}},
          TaskSpaceCoordinate{OrthotopeCoord{{2_n, 1_n, 1_n}}},
      }};
      std::set<TaskSpaceCoordinate> result = get_task_space_coordinates(task);
      CHECK(correct == result);
    }
  }

  TEST_CASE("get_task_space_maximum_coordinate") {
    SUBCASE("OperatorTaskSpace has 2 dimensions") {

      OperatorTaskSpace task =
          OperatorTaskSpace{MinimalOrthotope{{3_ge2, 2_ge2}}};

      TaskSpaceCoordinate correct =
          TaskSpaceCoordinate{OrthotopeCoord{{2_n, 1_n}}};
      TaskSpaceCoordinate result = get_task_space_maximum_coordinate(task);
      CHECK(correct == result);
    }

    SUBCASE("OperatorTaskSpace has 3 dimensions") {

      OperatorTaskSpace task =
          OperatorTaskSpace{MinimalOrthotope{{3_ge2, 2_ge2, 4_ge2}}};

      TaskSpaceCoordinate correct =
          TaskSpaceCoordinate{OrthotopeCoord{{2_n, 1_n, 3_n}}};
      TaskSpaceCoordinate result = get_task_space_maximum_coordinate(task);
      CHECK(correct == result);
    }
  }

  TEST_CASE("smallest_operator_task_space_for_coord_set") {
    SUBCASE("coord set is empty") {
      std::set<TaskSpaceCoordinate> coord_set = {};

      std::optional<OperatorTaskSpace> result =
          smallest_operator_task_space_for_coord_set(coord_set);
      std::optional<OperatorTaskSpace> correct = trivial_op_task_space();

      CHECK(result == correct);
    }

    SUBCASE("coord set contains coords of different dimensionalities") {
      std::set<TaskSpaceCoordinate> coord_set = {
          TaskSpaceCoordinate{OrthotopeCoord{{1_n}}},
          TaskSpaceCoordinate{OrthotopeCoord{{2_n, 3_n}}},
      };

      CHECK_THROWS(smallest_operator_task_space_for_coord_set(coord_set));
    }

    SUBCASE("coord set contains only 0-d coords") {
      std::set<TaskSpaceCoordinate> coord_set = {
          TaskSpaceCoordinate{OrthotopeCoord{{}}},
      };

      std::optional<OperatorTaskSpace> result =
          smallest_operator_task_space_for_coord_set(coord_set);
      std::optional<OperatorTaskSpace> correct = trivial_op_task_space();

      CHECK(result == correct);
    }

    SUBCASE("coord set is not orthotopic") {
      std::set<TaskSpaceCoordinate> coord_set = {
          TaskSpaceCoordinate{OrthotopeCoord{{0_n, 1_n}}},
          TaskSpaceCoordinate{OrthotopeCoord{{2_n, 4_n}}},
          TaskSpaceCoordinate{OrthotopeCoord{{3_n, 0_n}}},
      };

      std::optional<OperatorTaskSpace> result =
          smallest_operator_task_space_for_coord_set(coord_set);
      std::optional<OperatorTaskSpace> correct = OperatorTaskSpace{
          MinimalOrthotope{
              /*dims=*/{4_ge2, 5_ge2},
          },
      };

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

      std::optional<OperatorTaskSpace> result =
          smallest_operator_task_space_for_coord_set(coord_set);
      std::optional<OperatorTaskSpace> correct = OperatorTaskSpace{
          MinimalOrthotope{
              /*dims=*/{3_ge2, 2_ge2},
          },
      };

      CHECK(result == correct);
    }

    SUBCASE("coord set's space is not minimal") {
      std::set<TaskSpaceCoordinate> coord_set = {
          TaskSpaceCoordinate{OrthotopeCoord{{0_n, 0_n}}},
          TaskSpaceCoordinate{OrthotopeCoord{{1_n, 0_n}}},
          TaskSpaceCoordinate{OrthotopeCoord{{2_n, 0_n}}},
      };

      CHECK_THROWS(smallest_operator_task_space_for_coord_set(coord_set));
    }
  }

  TEST_CASE("strict_operator_task_space_for_coord_set") {
    SUBCASE("coord set is empty") {
      std::set<TaskSpaceCoordinate> coord_set = {};

      std::optional<OperatorTaskSpace> result =
          strict_operator_task_space_for_coord_set(coord_set);
      std::optional<OperatorTaskSpace> correct = std::nullopt;

      CHECK(result == correct);
    }

    SUBCASE("coord set contains coords of different dimensionalities") {
      std::set<TaskSpaceCoordinate> coord_set = {
          TaskSpaceCoordinate{OrthotopeCoord{{1_n}}},
          TaskSpaceCoordinate{OrthotopeCoord{{2_n, 3_n}}},
      };

      CHECK_THROWS(strict_operator_task_space_for_coord_set(coord_set));
    }

    SUBCASE("coord set contains only 0-d coords") {
      std::set<TaskSpaceCoordinate> coord_set = {
          TaskSpaceCoordinate{OrthotopeCoord{{}}},
      };

      std::optional<OperatorTaskSpace> result =
          strict_operator_task_space_for_coord_set(coord_set);
      std::optional<OperatorTaskSpace> correct = trivial_op_task_space();

      CHECK(result == correct);
    }

    SUBCASE("coord set is not orthotopic") {
      std::set<TaskSpaceCoordinate> coord_set = {
          TaskSpaceCoordinate{OrthotopeCoord{{0_n, 1_n}}},
          TaskSpaceCoordinate{OrthotopeCoord{{2_n, 4_n}}},
          TaskSpaceCoordinate{OrthotopeCoord{{3_n, 0_n}}},
      };

      std::optional<OperatorTaskSpace> result =
          strict_operator_task_space_for_coord_set(coord_set);
      std::optional<OperatorTaskSpace> correct = std::nullopt;

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

      std::optional<OperatorTaskSpace> result =
          strict_operator_task_space_for_coord_set(coord_set);
      std::optional<OperatorTaskSpace> correct = OperatorTaskSpace{
          MinimalOrthotope{
              /*dims=*/{3_ge2, 2_ge2},
          },
      };

      CHECK(result == correct);
    }

    SUBCASE("coord set's space is not minimal") {
      std::set<TaskSpaceCoordinate> coord_set = {
          TaskSpaceCoordinate{OrthotopeCoord{{0_n, 0_n}}},
          TaskSpaceCoordinate{OrthotopeCoord{{1_n, 0_n}}},
          TaskSpaceCoordinate{OrthotopeCoord{{2_n, 0_n}}},
      };

      CHECK_THROWS(strict_operator_task_space_for_coord_set(coord_set));
    }
  }
}
