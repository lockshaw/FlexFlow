#include <doctest/doctest.h>
#include "compiler/machine_mapping/machine_view_1d_projection.h"
#include "op-attrs/task_space_coordinate.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("projection_1d_get_flat_machine_space_offset") {
    OperatorTaskSpace task_space = OperatorTaskSpace{
        MinimalOrthotope{{
            2_ge2,
            2_ge2,
        }},
    };

    SUBCASE("2D onto 1D") {
      MachineView1dProjection projection = MachineView1dProjection{
        {
          stride_t{1_p},
          stride_t{2_p},
        },
      };

      SUBCASE("coordinate (0, 0)") {
        TaskSpaceCoordinate coord = make_task_space_coordinate({0_n, 0_n});

        FlatMachineSpaceOffset result = projection_1d_get_flat_machine_space_offset(task_space, projection, coord);

        FlatMachineSpaceOffset correct = FlatMachineSpaceOffset{0};

        CHECK(result == correct);
      }

      SUBCASE("coordinate (0, 1)") {
        TaskSpaceCoordinate coord = make_task_space_coordinate({0_n, 1_n});

        FlatMachineSpaceOffset result = projection_1d_get_flat_machine_space_offset(task_space, projection, coord);

        FlatMachineSpaceOffset correct = FlatMachineSpaceOffset{2};

        CHECK(result == correct);
      }

      SUBCASE("coordinate (1, 0)") {
        TaskSpaceCoordinate coord = make_task_space_coordinate({1_n, 0_n});

        FlatMachineSpaceOffset result = projection_1d_get_flat_machine_space_offset(task_space, projection, coord);

        FlatMachineSpaceOffset correct = FlatMachineSpaceOffset{4};

        CHECK(result == correct);
      }

      SUBCASE("coordinate (1, 1)") {
        TaskSpaceCoordinate coord = make_task_space_coordinate({1_n, 1_n});

        FlatMachineSpaceOffset result = projection_1d_get_flat_machine_space_offset(task_space, projection, coord);

        FlatMachineSpaceOffset correct = FlatMachineSpaceOffset{6};

        CHECK(result == correct);
      }
    }

    SUBCASE("reversed_stries") {
      MachineView1dProjection projection = MachineView1dProjection{
        {
          stride_t{2_p},
          stride_t{1_p},
        },
      };

      SUBCASE("coordinate (0, 0)") {
        TaskSpaceCoordinate coord = make_task_space_coordinate({0_n, 0_n});

        FlatMachineSpaceOffset result = projection_1d_get_flat_machine_space_offset(task_space, projection, coord);

        FlatMachineSpaceOffset correct = FlatMachineSpaceOffset{0};

        CHECK(result == correct);
      }

      SUBCASE("coordinate (0, 1)") {
        TaskSpaceCoordinate coord = make_task_space_coordinate({0_n, 1_n});

        FlatMachineSpaceOffset result = projection_1d_get_flat_machine_space_offset(task_space, projection, coord);

        FlatMachineSpaceOffset correct = FlatMachineSpaceOffset{1};

        CHECK(result == correct);
      }

      SUBCASE("coordinate (1, 0)") {
        TaskSpaceCoordinate coord = make_task_space_coordinate({1_n, 0_n});

        FlatMachineSpaceOffset result = projection_1d_get_flat_machine_space_offset(task_space, projection, coord);

        FlatMachineSpaceOffset correct = FlatMachineSpaceOffset{4};

        CHECK(result == correct);
      }

      SUBCASE("coordinate (1, 1)") {
        TaskSpaceCoordinate coord = make_task_space_coordinate({1_n, 1_n});

        FlatMachineSpaceOffset result = projection_1d_get_flat_machine_space_offset(task_space, projection, coord);

        FlatMachineSpaceOffset correct = FlatMachineSpaceOffset{5};

        CHECK(result == correct);
      }
    }
  }
}
