#include "compiler/machine_mapping/start_invariant_machine_view.h"
#include "op-attrs/task_space_coordinate.h"
#include "pcg/machine_compute_resource_slice.h"
#include "utils/fmt/unordered_set.h"
#include "utils/fmt/vector.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("StartInvariantMachineView - utility functions") {
    StartInvariantMachineView simv = StartInvariantMachineView{
        MachineView2dProjection{
            {MachineViewDimension{stride_t{2_p},
                                  MachineSpecificationDimension::INTER_NODE},
             MachineViewDimension{stride_t{2_p},
                                  MachineSpecificationDimension::INTER_NODE}},
        },
    };

    SUBCASE("get_expected_task_space_num_dims") {
      nonnegative_int result = get_expected_task_space_num_dims(simv);
      nonnegative_int correct = 2_n;
      CHECK(result == correct);
    }

    SUBCASE("start_invariant_mv_get_strides") {
      std::vector<stride_t> result = start_invariant_mv_get_strides(simv);
      std::vector<stride_t> correct = {stride_t{2_p}, stride_t{2_p}};
      CHECK(result == correct);
    }
  }

  TEST_CASE("StartInvariantMachineView - conversions") {
    MachineSpaceCoordinate start = MachineSpaceCoordinate{1_n, 2_n};

    std::vector<MachineViewDimension> dimensions = {
        MachineViewDimension{stride_t{2_p},
                             MachineSpecificationDimension::INTER_NODE},
        MachineViewDimension{stride_t{3_p},
                             MachineSpecificationDimension::INTRA_NODE}};

    StartInvariantMachineView simv =
        StartInvariantMachineView{MachineView2dProjection{dimensions}};

    MachineView mv = MachineView{
        start,
        simv,
    };

    SUBCASE("start_invariant_from_machine_view") {
      StartInvariantMachineView result = start_invariant_from_machine_view(mv);

      StartInvariantMachineView correct = simv;

      CHECK(result == correct);
    }

    SUBCASE("machine_view_from_start_invariant") {
      MachineView result = machine_view_from_start_invariant(simv, start);

      MachineView correct = mv;

      CHECK(result == correct);
    }

    SUBCASE("conversion is invertible") {
      SUBCASE("MachineView -> StartInvariant -> MachineView") {
        MachineView result = machine_view_from_start_invariant(
            start_invariant_from_machine_view(mv), start);

        MachineView correct = mv;

        CHECK(result == correct);
      }

      SUBCASE("StartInvariant -> MachineView -> StartInvariant") {
        StartInvariantMachineView result = start_invariant_from_machine_view(
            machine_view_from_start_invariant(simv, start));

        StartInvariantMachineView correct = simv;

        CHECK(result == correct);
      }
    }
  }

  TEST_CASE("StartInvariantMachineView - get_machine_space_offset") {
    SUBCASE("1D case") {
      // This operator has shape (3,), and thus 3 tasks.
      // The (only) dimension is projected on the INTRA (device) dimension with
      // a stride of 2. The machine space has 1 node and 6 devices per node.
      /**
       * The tasks will thus be distributed like this:
       *  +-------+-------+-------+-------+-------+-------+
       *  | (0,)  |       | (1,)  |       | (2,)  |       |
       *  +-------+-------+-------+-------+-------+-------+
       */
      OperatorTaskSpace task = OperatorTaskSpace{
          MinimalOrthotope{{
              3_ge2,
          }},
      };

      MachineComputeResourceSlice machine_space = MachineComputeResourceSlice{
          /*num_nodes=*/1_p,
          /*num_gpus_per_node=*/6_p,
      };

      StartInvariantMachineView simv = StartInvariantMachineView{
          MachineView2dProjection{
              /*dimensions=*/{
                  MachineViewDimension{
                      stride_t{2_p},
                      MachineSpecificationDimension::INTRA_NODE,
                  },
              },
          },
      };

      MachineComputeSpecification ms = MachineComputeSpecification{
          /*num_nodes=*/1_p,
          /*num_cpus_per_node=*/6_p,
          /*num_gpus_per_node=*/6_p,
      };

      SUBCASE("get_machine_space_offset") {
        SUBCASE("Task with TaskSpaceCoordinate = (0,)") {
          TaskSpaceCoordinate coord = make_task_space_coordinate({0_n});

          UnresolvedMachineSpaceOffset result =
              get_machine_space_offset(task, simv, coord);

          UnresolvedMachineSpaceOffset correct = UnresolvedMachineSpaceOffset{
              MachineSpaceOffset{0, 0},
          };

          CHECK(correct == result);
        }

        SUBCASE("Task with TaskSpaceCoordinate = (1,)") {
          TaskSpaceCoordinate coord = make_task_space_coordinate({1_n});

          UnresolvedMachineSpaceOffset result =
              get_machine_space_offset(task, simv, coord);

          UnresolvedMachineSpaceOffset correct = UnresolvedMachineSpaceOffset{
              MachineSpaceOffset{0, 2},
          };

          CHECK(correct == result);
        }

        SUBCASE("Task with TaskSpaceCoordinate = (2,)") {
          TaskSpaceCoordinate coord = make_task_space_coordinate({2_n});

          UnresolvedMachineSpaceOffset result =
              get_machine_space_offset(task, simv, coord);

          UnresolvedMachineSpaceOffset correct = UnresolvedMachineSpaceOffset{
              MachineSpaceOffset{0, 4},
          };

          CHECK(correct == result);
        }
      }

      SUBCASE("get_machine_space_offsets") {
        std::unordered_set<UnresolvedMachineSpaceOffset> result =
            get_machine_space_offsets(task, simv);

        std::unordered_set<UnresolvedMachineSpaceOffset> correct = {
            UnresolvedMachineSpaceOffset{MachineSpaceOffset{0, 0}},
            UnresolvedMachineSpaceOffset{MachineSpaceOffset{0, 2}},
            UnresolvedMachineSpaceOffset{MachineSpaceOffset{0, 4}},
        };

        CHECK(correct == result);
      }
    }

    SUBCASE("2D case") {
      // This operator has shape (2, 2), and thus 2 * 2 = 4 tasks.
      // The first dimension is projected onto the INTER (node) dimension with
      // stride 1, while the second dimension is projected onto the INTRA
      // (device) dimension with stride 2. The machine space has 2 nodes and 4
      // devices per node.

      /**
       * The tasks will thus be distributed like this:
       *  +-------+-------+-------+-------+
       *  | (0,0) |       | (0,1) |       |
       *  +-------+-------+-------+-------+
       *  | (1,0) |       | (1,1) |       |
       *  +-------+-------+-------+-------+
       */

      OperatorTaskSpace task = OperatorTaskSpace{
          MinimalOrthotope{{
              2_ge2,
              2_ge2,
          }},
      };

      StartInvariantMachineView simv = StartInvariantMachineView{
          MachineView2dProjection{
              {MachineViewDimension{
                   stride_t{1_p},
                   MachineSpecificationDimension::INTER_NODE,
               },
               MachineViewDimension{
                   stride_t{2_p},
                   MachineSpecificationDimension::INTRA_NODE,
               }},
          },
      };

      MachineComputeSpecification ms = MachineComputeSpecification{
          /*num_nodes=*/2_p,
          /*num_cpus_per_node=*/4_p,
          /*num_gpus_per_node=*/4_p,
      };

      MachineComputeResourceSlice machine_space =
          compute_slice_from_specification(ms);

      SUBCASE("get_machine_space_offset") {
        SUBCASE("Task with TaskSpaceCoordinate = (0,0)") {
          TaskSpaceCoordinate coord = make_task_space_coordinate({0_n, 0_n});

          UnresolvedMachineSpaceOffset result =
              get_machine_space_offset(task, simv, coord);

          UnresolvedMachineSpaceOffset correct = UnresolvedMachineSpaceOffset{
              MachineSpaceOffset{0, 0},
          };

          CHECK(correct == result);
        }

        SUBCASE("Task with TaskSpaceCoordinate = (0,1)") {
          TaskSpaceCoordinate coord = make_task_space_coordinate({0_n, 1_n});

          UnresolvedMachineSpaceOffset result =
              get_machine_space_offset(task, simv, coord);

          UnresolvedMachineSpaceOffset correct = UnresolvedMachineSpaceOffset{
              MachineSpaceOffset{0, 2},
          };

          CHECK(correct == result);
        }

        SUBCASE("Task with TaskSpaceCoordinate = (1,0)") {
          TaskSpaceCoordinate coord = make_task_space_coordinate({1_n, 0_n});

          UnresolvedMachineSpaceOffset result =
              get_machine_space_offset(task, simv, coord);

          UnresolvedMachineSpaceOffset correct = UnresolvedMachineSpaceOffset{
              MachineSpaceOffset{1, 0},
          };

          CHECK(correct == result);
        }

        SUBCASE("Task with TaskSpaceCoordinate = (1,1)") {
          TaskSpaceCoordinate coord = make_task_space_coordinate({1_n, 1_n});

          UnresolvedMachineSpaceOffset result =
              get_machine_space_offset(task, simv, coord);

          UnresolvedMachineSpaceOffset correct = UnresolvedMachineSpaceOffset{
              MachineSpaceOffset{1, 2},
          };

          CHECK(correct == result);
        }
      }

      SUBCASE("get_machine_space_offsets") {
        std::unordered_set<UnresolvedMachineSpaceOffset> result =
            get_machine_space_offsets(task, simv);

        std::unordered_set<UnresolvedMachineSpaceOffset> correct = {
            UnresolvedMachineSpaceOffset{MachineSpaceOffset{0, 0}},
            UnresolvedMachineSpaceOffset{MachineSpaceOffset{0, 2}},
            UnresolvedMachineSpaceOffset{MachineSpaceOffset{1, 0}},
            UnresolvedMachineSpaceOffset{MachineSpaceOffset{1, 2}},
        };

        CHECK(correct == result);
      }
    }
  }
}
