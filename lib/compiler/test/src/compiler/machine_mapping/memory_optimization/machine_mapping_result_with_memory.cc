#include "compiler/machine_mapping/memory_optimization/machine_mapping_with_memory_result.h"
#include "pcg/machine_view.h"
#include "utils/nonnegative_int/nonnegative_int.h"
#include <catch2/catch_test_macros.hpp>

using namespace FlexFlow;


  TEST_CASE("remove_non_pareto_optimal_machine_mapping_result") {
    MachineView machine_view_0 = MachineView{
        /*start=*/MachineSpaceCoordinate{
            /*node_idx=*/0_n,
            /*device_idx=*/0_n,
            /*device_type=*/DeviceType::GPU,
        },
        /*dimensions=*/
        {
            MachineViewDimension{
                stride_t{1_n},
                MachineSpecificationDimension::INTRA_NODE,
            },
        },
    };

    MachineView machine_view_1 = MachineView{
        /*start=*/MachineSpaceCoordinate{
            /*node_idx=*/0_n,
            /*device_idx=*/0_n,
            /*device_type=*/DeviceType::GPU,
        },
        /*dimensions=*/
        {
            MachineViewDimension{
                stride_t{2_n},
                MachineSpecificationDimension::INTRA_NODE,
            },
        },
    };

    MachineView machine_view_2 = MachineView{
        /*start=*/MachineSpaceCoordinate{
            /*node_idx=*/0_n,
            /*device_idx=*/0_n,
            /*device_type=*/DeviceType::GPU,
        },
        /*dimensions=*/
        {
            MachineViewDimension{
                stride_t{4_n},
                MachineSpecificationDimension::INTRA_NODE,
            },
        },
    };

    OpCostMetrics cost1 = OpCostMetrics{
        /*forward_runtime=*/2.0,
        /*backward_runtime=*/2.0,
        /*memory=*/2_n,
    };

    OpCostMetrics cost2 = OpCostMetrics{
        /*forward_runtime=*/4.0,
        /*backward_runtime=*/4.0,
        /*memory=*/1_n,
    };

    OpCostMetrics cost3 = OpCostMetrics{
        /*forward_runtime=*/2.0,
        /*backward_runtime=*/2.0,
        /*memory=*/3_n,
    };

    MachineMappingForSingleLayer mm1 = MachineMappingForSingleLayer{
        cost1,
        ParallelLayerGuidObliviousMachineMapping{
            {
                {
                    BinaryTreePath{{}},
                    machine_view_0,
                },
            },
        },
    };

    MachineMappingForSingleLayer mm2 = MachineMappingForSingleLayer{
        cost2,
        ParallelLayerGuidObliviousMachineMapping{
            {
                {
                    BinaryTreePath{{}},
                    machine_view_1,
                },
            },
        },
    };

    MachineMappingForSingleLayer mm3 = MachineMappingForSingleLayer{
        cost3,
        ParallelLayerGuidObliviousMachineMapping{
            {
                {
                    BinaryTreePath{{}},
                    machine_view_2,
                },
            },
        },
    };

    SECTION("empty") {
      MachineMappingWithMemoryResult before_remove =
          empty_machine_mapping_with_memory_result();
      MachineMappingWithMemoryResult result =
          remove_non_pareto_optimal_machine_mapping_result(before_remove);
      MachineMappingWithMemoryResult correct =
          empty_machine_mapping_with_memory_result();

      CHECK(result == correct);
    }

    SECTION("all solutions are pareto-optimal") {
      MachineMappingWithMemoryResult before_remove =
          MachineMappingWithMemoryResult{
              {
                  mm1,
                  mm2,
              },
          };
      MachineMappingWithMemoryResult result =
          remove_non_pareto_optimal_machine_mapping_result(before_remove);
      MachineMappingWithMemoryResult correct = before_remove;

      CHECK(result == correct);
    }

    SECTION("there exists a non-pareto-optimal solution") {
      MachineMappingWithMemoryResult before_remove =
          MachineMappingWithMemoryResult{
              {
                  mm1,
                  mm2,
                  mm3,
              },
          };
      MachineMappingWithMemoryResult result =
          remove_non_pareto_optimal_machine_mapping_result(before_remove);
      MachineMappingWithMemoryResult correct = MachineMappingWithMemoryResult{
          {
              mm1,
              mm2,
          },
      };

      CHECK(result == correct);
    }
  }

  TEST_CASE("series_combine(float, MachineMappingWithMemoryResult const &, "
            "MachineMappingWithMemoryResult const &, "
            "std::optional<ParallelSplitTransformation> const&)") {
    MachineView machine_view_0 = MachineView{
        /*start=*/MachineSpaceCoordinate{
            /*node_idx=*/0_n,
            /*device_idx=*/0_n,
            /*device_type=*/DeviceType::GPU,
        },
        /*dimensions=*/
        {
            MachineViewDimension{
                stride_t{1_n},
                MachineSpecificationDimension::INTRA_NODE,
            },
        },
    };

    MachineView machine_view_1 = MachineView{
        /*start=*/MachineSpaceCoordinate{
            /*node_idx=*/0_n,
            /*device_idx=*/0_n,
            /*device_type=*/DeviceType::GPU,
        },
        /*dimensions=*/
        {
            MachineViewDimension{
                stride_t{2_n},
                MachineSpecificationDimension::INTRA_NODE,
            },
        },
    };

    OpCostMetrics pre_cost = OpCostMetrics{
        /*forward_runtime=*/2.0,
        /*backward_runtime=*/2.0,
        /*memory=*/2_n,
    };
    MachineMappingWithMemoryResult pre = MachineMappingWithMemoryResult{{
        MachineMappingForSingleLayer{
            pre_cost,
            ParallelLayerGuidObliviousMachineMapping{
                {
                    {
                        BinaryTreePath{
                            {BinaryTreePathEntry::LEFT_CHILD},
                        },
                        machine_view_0,
                    },
                    {
                        BinaryTreePath{
                            {BinaryTreePathEntry::RIGHT_CHILD},
                        },
                        machine_view_1,
                    },
                },
            },
        },
    }};

    OpCostMetrics post_cost = OpCostMetrics{
        /*forward_runtime=*/4.0,
        /*backward_runtime=*/4.0,
        /*memory=*/1_n,
    };

    MachineMappingWithMemoryResult post = MachineMappingWithMemoryResult{{
        MachineMappingForSingleLayer{
            post_cost,
            ParallelLayerGuidObliviousMachineMapping{
                {
                    {
                        BinaryTreePath{{}},
                        machine_view_1,
                    },
                },
            },
        },
    }};

    MachineMappingWithMemoryResult empty =
        empty_machine_mapping_with_memory_result();

    float comm_cost = 3.0;

    SECTION("pre is empty") {
      MachineMappingWithMemoryResult result = series_combine(
          comm_cost, empty, post, ParallelSplitTransformation::LthenR);
      MachineMappingWithMemoryResult correct = empty;

      CHECK(result == correct);
    }

    SECTION("post is empty") {
      MachineMappingWithMemoryResult result = series_combine(
          comm_cost, pre, empty, ParallelSplitTransformation::LthenR);
      MachineMappingWithMemoryResult correct = empty;

      CHECK(result == correct);
    }

    SECTION("both are nonempty") {
      MachineMappingWithMemoryResult no_parallel_split_transform =
          MachineMappingWithMemoryResult{
              {
                  MachineMappingForSingleLayer{
                      /*cost=*/OpCostMetrics{
                          /*forward_runtime=*/pre_cost.forward_runtime +
                              comm_cost + post_cost.forward_runtime,
                          /*backward_runtime=*/pre_cost.backward_runtime +
                              comm_cost + post_cost.backward_runtime,
                          /*memory=*/pre_cost.memory + post_cost.memory,
                      },
                      /*machine_mapping=*/
                      ParallelLayerGuidObliviousMachineMapping{{
                          {
                              BinaryTreePath{{
                                  BinaryTreePathEntry::LEFT_CHILD,
                                  BinaryTreePathEntry::LEFT_CHILD,
                              }},
                              machine_view_0,
                          },
                          {
                              BinaryTreePath{{
                                  BinaryTreePathEntry::LEFT_CHILD,
                                  BinaryTreePathEntry::RIGHT_CHILD,
                              }},
                              machine_view_1,
                          },
                          {
                              BinaryTreePath{{
                                  BinaryTreePathEntry::RIGHT_CHILD,
                              }},
                              machine_view_1,
                          },
                      }},
                  },
              },
          };

      SECTION("parallel_split_transformation = std::nullopt") {
        MachineMappingWithMemoryResult result =
            series_combine(comm_cost, pre, post, std::nullopt);
        MachineMappingWithMemoryResult correct = no_parallel_split_transform;

        CHECK(result == correct);
      }

      SECTION("parallel_split_transformation = LthenR") {
        MachineMappingWithMemoryResult result = series_combine(
            comm_cost, pre, post, ParallelSplitTransformation::LthenR);
        MachineMappingWithMemoryResult correct = no_parallel_split_transform;

        CHECK(result == correct);
      }

      SECTION("parallel_split_transformation = RthenL") {
        MachineMappingWithMemoryResult result = series_combine(
            comm_cost, pre, post, ParallelSplitTransformation::RthenL);
        MachineMappingWithMemoryResult correct = MachineMappingWithMemoryResult{
            {
                MachineMappingForSingleLayer{
                    /*cost=*/OpCostMetrics{
                        /*forward_runtime=*/pre_cost.forward_runtime +
                            comm_cost + post_cost.forward_runtime,
                        /*backward_runtime=*/pre_cost.backward_runtime +
                            comm_cost + post_cost.backward_runtime,
                        /*memory=*/pre_cost.memory + post_cost.memory,
                    },
                    /*machine_mapping=*/
                    ParallelLayerGuidObliviousMachineMapping{{
                        {
                            BinaryTreePath{{
                                BinaryTreePathEntry::RIGHT_CHILD,
                                BinaryTreePathEntry::LEFT_CHILD,
                            }},
                            machine_view_0,
                        },
                        {
                            BinaryTreePath{{
                                BinaryTreePathEntry::RIGHT_CHILD,
                                BinaryTreePathEntry::RIGHT_CHILD,
                            }},
                            machine_view_1,
                        },
                        {
                            BinaryTreePath{{
                                BinaryTreePathEntry::LEFT_CHILD,
                            }},
                            machine_view_1,
                        },
                    }},
                },
            },
        };

        CHECK(result == correct);
      }
    }
  }

  TEST_CASE("parallel_combine(float, MachineMappingWithMemoryResult const &, "
            "MachineMappingWithMemoryResult const &, "
            "std::optional<ParallelSplitTransformation> const&)") {
    MachineView machine_view_0 = MachineView{
        /*start=*/MachineSpaceCoordinate{
            /*node_idx=*/0_n,
            /*device_idx=*/0_n,
            /*device_type=*/DeviceType::GPU,
        },
        /*dimensions=*/
        {
            MachineViewDimension{
                stride_t{1_n},
                MachineSpecificationDimension::INTRA_NODE,
            },
        },
    };

    MachineView machine_view_1 = MachineView{
        /*start=*/MachineSpaceCoordinate{
            /*node_idx=*/0_n,
            /*device_idx=*/0_n,
            /*device_type=*/DeviceType::GPU,
        },
        /*dimensions=*/
        {
            MachineViewDimension{
                stride_t{2_n},
                MachineSpecificationDimension::INTRA_NODE,
            },
        },
    };

    OpCostMetrics lhs_cost = OpCostMetrics{
        /*forward_runtime=*/2.0,
        /*backward_runtime=*/2.0,
        /*memory=*/2_n,
    };
    MachineMappingWithMemoryResult lhs = MachineMappingWithMemoryResult{{
        MachineMappingForSingleLayer{
            lhs_cost,
            ParallelLayerGuidObliviousMachineMapping{
                {
                    {
                        BinaryTreePath{
                            {BinaryTreePathEntry::LEFT_CHILD},
                        },
                        machine_view_0,
                    },
                    {
                        BinaryTreePath{
                            {BinaryTreePathEntry::RIGHT_CHILD},
                        },
                        machine_view_1,
                    },
                },
            },
        },
    }};

    OpCostMetrics rhs_cost = OpCostMetrics{
        /*forward_runtime=*/4.0,
        /*backward_runtime=*/4.0,
        /*memory=*/1_n,
    };
    MachineMappingWithMemoryResult rhs = MachineMappingWithMemoryResult{{
        MachineMappingForSingleLayer{
            rhs_cost,
            ParallelLayerGuidObliviousMachineMapping{
                {
                    {
                        BinaryTreePath{{}},
                        machine_view_1,
                    },
                },
            },
        },
    }};

    MachineMappingWithMemoryResult empty =
        empty_machine_mapping_with_memory_result();

    SECTION("lhs is empty") {
      MachineMappingWithMemoryResult result = parallel_combine(empty, rhs);
      MachineMappingWithMemoryResult correct = empty;

      CHECK(result == correct);
    }

    SECTION("rhs is empty") {
      MachineMappingWithMemoryResult result = parallel_combine(lhs, empty);
      MachineMappingWithMemoryResult correct = empty;

      CHECK(result == correct);
    }

    SECTION("both are nonempty") {
      MachineMappingWithMemoryResult result = parallel_combine(lhs, rhs);
      MachineMappingWithMemoryResult correct = MachineMappingWithMemoryResult{{
          MachineMappingForSingleLayer{
              /*cost=*/OpCostMetrics{
                  /*forward_runtime=*/std::max(lhs_cost.forward_runtime,
                                               rhs_cost.forward_runtime),
                  /*backward_runtime=*/
                  std::max(lhs_cost.backward_runtime,
                           rhs_cost.backward_runtime),
                  /*memory=*/std::max(lhs_cost.memory, rhs_cost.memory),
              },
              /*machine_mapping=*/
              ParallelLayerGuidObliviousMachineMapping{
                  {
                      {
                          BinaryTreePath{{BinaryTreePathEntry::LEFT_CHILD,
                                          BinaryTreePathEntry::LEFT_CHILD}},
                          machine_view_0,
                      },
                      {
                          BinaryTreePath{{BinaryTreePathEntry::LEFT_CHILD,
                                          BinaryTreePathEntry::RIGHT_CHILD}},
                          machine_view_1,
                      },
                      {
                          BinaryTreePath{{BinaryTreePathEntry::RIGHT_CHILD}},
                          machine_view_1,
                      },
                  },
              },
          },
      }};

      CHECK(result == correct);
    }
  }

  TEST_CASE("minimize_runtime(memory)") {
    MachineView machine_view_0 = MachineView{
        /*start=*/MachineSpaceCoordinate{
            /*node_idx=*/0_n,
            /*device_idx=*/0_n,
            /*device_type=*/DeviceType::GPU,
        },
        /*dimensions=*/
        {
            MachineViewDimension{
                stride_t{1_n},
                MachineSpecificationDimension::INTRA_NODE,
            },
        },
    };

    MachineView machine_view_1 = MachineView{
        /*start=*/MachineSpaceCoordinate{
            /*node_idx=*/0_n,
            /*device_idx=*/0_n,
            /*device_type=*/DeviceType::GPU,
        },
        /*dimensions=*/
        {
            MachineViewDimension{
                stride_t{2_n},
                MachineSpecificationDimension::INTRA_NODE,
            },
        },
    };

    MachineView machine_view_2 = MachineView{
        /*start=*/MachineSpaceCoordinate{
            /*node_idx=*/0_n,
            /*device_idx=*/0_n,
            /*device_type=*/DeviceType::GPU,
        },
        /*dimensions=*/
        {
            MachineViewDimension{
                stride_t{4_n},
                MachineSpecificationDimension::INTRA_NODE,
            },
        },
    };

    OpCostMetrics cost1 = OpCostMetrics{
        /*forward_runtime=*/2.0,
        /*backward_runtime=*/2.0,
        /*memory=*/2_n,
    };
    OpCostMetrics cost2 = OpCostMetrics{
        /*forward_runtime=*/4.0,
        /*backward_runtime=*/4.0,
        /*memory=*/1_n,
    };
    OpCostMetrics cost3 = OpCostMetrics{
        /*forward_runtime=*/2.0,
        /*backward_runtime=*/2.0,
        /*memory=*/3_n,
    };

    MachineMappingForSingleLayer mm1 = MachineMappingForSingleLayer{
        cost1,
        ParallelLayerGuidObliviousMachineMapping{
            {
                {
                    BinaryTreePath{{}},
                    machine_view_0,
                },
            },
        },
    };

    MachineMappingForSingleLayer mm2 = MachineMappingForSingleLayer{
        cost2,
        ParallelLayerGuidObliviousMachineMapping{
            {
                {
                    BinaryTreePath{{}},
                    machine_view_1,
                },
            },
        },
    };

    MachineMappingForSingleLayer mm3 = MachineMappingForSingleLayer{
        cost3,
        ParallelLayerGuidObliviousMachineMapping{
            {
                {
                    BinaryTreePath{{}},
                    machine_view_2,
                },
            },
        },
    };

    MachineMappingWithMemoryResult result1 = MachineMappingWithMemoryResult{
        {
            mm1,
            mm2,
        },
    };

    MachineMappingWithMemoryResult result2 = MachineMappingWithMemoryResult{
        {
            mm2,
            mm3,
        },
    };

    MachineMappingWithMemoryResult result = minimize_runtime(result1, result2);
    MachineMappingWithMemoryResult correct = MachineMappingWithMemoryResult{
        {
            mm1,
            mm2,
        },
    };

    CHECK(result == correct);
  }
