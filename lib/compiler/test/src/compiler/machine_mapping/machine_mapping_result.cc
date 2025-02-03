#include "compiler/machine_mapping/machine_mapping_result.h"
#include "pcg/machine_view.h"
#include <catch2/catch_test_macros.hpp>

using namespace FlexFlow;


  TEST_CASE("series_combine") {
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

    float pre_cost = 2.0;
    MachineMappingResult pre = MachineMappingResult{
        FeasibleMachineMappingResult{
            /*runtime=*/pre_cost,
            /*machine_mapping=*/
            ParallelLayerGuidObliviousMachineMapping{{
                {
                    BinaryTreePath{{
                        BinaryTreePathEntry::LEFT_CHILD,
                    }},
                    machine_view_0,
                },
                {
                    BinaryTreePath{{
                        BinaryTreePathEntry::RIGHT_CHILD,
                    }},
                    machine_view_1,
                },
            }},
        },
    };

    float post_cost = 4.0;
    MachineMappingResult post = MachineMappingResult{
        FeasibleMachineMappingResult{
            /*runtime=*/post_cost,
            /*machine_mapping=*/
            ParallelLayerGuidObliviousMachineMapping{{
                {
                    BinaryTreePath{{}},
                    machine_view_1,
                },
            }},
        },
    };

    MachineMappingResult infeasible = infeasible_machine_mapping_result();

    float comm_cost = 3.0;

    SECTION("pre is infeasible") {
      MachineMappingResult result = series_combine(
          comm_cost, infeasible, post, ParallelSplitTransformation::LthenR);
      MachineMappingResult correct = infeasible;

      CHECK(result == correct);
    }

    SECTION("post is infeasible") {
      MachineMappingResult result = series_combine(
          comm_cost, pre, infeasible, ParallelSplitTransformation::LthenR);
      MachineMappingResult correct = infeasible;

      CHECK(result == correct);
    }

    SECTION("both are infeasible") {
      MachineMappingResult result =
          series_combine(comm_cost,
                         infeasible,
                         infeasible,
                         ParallelSplitTransformation::LthenR);
      MachineMappingResult correct = infeasible;

      CHECK(result == correct);
    }

    SECTION("both are feasible") {
      MachineMappingResult no_parallel_split_transform = MachineMappingResult{
          FeasibleMachineMappingResult{
              /*runtime=*/pre_cost + comm_cost + post_cost,
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
      };

      SECTION("parallel_split_transformation = std::nullopt") {
        MachineMappingResult result =
            series_combine(comm_cost, pre, post, std::nullopt);
        MachineMappingResult correct = no_parallel_split_transform;

        CHECK(result == correct);
      }

      SECTION("parallel_split_transformation = LthenR") {
        MachineMappingResult result = series_combine(
            comm_cost, pre, post, ParallelSplitTransformation::LthenR);
        MachineMappingResult correct = no_parallel_split_transform;

        CHECK(result == correct);
      }

      SECTION("parallel_split_transformation = RthenL") {
        MachineMappingResult result = series_combine(
            comm_cost, pre, post, ParallelSplitTransformation::RthenL);
        MachineMappingResult correct = MachineMappingResult{
            FeasibleMachineMappingResult{
                /*runtime=*/pre_cost + comm_cost + post_cost,
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
        };

        CHECK(result == correct);
      }
    }
  }

  TEST_CASE("parallel_combine") {
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

    MachineMappingResult lhs = MachineMappingResult{
        FeasibleMachineMappingResult{
            /*runtime=*/2.0,
            /*machine_mapping=*/
            ParallelLayerGuidObliviousMachineMapping{{
                {
                    BinaryTreePath{{
                        BinaryTreePathEntry::LEFT_CHILD,
                    }},
                    machine_view_0,
                },
                {
                    BinaryTreePath{{
                        BinaryTreePathEntry::RIGHT_CHILD,
                    }},
                    machine_view_1,
                },
            }},
        },
    };

    MachineMappingResult rhs = MachineMappingResult{
        FeasibleMachineMappingResult{
            /*runtime=*/4.0,
            /*machine_mapping=*/
            ParallelLayerGuidObliviousMachineMapping{{
                {
                    BinaryTreePath{{}},
                    machine_view_1,
                },
            }},
        },
    };

    MachineMappingResult infeasible = infeasible_machine_mapping_result();

    SECTION("lhs is infeasible") {
      MachineMappingResult result = parallel_combine(infeasible, rhs);
      MachineMappingResult correct = infeasible;

      CHECK(result == correct);
    }

    SECTION("rhs is infeasible") {
      MachineMappingResult result = parallel_combine(lhs, infeasible);
      MachineMappingResult correct = infeasible;

      CHECK(result == correct);
    }

    SECTION("both are infeasible") {
      MachineMappingResult result = parallel_combine(infeasible, infeasible);
      MachineMappingResult correct = infeasible;

      CHECK(result == correct);
    }

    SECTION("both are feasible") {
      MachineMappingResult result = parallel_combine(lhs, rhs);
      MachineMappingResult correct = MachineMappingResult{
          FeasibleMachineMappingResult{
              /*runtime=*/4.0,
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
      };

      CHECK(result == correct);
    }
  }

  TEST_CASE("minimize_runtime") {
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

    MachineMappingResult faster = MachineMappingResult{
        FeasibleMachineMappingResult{
            /*runtime=*/2.0,
            /*machine_mapping=*/
            ParallelLayerGuidObliviousMachineMapping{{
                {
                    BinaryTreePath{{
                        BinaryTreePathEntry::LEFT_CHILD,
                    }},
                    machine_view_0,
                },
                {
                    BinaryTreePath{{
                        BinaryTreePathEntry::RIGHT_CHILD,
                    }},
                    machine_view_1,
                },
            }},
        },
    };

    MachineMappingResult slower = MachineMappingResult{
        FeasibleMachineMappingResult{
            /*runtime=*/4.0,
            /*machine_mapping=*/
            ParallelLayerGuidObliviousMachineMapping{{
                {
                    BinaryTreePath{{}},
                    machine_view_1,
                },
            }},
        },
    };

    MachineMappingResult infeasible = infeasible_machine_mapping_result();

    SECTION("lhs is infeasible") {
      MachineMappingResult result = minimize_runtime(infeasible, slower);
      MachineMappingResult correct = slower;

      CHECK(result == correct);
    }

    SECTION("rhs is infeasible") {
      MachineMappingResult result = minimize_runtime(slower, infeasible);
      MachineMappingResult correct = slower;

      CHECK(result == correct);
    }

    SECTION("both are infeasible") {
      MachineMappingResult result = minimize_runtime(infeasible, infeasible);
      MachineMappingResult correct = infeasible;

      CHECK(result == correct);
    }

    SECTION("both are feasible") {
      SECTION("lhs is faster") {
        MachineMappingResult result = minimize_runtime(faster, slower);
        MachineMappingResult correct = faster;

        CHECK(result == correct);
      }

      SECTION("rhs is faster") {
        MachineMappingResult result = minimize_runtime(slower, faster);
        MachineMappingResult correct = faster;

        CHECK(result == correct);
      }

      SECTION("lhs and rhs have the same speed") {
        MachineMappingResult result = minimize_runtime(slower, slower);
        MachineMappingResult correct = slower;

        CHECK(result == correct);
      }
    }
  }
