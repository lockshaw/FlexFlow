#include "compiler/machine_mapping/memory_optimization/get_optimal_machine_mapping_with_memory.h"
#include "../../cost_estimator_for_test.h"
#include "compiler/machine_mapping/abstracted_tensor_set_movement/abstracted_tensor_set_movement.h"
#include "compiler/machine_mapping/machine_mapping_constraints.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/machine_mapping_problem_tree.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/unmapped_op_cost_estimate_key.h"
#include "compiler/machine_mapping/memory_optimization/machine_mapping_with_memory_cache.h"
#include "pcg/machine_view.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_builder.h"
#include "utils/containers/get_only.h"
#include "utils/full_binary_tree/binary_tree_path.h"
#include "utils/nonnegative_int/nonnegative_int.h"
#include <catch2/catch_test_macros.hpp>

using namespace FlexFlow;


  TEST_CASE("get_optimal_machine_mapping_with_memory") {
    auto make_leaf = [](UnmappedOpCostEstimateKey const &k) {
      return MachineMappingProblemTree{k};
    };

    auto make_series_split =
        [](AbstractedTensorSetMovement const &tensor_set_movement,
           MachineMappingProblemTree const &lhs,
           MachineMappingProblemTree const &rhs) {
          return MachineMappingProblemTree{
              MMProblemTreeSeriesSplit{
                  /*tensor_set_movement=*/tensor_set_movement,
                  /*left_child=*/lhs,
                  /*right_child=*/rhs,
              },
          };
        };

    auto make_parallel_split = [](MachineMappingProblemTree const &lhs,
                                  MachineMappingProblemTree const &rhs) {
      return MachineMappingProblemTree{
          MMProblemTreeParallelSplit{
              /*left_child=*/lhs,
              /*right_child=*/rhs,
          },
      };
    };

    MachineView mv1 = MachineView{
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

    MachineView mv2 = MachineView{
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

    MachineSpecification full_machine_spec = MachineSpecification{
        /*num_nodes=*/2_n,
        /*num_cpus_per_node=*/1_n,
        /*num_gpus_per_node=*/1_n,
        /*inter_node_bandwidth=*/1,
        /*intra_node_bandwidth=*/1,
    };

    MachineSpecification split_machine_spec = MachineSpecification{
        /*num_nodes=*/1_n,
        /*num_cpus_per_node=*/1_n,
        /*num_gpus_per_node=*/1_n,
        /*inter_node_bandwidth=*/1,
        /*intra_node_bandwidth=*/1,
    };

    auto allowed_machine_views1 = [&](UnmappedOpCostEstimateKey const &,
                                      MachineSpecification const &resources) {
      if (resources == full_machine_spec) {
        return std::unordered_set<MachineView>{mv1, mv2};
      } else {
        return std::unordered_set<MachineView>{mv2};
      }
    };

    UnmappedOpCostEstimateKey k1 = UnmappedOpCostEstimateKey{
        /*op_attrs=*/PCGOperatorAttrs{InputAttrs{}},
        /*input_shapes=*/{},
        /*weight_shapes=*/{},
        /*output_shapes=*/{},
    };

    UnmappedOpCostEstimateKey k2 = UnmappedOpCostEstimateKey{
        /*op_attrs=*/PCGOperatorAttrs{ElementBinaryAttrs{
            /*type=*/OperatorType::EW_ADD,
            /*compute_type=*/DataType::FLOAT,
            /*should_broadcast_lhs=*/false,
            /*should_broadcast_rhs=*/false,
        }},
        /*input_shapes=*/{},
        /*weight_shapes=*/{},
        /*output_shapes=*/{},
    };

    ParallelTensorShape tensor_shape1 = ParallelTensorShape{
        ParallelTensorDims{
            FFOrdered<ShardParallelDim>{},
            ReplicaParallelDimSet{
                SumDegree{1_n},
                DiscardCopyDegree{1_n},
            },
        },
        DataType::FLOAT,
    };

    AbstractedTensorSetMovement movement1 = AbstractedTensorSetMovement{{
        AbstractedSingleTensorMovement{
            /*parallel_tensor_shape=*/tensor_shape1,
            /*src_machine_views=*/{},
            /*dst_machine_views=*/{},
        },
    }};

    ParallelLayerGuidObliviousMachineMapping mm1 =
        ParallelLayerGuidObliviousMachineMapping{{
            {binary_tree_root_path(), mv1},
        }};
    ParallelLayerGuidObliviousMachineMapping mm2 =
        ParallelLayerGuidObliviousMachineMapping{{
            {binary_tree_root_path(), mv2},
        }};

    CostEstimator cost_estimator = make_fake_cost_estimator(
        std::unordered_map<OpCostEstimateKey, OpCostMetrics>{{
            {map_unmapped_op_cost_estimate_key(k1, mv1),
             OpCostMetrics{/*forward_runtime=*/1.0,
                           /*backward_runtime=*/1.0,
                           /*memory=*/nonnegative_int{2}}},
            {map_unmapped_op_cost_estimate_key(k2, mv1),
             OpCostMetrics{/*forward_runtime=*/2.0,
                           /*backward_runtime=*/2.0,
                           /*memory=*/nonnegative_int{3}}},
            {map_unmapped_op_cost_estimate_key(k1, mv2),
             OpCostMetrics{/*forward_runtime=*/1.5,
                           /*backward_runtime=*/1.5,
                           /*memory=*/nonnegative_int{1}}},
            {map_unmapped_op_cost_estimate_key(k2, mv2),
             OpCostMetrics{/*forward_runtime=*/2.5,
                           /*backward_runtime=*/2.5,
                           /*memory=*/nonnegative_int{2}}},
        }},
        std::unordered_map<TensorSetMovement, float>{{
            {TensorSetMovement{/*movements=*/{}}, /*cost=*/0.0},
            {concretize_abstracted_tensor_set_movement(movement1, mm1, mm1),
             /*cost=*/0.1},
            {concretize_abstracted_tensor_set_movement(movement1, mm2, mm2),
             /*cost=*/0.2},
            {concretize_abstracted_tensor_set_movement(movement1, mm1, mm2),
             /*cost=*/0.3},
            {concretize_abstracted_tensor_set_movement(movement1, mm2, mm1),
             /*cost=*/0.4},
        }});

    MachineMappingContext context = MachineMappingContext{
        cost_estimator,
        allowed_machine_views1,
    };

    MachineMappingWithMemoryCache cache =
        empty_machine_mapping_with_memory_cache();

    SECTION("single layer") {
      MachineMappingProblemTree problem_tree = make_leaf(k1);

      MachineMappingConstraints constraints =
          get_unconstrained_solution_for_layers(
              get_all_leaf_paths(problem_tree));

      MachineMappingWithMemoryResult result =
          get_optimal_machine_mapping_with_memory(
              cache, context, problem_tree, full_machine_spec, constraints);
      MachineMappingWithMemoryResult correct = MachineMappingWithMemoryResult{{
          MachineMappingForSingleLayer{
              OpCostMetrics{/*forward_runtime=*/1.0,
                            /*backward_runtime=*/1.0,
                            /*memory=*/nonnegative_int{2}},
              ParallelLayerGuidObliviousMachineMapping{{
                  {binary_tree_root_path(), mv1},
              }},
          },
          MachineMappingForSingleLayer{
              OpCostMetrics{/*forward_runtime=*/1.5,
                            /*backward_runtime=*/1.5,
                            /*memory=*/nonnegative_int{1}},
              ParallelLayerGuidObliviousMachineMapping{{
                  {binary_tree_root_path(), mv2},
              }},
          },
      }};

      CHECK(result == correct);
    }

    SECTION("pair of layers in sequence") {
      MachineMappingProblemTree problem_tree =
          make_series_split(movement1, make_leaf(k1), make_leaf(k2));

      MachineMappingConstraints constraints =
          get_unconstrained_solution_for_layers(
              get_all_leaf_paths(problem_tree));

      MachineMappingWithMemoryResult result =
          get_optimal_machine_mapping_with_memory(
              cache, context, problem_tree, full_machine_spec, constraints);
      MachineMappingWithMemoryResult correct = MachineMappingWithMemoryResult{{
          MachineMappingForSingleLayer{
              OpCostMetrics{
                  /*forward_runtime=*/1.0 + 2.0 + 0.1,
                  /*backward_runtime=*/1.0 + 2.0 + 0.1,
                  /*memory=*/nonnegative_int{2 + 3},
              },
              ParallelLayerGuidObliviousMachineMapping{{
                  {
                      BinaryTreePath{{
                          BinaryTreePathEntry::LEFT_CHILD,
                      }},
                      mv1,
                  },
                  {
                      BinaryTreePath{{
                          BinaryTreePathEntry::RIGHT_CHILD,
                      }},
                      mv1,
                  },
              }},
          },
          MachineMappingForSingleLayer{
              OpCostMetrics{/*forward_runtime=*/1.5 + 2.5 + 0.1,
                            /*backward_runtime=*/1.5 + 2.5 + 0.1,
                            /*memory=*/nonnegative_int{1 + 2}},
              ParallelLayerGuidObliviousMachineMapping{{
                  {
                      BinaryTreePath{{
                          BinaryTreePathEntry::LEFT_CHILD,
                      }},
                      mv2,
                  },
                  {
                      BinaryTreePath{{
                          BinaryTreePathEntry::RIGHT_CHILD,
                      }},
                      mv2,
                  },
              }},
          },
      }};

      CHECK(result == correct);
    }

    SECTION("pair of layers in parallel") {
      MachineMappingProblemTree problem_tree =
          make_parallel_split(make_leaf(k1), make_leaf(k2));

      MachineMappingConstraints constraints =
          get_unconstrained_solution_for_layers(
              get_all_leaf_paths(problem_tree));

      MachineMappingWithMemoryResult result =
          get_optimal_machine_mapping_with_memory(
              cache, context, problem_tree, full_machine_spec, constraints);
      MachineMappingWithMemoryResult correct =
          MachineMappingWithMemoryResult{{MachineMappingForSingleLayer{
              OpCostMetrics{/*forward_runtime=*/2.5,
                            /*backward_runtime=*/2.5,
                            /*memory=*/nonnegative_int{2}},
              ParallelLayerGuidObliviousMachineMapping{{
                  {
                      BinaryTreePath{{
                          BinaryTreePathEntry::LEFT_CHILD,
                      }},
                      mv2,
                  },
                  {
                      BinaryTreePath{{
                          BinaryTreePathEntry::RIGHT_CHILD,
                      }},
                      mv2,
                  },
              }},

          }}};

      CHECK(result == correct);
    }
  }
