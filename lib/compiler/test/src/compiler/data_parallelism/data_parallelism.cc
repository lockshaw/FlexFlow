#include <doctest/doctest.h>
#include "compiler/data_parallelism/data_parallelism.h"
#include "pcg/computation_graph_builder.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_builder.h"
#include "compiler/search_result.h"
#include "pcg/mapped_parallel_computation_graph/mapped_parallel_computation_graph.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("apply_data_parallelism") {
    TensorShape input_shape = TensorShape{
      TensorDims{
        FFOrdered<positive_int>{
          9_p,
          3_p,
          6_p,
        },
      },
      DataType::FLOAT,
    };

    ComputationGraph cg = [&] {
      ComputationGraphBuilder b;

      tensor_guid_t t1 = b.create_input(input_shape);
      tensor_guid_t t2 = b.create_input(input_shape);

      tensor_guid_t t3 = b.add(t1, t2);
      tensor_guid_t t4 = b.dense(
        /*input=*/t1,
        /*outDim=*/6_p,
        /*activation=*/std::nullopt,
        /*use_bias=*/false,
        /*data_type=*/DataType::FLOAT,
        /*projection_initializer=*/std::nullopt,
        /*bias_initializer=*/std::nullopt);

      return b.computation_graph;
    }();

    int_ge_two degree = 3_ge2;

    MachineComputeSpecification machine_spec = MachineComputeSpecification{
      /*num_nodes=*/4_p,
      /*num_cpus_per_node=*/1_p,
      /*num_gpus_per_node=*/2_p,
    };

    MappedParallelComputationGraph result = get_mapped_pcg_from_search_result(apply_data_parallelism(cg, degree), machine_spec);

    std::string add_name = "add";
    std::string dense_name = "dense";

    ParallelComputationGraph correct_pcg = [&] {
      ParallelComputationGraphBuilder b;

      parallel_tensor_guid_t t1 = b.create_input_tensor(input_shape);
      parallel_tensor_guid_t t2 = b.create_input_tensor(input_shape);
      t1 = b.parallel_partition(t1, ff_dim_t{0_n}, degree.positive_int_from_int_ge_two());
      t2 = b.parallel_partition(t2, ff_dim_t{0_n}, degree.positive_int_from_int_ge_two());


      parallel_tensor_guid_t t3 = b.add(t1, t2, add_name);
      parallel_tensor_guid_t t4 = b.dense(
        /*input=*/t1,
        /*outDim=*/6_p,
        /*activation=*/std::nullopt,
        /*use_bias=*/false,
        /*data_type=*/DataType::FLOAT,
        /*projection_initializer=*/std::nullopt,
        /*bias_initializer=*/std::nullopt,
        /*name=*/dense_name);

      return b.pcg;
    }();

    parallel_layer_guid_t l_add = get_parallel_layer_by_name(correct_pcg, add_name);
    parallel_layer_guid_t l_dense = get_parallel_layer_by_name(correct_pcg, dense_name);

      auto ptensor_coord = [](nonnegative_int discard_copy_component,
                              nonnegative_int batch_component) -> ParallelTensorSpaceCoordinate {
        return ParallelTensorSpaceCoordinate{
          /*sum_component=*/0_n,
          /*discard_copy_component=*/discard_copy_component,
          /*shard_components=*/FFOrdered{batch_component, 0_n, 0_n},
        };
      };


    std::unordered_map<parallel_layer_guid_t, MappedOperatorTaskGroup> correct_mapping = {
      {
        l_add,
        MappedOperatorTaskGroup{
          bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding>{
            {
              MachineSpaceCoordinate{0_n, 0_n},
              OperatorAtomicTaskShardBinding{
                {
                  {TensorSlotName::LHS_INPUT, ptensor_coord(0_n, 0_n)},
                  {TensorSlotName::RHS_INPUT, ptensor_coord(0_n, 0_n)},
                },
              },
            },
            {
              MachineSpaceCoordinate{0_n, 1_n},
              OperatorAtomicTaskShardBinding{
                {
                  {TensorSlotName::LHS_INPUT, ptensor_coord(0_n, 1_n)},
                  {TensorSlotName::RHS_INPUT, ptensor_coord(0_n, 1_n)},
                },
              },
            },
            {
              MachineSpaceCoordinate{1_n, 0_n},
              OperatorAtomicTaskShardBinding{
                {
                  {TensorSlotName::LHS_INPUT, ptensor_coord(0_n, 2_n)},
                  {TensorSlotName::RHS_INPUT, ptensor_coord(0_n, 2_n)},
                },
              },
            },
          },
        },
      },
      {
        l_dense,
        MappedOperatorTaskGroup{
          bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding>{
            {
              MachineSpaceCoordinate{0_n, 0_n},
              OperatorAtomicTaskShardBinding{
                {
                  {TensorSlotName::INPUT, ptensor_coord(0_n, 0_n)},
                  {TensorSlotName::WEIGHT, ptensor_coord(0_n, 0_n)},
                },
              },
            },
            {
              MachineSpaceCoordinate{0_n, 1_n},
              OperatorAtomicTaskShardBinding{
                {
                  {TensorSlotName::INPUT, ptensor_coord(0_n, 1_n)},
                  {TensorSlotName::WEIGHT, ptensor_coord(1_n, 0_n)},
                },
              },
            },
            {
              MachineSpaceCoordinate{1_n, 0_n},
              OperatorAtomicTaskShardBinding{
                {
                  {TensorSlotName::INPUT, ptensor_coord(0_n, 2_n)},
                  {TensorSlotName::WEIGHT, ptensor_coord(2_n, 0_n)},
                },
              },
            },
          },
        },
      },
    };

    MappedParallelComputationGraph correct_mpcg = 
      mapped_pcg_from_pcg_and_mapped_op_task_groups(correct_pcg, correct_mapping);

    // Extra asserts are only here to improve the error message quality on a
    // test failure. Only the last check is actually needed to guarantee
    // correctness, and is a strictly stronger condition thean the first two.
    ASSERT(mpcg_get_parallel_layers(result).size() == mpcg_get_parallel_layers(correct_mpcg).size());
    ASSERT(pcgs_are_isomorphic(pcg_from_mpcg(result), pcg_from_mpcg(correct_mpcg)));
    ASSERT(mapped_pcgs_are_isomorphic(result, correct_mpcg));
  }
}
