#include "compiler/data_parallelism/data_parallelism.h"
#include "compiler/search_result.h"
#include "op-attrs/initializer_attrs.h"
#include "pcg/computation_graph_builder.h"
#include "pcg/mapped_parallel_computation_graph/mapped_parallel_computation_graph.h"
#include "pcg/parallel_computation_graph/explicit_parallel_computation_graph_builder.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_builder.h"
#include <doctest/doctest.h>
#include "models/split_test/split_test.h"
#include "models/transformer/transformer.h"
#include "models/inception_v3/inception_v3.h"
#include "models/candle_uno/candle_uno.h"
#include "models/bert/bert.h"
#include "models/dlrm/dlrm.h"
#include "models/yolov10/yolov10.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("apply_data_parallelism") {
    SUBCASE("small sample computation graph") {
      TensorShape input_shape = TensorShape{
          TensorDims{
              FFOrdered<positive_int>{
                  9_p,
                  6_p,
              },
          },
          DataType::FLOAT,
      };

      TensorShape weight_shape = TensorShape{
          TensorDims{
              FFOrdered<positive_int>{
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
            /*outDim=*/3_p,
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

      MappedParallelComputationGraph result = get_mapped_pcg_from_search_result(
          apply_data_parallelism(cg, degree), machine_spec);

      std::string input1_name = "input1";
      std::string input2_name = "input2";
      std::string partition1_name = "partition1";
      std::string partition2_name = "partition2";
      std::string projector_weight_name = "projector_weight";
      std::string replicate_name = "replicate";
      std::string add_name = "add";
      std::string dense_name = "dense";

      ParallelComputationGraph correct_pcg = [&] {
        ExplicitParallelComputationGraphBuilder b;

        parallel_tensor_guid_t t1 =
            b.create_input_tensor(input_shape, input1_name);
        t1 = b.parallel_partition(t1, ff_dim_t{0_n}, degree, partition1_name);

        parallel_tensor_guid_t t2 =
            b.create_input_tensor(input_shape, input2_name);
        t2 = b.parallel_partition(t2, ff_dim_t{0_n}, degree, partition2_name);

        parallel_tensor_guid_t t3 = b.add(t1, t2, add_name);

        parallel_tensor_guid_t t_projector = b.create_weight_tensor(
            weight_shape, make_zero_initializer(), projector_weight_name);
        t_projector = b.parallel_replicate(t_projector, degree, replicate_name);

        parallel_tensor_guid_t t4 = b.dense(
            /*input=*/t1,
            /*outDim=*/3_p,
            /*projector=*/t_projector,
            /*bias=*/std::nullopt,
            /*activation=*/std::nullopt,
            /*data_type=*/DataType::FLOAT,
            /*name=*/dense_name);

        return b.pcg;
      }();

      parallel_layer_guid_t l_input1 =
          get_parallel_layer_by_name(correct_pcg, input1_name);
      parallel_layer_guid_t l_input2 =
          get_parallel_layer_by_name(correct_pcg, input2_name);
      parallel_layer_guid_t l_projector_weight =
          get_parallel_layer_by_name(correct_pcg, projector_weight_name);
      parallel_layer_guid_t l_replicate =
          get_parallel_layer_by_name(correct_pcg, replicate_name);
      parallel_layer_guid_t l_partition1 =
          get_parallel_layer_by_name(correct_pcg, partition1_name);
      parallel_layer_guid_t l_partition2 =
          get_parallel_layer_by_name(correct_pcg, partition2_name);
      parallel_layer_guid_t l_add =
          get_parallel_layer_by_name(correct_pcg, add_name);
      parallel_layer_guid_t l_dense =
          get_parallel_layer_by_name(correct_pcg, dense_name);

      auto ptensor_coord =
          [](nonnegative_int discard_copy_component,
             nonnegative_int batch_component) -> ParallelTensorSpaceCoordinate {
        return ParallelTensorSpaceCoordinate{
            /*sum_component=*/0_n,
            /*discard_copy_component=*/discard_copy_component,
            /*shard_components=*/FFOrdered{batch_component, 0_n},
        };
      };

      MappedOperatorTaskGroup input_mapping = MappedOperatorTaskGroup{
          bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding>{
              {
                  MachineSpaceCoordinate{0_n, 0_n},
                  OperatorAtomicTaskShardBinding{
                      {
                          {TensorSlotName::OUTPUT, ptensor_coord(0_n, 0_n)},
                      },
                  },
              },
          },
      };

      MappedOperatorTaskGroup partition_mapping = MappedOperatorTaskGroup{
          bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding>{
              {
                  MachineSpaceCoordinate{0_n, 0_n},
                  OperatorAtomicTaskShardBinding{
                      {
                          {TensorSlotName::INPUT, ptensor_coord(0_n, 0_n)},
                          {TensorSlotName::OUTPUT, ptensor_coord(0_n, 0_n)},
                      },
                  },
              },
              {
                  MachineSpaceCoordinate{0_n, 1_n},
                  OperatorAtomicTaskShardBinding{
                      {
                          {TensorSlotName::INPUT, ptensor_coord(0_n, 0_n)},
                          {TensorSlotName::OUTPUT, ptensor_coord(0_n, 1_n)},
                      },
                  },
              },
          },
      };

      MappedOperatorTaskGroup replicate_mapping = MappedOperatorTaskGroup{
          bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding>{
              {
                  MachineSpaceCoordinate{0_n, 0_n},
                  OperatorAtomicTaskShardBinding{
                      {
                          {TensorSlotName::INPUT, ptensor_coord(0_n, 0_n)},
                          {TensorSlotName::OUTPUT, ptensor_coord(0_n, 0_n)},
                      },
                  },
              },
              {
                  MachineSpaceCoordinate{0_n, 1_n},
                  OperatorAtomicTaskShardBinding{
                      {
                          {TensorSlotName::INPUT, ptensor_coord(0_n, 0_n)},
                          {TensorSlotName::OUTPUT, ptensor_coord(1_n, 0_n)},
                      },
                  },
              },
          },
      };

      std::map<parallel_layer_guid_t, MappedOperatorTaskGroup>
          correct_mapping = {
              {
                  l_input1,
                  input_mapping,
              },
              {
                  l_input2,
                  input_mapping,
              },
              {
                  l_projector_weight,
                  input_mapping,
              },
              {
                  l_partition1,
                  partition_mapping,
              },
              {
                  l_partition2,
                  partition_mapping,
              },
              {
                  l_replicate,
                  replicate_mapping,
              },
              {
                  l_add,
                  MappedOperatorTaskGroup{
                      bidict<MachineSpaceCoordinate,
                             OperatorAtomicTaskShardBinding>{
                          {
                              MachineSpaceCoordinate{0_n, 0_n},
                              OperatorAtomicTaskShardBinding{
                                  {
                                      {TensorSlotName::LHS_INPUT,
                                       ptensor_coord(0_n, 0_n)},
                                      {TensorSlotName::RHS_INPUT,
                                       ptensor_coord(0_n, 0_n)},
                                      {TensorSlotName::OUTPUT,
                                       ptensor_coord(0_n, 0_n)},
                                  },
                              },
                          },
                          {
                              MachineSpaceCoordinate{0_n, 1_n},
                              OperatorAtomicTaskShardBinding{
                                  {
                                      {TensorSlotName::LHS_INPUT,
                                       ptensor_coord(0_n, 1_n)},
                                      {TensorSlotName::RHS_INPUT,
                                       ptensor_coord(0_n, 1_n)},
                                      {TensorSlotName::OUTPUT,
                                       ptensor_coord(0_n, 1_n)},
                                  },
                              },
                          },
                          {
                              MachineSpaceCoordinate{1_n, 0_n},
                              OperatorAtomicTaskShardBinding{
                                  {
                                      {TensorSlotName::LHS_INPUT,
                                       ptensor_coord(0_n, 2_n)},
                                      {TensorSlotName::RHS_INPUT,
                                       ptensor_coord(0_n, 2_n)},
                                      {TensorSlotName::OUTPUT,
                                       ptensor_coord(0_n, 2_n)},
                                  },
                              },
                          },
                      },
                  },
              },
              {
                  l_dense,
                  MappedOperatorTaskGroup{
                      bidict<MachineSpaceCoordinate,
                             OperatorAtomicTaskShardBinding>{
                          {
                              MachineSpaceCoordinate{0_n, 0_n},
                              OperatorAtomicTaskShardBinding{
                                  {
                                      {TensorSlotName::INPUT,
                                       ptensor_coord(0_n, 0_n)},
                                      {TensorSlotName::WEIGHT,
                                       ptensor_coord(0_n, 0_n)},
                                      {TensorSlotName::OUTPUT,
                                       ptensor_coord(0_n, 0_n)},
                                  },
                              },
                          },
                          {
                              MachineSpaceCoordinate{0_n, 1_n},
                              OperatorAtomicTaskShardBinding{
                                  {
                                      {TensorSlotName::INPUT,
                                       ptensor_coord(0_n, 1_n)},
                                      {TensorSlotName::WEIGHT,
                                       ptensor_coord(1_n, 0_n)},
                                      {TensorSlotName::OUTPUT,
                                       ptensor_coord(0_n, 1_n)},
                                  },
                              },
                          },
                          {
                              MachineSpaceCoordinate{1_n, 0_n},
                              OperatorAtomicTaskShardBinding{
                                  {
                                      {TensorSlotName::INPUT,
                                       ptensor_coord(0_n, 2_n)},
                                      {TensorSlotName::WEIGHT,
                                       ptensor_coord(2_n, 0_n)},
                                      {TensorSlotName::OUTPUT,
                                       ptensor_coord(0_n, 2_n)},
                                  },
                              },
                          },
                      },
                  },
              },
          };

      MappedParallelComputationGraph correct_mpcg =
          mapped_pcg_from_pcg_and_mapped_op_task_groups(correct_pcg,
                                                        correct_mapping);

      // Extra asserts are only here to improve the error message quality on a
      // test failure. Only the last check is actually needed to guarantee
      // correctness, and is a strictly stronger condition thean the first two.
      ASSERT(mpcg_get_parallel_layers(result).size() ==
             mpcg_get_parallel_layers(correct_mpcg).size());
      ASSERT(pcgs_are_isomorphic(pcg_from_mpcg(result),
                                 pcg_from_mpcg(correct_mpcg)));
      ASSERT(mapped_pcgs_are_isomorphic(result, correct_mpcg));
    }

    SUBCASE("everything in lib/models can be made data parallel") {
      int_ge_two degree = 4_ge2;

      MachineComputeSpecification machine_spec = MachineComputeSpecification{
          /*num_nodes=*/4_p,
          /*num_cpus_per_node=*/1_p,
          /*num_gpus_per_node=*/2_p,
      };
      
      SUBCASE("split test") {
        ComputationGraph cg =
            get_split_test_computation_graph(/*batch_size=*/8_p);

        MappedParallelComputationGraph result = get_mapped_pcg_from_search_result(
            apply_data_parallelism(cg, degree), machine_spec);

        // currently just check that it doesn't crash
      }

      SUBCASE("transformer") {
        ComputationGraph cg =
            get_transformer_computation_graph(get_default_transformer_config());

        MappedParallelComputationGraph result = get_mapped_pcg_from_search_result(
            apply_data_parallelism(cg, degree), machine_spec);
        
        // currently just check that it doesn't crash
      }

      SUBCASE("inception_v3") {
        ComputationGraph cg = get_inception_v3_computation_graph(
            get_default_inception_v3_training_config());

        MappedParallelComputationGraph result = get_mapped_pcg_from_search_result(
            apply_data_parallelism(cg, degree), machine_spec);

        // currently just check that it doesn't crash
      }

      SUBCASE("candle_uno") {
        ComputationGraph cg =
            get_candle_uno_computation_graph(get_default_candle_uno_config());

        MappedParallelComputationGraph result = get_mapped_pcg_from_search_result(
            apply_data_parallelism(cg, degree), machine_spec);

        // currently just check that it doesn't crash
      }

      SUBCASE("bert") {
        ComputationGraph cg =
            get_bert_computation_graph(get_default_bert_config());

        MappedParallelComputationGraph result = get_mapped_pcg_from_search_result(
            apply_data_parallelism(cg, degree), machine_spec);

        // currently just check that it doesn't crash
      }

      SUBCASE("dlrm") {
        ComputationGraph cg =
            get_dlrm_computation_graph(get_default_dlrm_config());

        MappedParallelComputationGraph result = get_mapped_pcg_from_search_result(
            apply_data_parallelism(cg, degree), machine_spec);

        // currently just check that it doesn't crash
      }

      SUBCASE("yolov10x") {
        ComputationGraph cg = get_yolov10_computation_graph(get_yolov10x_config(
            /*batch_size=*/8_p,
            /*end2end=*/false));

        MappedParallelComputationGraph result = get_mapped_pcg_from_search_result(
            apply_data_parallelism(cg, degree), machine_spec);

        // currently just check that it doesn't crash
      }
    }
  }
}
