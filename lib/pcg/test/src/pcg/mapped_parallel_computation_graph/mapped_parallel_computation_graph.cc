#include <doctest/doctest.h>
#include "op-attrs/initializer_attrs.h"
#include "pcg/mapped_parallel_computation_graph/mapped_parallel_computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_builder.h"
#include "utils/containers/require_only_key.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("mapped_pcgs_are_isomorphic") {
    auto make_mpcg = []() -> MappedParallelComputationGraph {
      TensorShape input_shape = TensorShape{
        TensorDims{
          FFOrdered<positive_int>{
            8_p,
            3_p,
            6_p,
          },
        },
        DataType::FLOAT,
      };

      ParallelComputationGraphBuilder b;

      parallel_tensor_guid_t t1 = b.create_input_tensor(input_shape);
      t1 = b.parallel_partition(t1, ff_dim_t{0_n}, 2_p);
      parallel_tensor_guid_t t2 = b.create_input_tensor(input_shape);
      t2 = b.parallel_partition(t2, ff_dim_t{0_n}, 2_p);

      std::string add_name = "add";

      parallel_tensor_guid_t t3 = b.add(t1, t2, add_name);

      ParallelComputationGraph pcg = b.pcg;

      parallel_layer_guid_t l_add = get_parallel_layer_by_name(pcg, add_name);

      auto machine_coord = [](nonnegative_int x) -> MachineSpaceCoordinate {
        return MachineSpaceCoordinate{
          /*node_idx=*/0_n,
          /*device_idx=*/x,
        };
      };

      auto ptensor_coord = [](nonnegative_int x) -> ParallelTensorSpaceCoordinate {
        return ParallelTensorSpaceCoordinate{
          /*sum_component=*/0_n,
          /*discard_copy_component=*/0_n,
          /*shard_components=*/FFOrdered{x, 0_n, 0_n},
        };
      };


      std::unordered_map<parallel_layer_guid_t, MappedOperatorTaskGroup> mapped_tasks = {
        {
          l_add,
          MappedOperatorTaskGroup{
            bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding>{
              {
                machine_coord(0_n),
                OperatorAtomicTaskShardBinding{
                  {
                    {TensorSlotName::LHS_INPUT, ptensor_coord(0_n)},
                    {TensorSlotName::RHS_INPUT, ptensor_coord(0_n)},
                  },
                }
              },
              {
                machine_coord(1_n),
                OperatorAtomicTaskShardBinding{
                  {
                    {TensorSlotName::LHS_INPUT, ptensor_coord(1_n)},
                    {TensorSlotName::RHS_INPUT, ptensor_coord(1_n)},
                  },
                }
              },
            },
          }
        },
      };

      return mapped_pcg_from_pcg_and_mapped_op_task_groups(pcg, mapped_tasks);
    };

    MappedParallelComputationGraph mpcg1 = make_mpcg();
    MappedParallelComputationGraph mpcg2 = make_mpcg();

    CHECK(mapped_pcgs_are_isomorphic(mpcg1, mpcg2));
  }
}
