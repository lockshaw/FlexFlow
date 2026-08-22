#include "op-attrs/ops/repartition.h"
#include "op-attrs/operator_task_space.h"
#include "test/utils/doctest/fmt/expected.h"
#include <doctest/doctest.h>
#include "op-attrs/parallel_tensor_dim_idx_t.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("repartition_get_output_parallel_shape") {
    ff_dim_t dim = ff_dim_t{2_n};
    RepartitionAttrs attrs = RepartitionAttrs{
        /*repartition_dim=*/dim,
        /*repartition_degree=*/4_ge2,
    };

    ParallelTensorShape input = ParallelTensorShape{
        ParallelTensorDims{
            FFOrdered<ShardParallelDim>{
                ShardParallelDim{12_p, 2_p},
                ShardParallelDim{14_p, 1_p},
                ShardParallelDim{16_p, 3_p},
                ShardParallelDim{18_p, 2_p},
            },
            ReplicaParallelDimSet{
                SumDegree{3_p},
                DiscardCopyDegree{2_p},
            },
        },
        DataType::FLOAT,
    };

    ParallelTensorShape result =
        repartition_get_output_parallel_shape(attrs, input);

    ParallelTensorShape correct = [&] {
      ParallelTensorShape output = input;
      output.dims.shard_dims.at(dim).degree = 12_p;
      return output;
    }();

    CHECK(result == correct);
  }

  TEST_CASE("repartition_get_output_parallel_dim_degrees") {
    RepartitionAttrs attrs = RepartitionAttrs{
        /*repartition_dim=*/ff_dim_t{1_n},
        /*repartition_degree=*/2_ge2,
    };

    ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
        SumDegree{2_p},
        DiscardCopyDegree{1_p},
        FFOrdered<positive_int>{
            2_p,
            3_p,
        },
    };

    ParallelTensorDimDegrees result =
        repartition_get_output_parallel_dim_degrees(attrs, input_degrees);

    ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
        SumDegree{2_p},
        DiscardCopyDegree{1_p},
        FFOrdered<positive_int>{
            2_p,
            6_p,
        },
    };

    CHECK(result == correct);
  }

  TEST_CASE("repartition_get_operator_task_space") {
    RepartitionAttrs attrs = RepartitionAttrs{
        /*repartition_dim=*/ff_dim_t{1_n},
        /*repartition_degree=*/3_ge2,
    };

    SUBCASE("input tensor is already parallel in the repartion dimension") {
      ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
          SumDegree{2_p},
          DiscardCopyDegree{1_p},
          FFOrdered<positive_int>{
              2_p,
              3_p,
          },
      };

      OperatorTaskSpace result =
          repartition_get_operator_task_space(attrs, input_degrees);

      OperatorTaskSpace correct = operator_task_space_from_minimal_dim_domain(
          MinimalDimDomain<operator_task_space_dim_idx_t>{
              std::map<operator_task_space_dim_idx_t, int_ge_two>{
                  {operator_task_space_dim_idx_t{0_n}, 2_ge2},
                  {operator_task_space_dim_idx_t{1_n}, 2_ge2},
                  {operator_task_space_dim_idx_t{2_n}, 9_ge2},
              },
          });

      CHECK(result == correct);
    }

    SUBCASE("input tensor is not already parallel in the repartition dimension") {
      ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
          SumDegree{2_p},
          DiscardCopyDegree{1_p},
          FFOrdered<positive_int>{
              2_p,
              1_p,
          },
      };

      OperatorTaskSpace result =
          repartition_get_operator_task_space(attrs, input_degrees);
      OperatorTaskSpace correct = operator_task_space_from_minimal_dim_domain(
          MinimalDimDomain<operator_task_space_dim_idx_t>{
              std::map<operator_task_space_dim_idx_t, int_ge_two>{
                  {operator_task_space_dim_idx_t{0_n}, 2_ge2},
                  {operator_task_space_dim_idx_t{1_n}, 2_ge2},
                  {operator_task_space_dim_idx_t{2_n}, 3_ge2},
              },
          });

      CHECK(result == correct);
    }
  }

  TEST_CASE("repartition_get_operator_to_input_mapping") {
    RepartitionAttrs attrs = RepartitionAttrs{
        /*repartition_dim=*/ff_dim_t{1_n},
        /*repartition_degree=*/3_ge2,
    };

    auto op_task_coord = [&](nonnegative_int sum_coord,
                             nonnegative_int shard_dim1_coord) {
      return DimCoord<operator_task_space_dim_idx_t>{
        std::map<operator_task_space_dim_idx_t, nonnegative_int>{
          { operator_task_space_dim_idx_t{0_n}, sum_coord },
          { operator_task_space_dim_idx_t{1_n}, shard_dim1_coord },
        },
      };
    };

    SUBCASE("input tensor is already parallel in the repartion dimension") {
      ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
          SumDegree{2_p},
          DiscardCopyDegree{1_p},
          FFOrdered<positive_int>{
              1_p,
              2_p,
          },
      };

      auto input_coord = [&](nonnegative_int sum_coord,
                             nonnegative_int shard_dim1_coord)
        -> DimCoord<parallel_tensor_dim_idx_t>
      {
        return DimCoord<parallel_tensor_dim_idx_t>{
          std::map<parallel_tensor_dim_idx_t, nonnegative_int>{
            { sum_dim_idx(), sum_coord },
            { discard_copy_dim_idx(), 0_n },
            { shard_dim_idx(ff_dim_t{0_n}), 0_n },
            { shard_dim_idx(ff_dim_t{1_n}), shard_dim1_coord },
          },
        };
      };

      OperatorSpaceToParallelTensorSpaceMapping result =
          repartition_get_operator_to_input_mapping(attrs, input_degrees);

      OperatorSpaceToParallelTensorSpaceMapping correct =
        OperatorSpaceToParallelTensorSpaceMapping{
          DimDomainHemiuniqueMapping<
            operator_task_space_dim_idx_t,
            parallel_tensor_dim_idx_t
          >{
            /*coord_mapping=*/HemiuniqueBinaryRelation<
              DimCoord<operator_task_space_dim_idx_t>,
              DimCoord<parallel_tensor_dim_idx_t>
            >{
              ManyToOne<
                DimCoord<operator_task_space_dim_idx_t>,
                DimCoord<parallel_tensor_dim_idx_t>
              >{
                {
                  {op_task_coord(0_n, 0_n), op_task_coord(0_n, 1_n), op_task_coord(0_n, 2_n)},
                  input_coord(0_n, 0_n),
                },
                {
                  {op_task_coord(0_n, 3_n), op_task_coord(0_n, 4_n), op_task_coord(0_n, 5_n)},
                  input_coord(0_n, 1_n),
                },
                {
                  {op_task_coord(1_n, 0_n), op_task_coord(1_n, 1_n), op_task_coord(1_n, 2_n)},
                  input_coord(1_n, 0_n),
                },
                {
                  {op_task_coord(1_n, 3_n), op_task_coord(1_n, 4_n), op_task_coord(1_n, 5_n)},
                  input_coord(1_n, 1_n),
                }
              },
            },
            /*l_domain=*/
              lift_minimal_dim_domain(
                minimal_dim_domain_from_operator_task_space(
                  repartition_get_operator_task_space(attrs, input_degrees))),
            /*r_domain=*/
              dim_domain_from_parallel_tensor_dim_degrees(input_degrees),
          },
        };

      CHECK(result == correct);
    }

    SUBCASE("input tensor is not already parallel in the repartion dimension") {
      ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
          SumDegree{2_p},
          DiscardCopyDegree{1_p},
          FFOrdered<positive_int>{
              1_p,
              1_p,
          },
      };

      auto input_coord = [&](nonnegative_int sum_coord)
        -> DimCoord<parallel_tensor_dim_idx_t>
      {
        return DimCoord<parallel_tensor_dim_idx_t>{
          std::map<parallel_tensor_dim_idx_t, nonnegative_int>{
            { sum_dim_idx(), sum_coord },
            { discard_copy_dim_idx(), 0_n },
            { shard_dim_idx(ff_dim_t{0_n}), 0_n },
            { shard_dim_idx(ff_dim_t{1_n}), 0_n },
          },
        };
      };

      OperatorSpaceToParallelTensorSpaceMapping result =
          repartition_get_operator_to_input_mapping(attrs, input_degrees);

      OperatorSpaceToParallelTensorSpaceMapping correct =
        OperatorSpaceToParallelTensorSpaceMapping{
          DimDomainHemiuniqueMapping<
            operator_task_space_dim_idx_t,
            parallel_tensor_dim_idx_t
          >{
            /*coord_mapping=*/HemiuniqueBinaryRelation<
              DimCoord<operator_task_space_dim_idx_t>,
              DimCoord<parallel_tensor_dim_idx_t>
            >{
              ManyToOne<
                DimCoord<operator_task_space_dim_idx_t>,
                DimCoord<parallel_tensor_dim_idx_t>
              >{
                {
                  {op_task_coord(0_n, 0_n), op_task_coord(0_n, 1_n), op_task_coord(0_n, 2_n)},
                  input_coord(0_n),
                },
                {
                  {op_task_coord(1_n, 0_n), op_task_coord(1_n, 1_n), op_task_coord(1_n, 2_n)},
                  input_coord(1_n),
                },
              },
            },
            /*l_domain=*/
              lift_minimal_dim_domain(
                minimal_dim_domain_from_operator_task_space(
                  repartition_get_operator_task_space(attrs, input_degrees))),
            /*r_domain=*/
              dim_domain_from_parallel_tensor_dim_degrees(input_degrees),
          },
        };

      CHECK(result == correct);
    }
  }

  TEST_CASE("repartition_get_operator_to_output_mapping") {
    RepartitionAttrs attrs = RepartitionAttrs{
        /*repartition_dim=*/ff_dim_t{1_n},
        /*repartition_degree=*/2_ge2,
    };

    auto op_task_coord = [&](nonnegative_int sum_coord,
                             nonnegative_int shard_dim1_coord) {
      return DimCoord<operator_task_space_dim_idx_t>{
        std::map<operator_task_space_dim_idx_t, nonnegative_int>{
          { operator_task_space_dim_idx_t{0_n}, sum_coord },
          { operator_task_space_dim_idx_t{1_n}, shard_dim1_coord },
        },
      };
    };

    ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
        SumDegree{2_p},
        DiscardCopyDegree{1_p},
        FFOrdered<positive_int>{
            1_p,
            2_p,
        },
    };

    ParallelTensorDimDegrees output_degrees = ParallelTensorDimDegrees{
        SumDegree{2_p},
        DiscardCopyDegree{1_p},
        FFOrdered<positive_int>{
            1_p,
            4_p,
        },
    };

    auto output_coord = [&](nonnegative_int sum_coord,
                           nonnegative_int shard_dim1_coord)
      -> DimCoord<parallel_tensor_dim_idx_t>
    {
      return DimCoord<parallel_tensor_dim_idx_t>{
        std::map<parallel_tensor_dim_idx_t, nonnegative_int>{
          { sum_dim_idx(), sum_coord },
          { discard_copy_dim_idx(), 0_n },
          { shard_dim_idx(ff_dim_t{0_n}), 0_n },
          { shard_dim_idx(ff_dim_t{1_n}), shard_dim1_coord },
        },
      };
    };

    OperatorSpaceToParallelTensorSpaceBiuniqueMapping result =
        repartition_get_operator_to_output_mapping(attrs, input_degrees);

    OperatorSpaceToParallelTensorSpaceBiuniqueMapping correct =
      OperatorSpaceToParallelTensorSpaceBiuniqueMapping{
        DimDomainBiuniqueMapping<
          operator_task_space_dim_idx_t,
          parallel_tensor_dim_idx_t
        >{
          /*coord_mapping=*/bidict<
            DimCoord<operator_task_space_dim_idx_t>,
            DimCoord<parallel_tensor_dim_idx_t>
          >{
            {
              op_task_coord(0_n, 0_n),
              output_coord(0_n, 0_n),
            },
            {
              op_task_coord(0_n, 1_n),
              output_coord(0_n, 1_n),
            },
            {
              op_task_coord(0_n, 2_n),
              output_coord(0_n, 2_n),
            },
            {
              op_task_coord(0_n, 3_n),
              output_coord(0_n, 3_n),
            },
            {
              op_task_coord(1_n, 0_n),
              output_coord(1_n, 0_n),
            },
            {
              op_task_coord(1_n, 1_n),
              output_coord(1_n, 1_n),
            },
            {
              op_task_coord(1_n, 2_n),
              output_coord(1_n, 2_n),
            },
            {
              op_task_coord(1_n, 3_n),
              output_coord(1_n, 3_n),
            },
          },
          /*l_domain=*/
            lift_minimal_dim_domain(
              minimal_dim_domain_from_operator_task_space(
                repartition_get_operator_task_space(attrs, input_degrees))),
          /*r_domain=*/
            dim_domain_from_parallel_tensor_dim_degrees(output_degrees),
        },
      };

    CHECK(result == correct);
  }
}
