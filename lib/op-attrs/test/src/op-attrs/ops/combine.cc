#include "op-attrs/ops/combine.h"
#include "op-attrs/operator_task_space.h"
#include "test/utils/doctest/fmt/expected.h"
#include <doctest/doctest.h>
#include "op-attrs/parallel_tensor_dim_idx_t.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "utils/orthotope/minimal_dim_domain.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_biunique_mapping.dtg.h"
#include "utils/orthotope/dim_domain_biunique_mapping.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("combine_get_output_parallel_shape") {

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

    SUBCASE("valid") {
      ff_dim_t dim = ff_dim_t{2_n};
      int_ge_two degree = 3_ge2;
      CombineAttrs attrs = CombineAttrs{
          /*combine_dim=*/dim,
          /*combine_degree=*/degree,
      };

      ParallelTensorShape result =
          combine_get_output_parallel_shape(attrs, input);

      ParallelTensorShape correct = [&] {
        ParallelTensorShape output = input;
        output.dims.shard_dims.at(dim).degree = 1_p;
        return output;
      }();

      CHECK(result == correct);
    }

    SUBCASE("invalid") {
      ff_dim_t dim = ff_dim_t{2_n};
      int_ge_two degree = 4_ge2;
      CombineAttrs attrs = CombineAttrs{
          /*combine_dim=*/dim,
          /*combine_degree=*/degree,
      };

      CHECK_THROWS(combine_get_output_parallel_shape(attrs, input));
    }
  }

  TEST_CASE("combine_get_output_parallel_dim_degrees") {
    CombineAttrs attrs = CombineAttrs{
        /*combine_dim=*/ff_dim_t{0_n},
        /*combine_degree=*/3_ge2,
    };

    ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
        SumDegree{2_p},
        DiscardCopyDegree{1_p},
        FFOrdered<positive_int>{
            6_p,
            3_p,
        },
    };

    ParallelTensorDimDegrees result =
        combine_get_output_parallel_dim_degrees(attrs, input_degrees);

    ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
        SumDegree{2_p},
        DiscardCopyDegree{1_p},
        FFOrdered<positive_int>{
            2_p,
            3_p,
        },
    };

    CHECK(result == correct);
  }

  TEST_CASE("combine_get_operator_task_space") {
    CombineAttrs attrs = CombineAttrs{
        /*combine_dim=*/ff_dim_t{0_n},
        /*combine_degree=*/3_ge2,
    };

    ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
        SumDegree{2_p},
        DiscardCopyDegree{1_p},
        FFOrdered<positive_int>{
            4_p,
            2_p,
        },
    };

    OperatorTaskSpace result = combine_get_operator_task_space(attrs, input_degrees);

   OperatorTaskSpace correct = operator_task_space_from_minimal_dim_domain(
        MinimalDimDomain<operator_task_space_dim_idx_t>{
            std::map<operator_task_space_dim_idx_t, int_ge_two>{
                {operator_task_space_dim_idx_t{0_n}, 2_ge2},
                {operator_task_space_dim_idx_t{1_n}, 4_ge2},
                {operator_task_space_dim_idx_t{2_n}, 2_ge2},
            },
        });

    CHECK(result == correct);
  }

  TEST_CASE("combine_get_operator_to_input_mapping") {
    CombineAttrs attrs = CombineAttrs{
        /*combine_dim=*/ff_dim_t{1_n},
        /*combine_degree=*/3_ge2,
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
            6_p,
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

    OperatorSpaceToParallelTensorSpaceBiuniqueMapping result =
        combine_get_operator_to_input_mapping(attrs, input_degrees);

    OperatorSpaceToParallelTensorSpaceBiuniqueMapping correct =
      OperatorSpaceToParallelTensorSpaceBiuniqueMapping{
        DimDomainBiuniqueMapping<
          operator_task_space_dim_idx_t,
          parallel_tensor_dim_idx_t
        >{
          /*coord_mapping=*/
          bidict<
            DimCoord<operator_task_space_dim_idx_t>,
            DimCoord<parallel_tensor_dim_idx_t>
          >{
            {
              op_task_coord(0_n, 0_n),
              input_coord(0_n, 0_n),
            },
            {
              op_task_coord(0_n, 1_n),
              input_coord(0_n, 1_n),
            },
            {
              op_task_coord(0_n, 2_n),
              input_coord(0_n, 2_n),
            },
            {
              op_task_coord(0_n, 3_n),
              input_coord(0_n, 3_n),
            },
            {
              op_task_coord(0_n, 4_n),
              input_coord(0_n, 4_n),
            },
            {
              op_task_coord(0_n, 5_n),
              input_coord(0_n, 5_n),
            },
            {
              op_task_coord(1_n, 0_n),
              input_coord(1_n, 0_n),
            },
            {
              op_task_coord(1_n, 1_n),
              input_coord(1_n, 1_n),
            },
            {
              op_task_coord(1_n, 2_n),
              input_coord(1_n, 2_n),
            },
            {
              op_task_coord(1_n, 3_n),
              input_coord(1_n, 3_n),
            },
            {
              op_task_coord(1_n, 4_n),
              input_coord(1_n, 4_n),
            },
            {
              op_task_coord(1_n, 5_n),
              input_coord(1_n, 5_n),
            },
          },
          /*l_domain=*/
            lift_minimal_dim_domain(
              minimal_dim_domain_from_operator_task_space(
                combine_get_operator_task_space(attrs, input_degrees))),
          /*r_domain=*/
            dim_domain_from_parallel_tensor_dim_degrees(input_degrees),
        },
      };

    CHECK(result == correct);
  }

  TEST_CASE("combine_get_operator_to_output_mapping") {
    CombineAttrs attrs = CombineAttrs{
        /*combine_dim=*/ff_dim_t{1_n},
        /*combine_degree=*/3_ge2,
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

    SUBCASE("output tensor remains parallel in the partition dimension") {
      ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
          SumDegree{2_p},
          DiscardCopyDegree{1_p},
          FFOrdered<positive_int>{
              1_p,
              6_p,
          },
      };

      ParallelTensorDimDegrees output_degrees = ParallelTensorDimDegrees{
          SumDegree{2_p},
          DiscardCopyDegree{1_p},
          FFOrdered<positive_int>{
              1_p,
              2_p,
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

      OperatorSpaceToParallelTensorSpaceMapping result =
          combine_get_operator_to_output_mapping(attrs, input_degrees);

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
                  output_coord(0_n, 0_n),
                },
                {
                  {op_task_coord(0_n, 3_n), op_task_coord(0_n, 4_n), op_task_coord(0_n, 5_n)},
                  output_coord(0_n, 1_n),
                },
                {
                  {op_task_coord(1_n, 0_n), op_task_coord(1_n, 1_n), op_task_coord(1_n, 2_n)},
                  output_coord(1_n, 0_n),
                },
                {
                  {op_task_coord(1_n, 3_n), op_task_coord(1_n, 4_n), op_task_coord(1_n, 5_n)},
                  output_coord(1_n, 1_n),
                },
              },
            },
            /*l_domain=*/
              lift_minimal_dim_domain(
                minimal_dim_domain_from_operator_task_space(
                  combine_get_operator_task_space(attrs, input_degrees))),
            /*r_domain=*/
              dim_domain_from_parallel_tensor_dim_degrees(output_degrees),
          },
        };

      CHECK(result == correct);
    }

    SUBCASE("output tensor is no longer parallel in the combine dimension") {
      ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
          SumDegree{2_p},
          DiscardCopyDegree{1_p},
          FFOrdered<positive_int>{
              1_p,
              3_p,
          },
      };

      ParallelTensorDimDegrees output_degrees = ParallelTensorDimDegrees{
          SumDegree{2_p},
          DiscardCopyDegree{1_p},
          FFOrdered<positive_int>{
              1_p,
              1_p,
          },
      };

      auto output_coord = [&](nonnegative_int sum_coord)
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
          combine_get_operator_to_output_mapping(attrs, input_degrees);

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
                  output_coord(0_n),
                },
                {
                  {op_task_coord(1_n, 0_n), op_task_coord(1_n, 1_n), op_task_coord(1_n, 2_n)},
                  output_coord(1_n),
                },
              },
            },
            /*l_domain=*/
              lift_minimal_dim_domain(
                minimal_dim_domain_from_operator_task_space(
                  combine_get_operator_task_space(attrs, input_degrees))),
            /*r_domain=*/
              dim_domain_from_parallel_tensor_dim_degrees(output_degrees),
          },
        };

      CHECK(result == correct);
    }
  }
}
