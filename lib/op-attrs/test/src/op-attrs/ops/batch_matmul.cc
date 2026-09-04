#include "op-attrs/ops/batch_matmul.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("batch_matmul_get_output_shape") {
    BatchMatmulAttrs attrs = BatchMatmulAttrs{};

    TensorShape lhs = TensorShape{
        TensorDims{
            FFOrdered<positive_int>{
                5_p,
                4_p,
                12_p,
                3_p,
            },
        },
        DataType::FLOAT,
    };

    SUBCASE("inner dimensions match") {
      TensorShape rhs = TensorShape{
          TensorDims{
              FFOrdered<positive_int>{
                  5_p,
                  4_p,
                  3_p,
                  8_p,
              },
          },
          DataType::FLOAT,
      };

      TensorShape result = batch_matmul_get_output_shape(attrs, lhs, rhs);

      TensorShape correct = TensorShape{
          TensorDims{
              FFOrdered<positive_int>{
                  5_p,
                  4_p,
                  12_p,
                  8_p,
              },
          },
          DataType::FLOAT,
      };

      CHECK(result == correct);
    }

    SUBCASE("inner dimensions don't match") {
      TensorShape rhs = TensorShape{
          TensorDims{
              FFOrdered<positive_int>{
                  5_p,
                  4_p,
                  4_p,
                  8_p,
              },
          },
          DataType::FLOAT,
      };

      CHECK_THROWS(batch_matmul_get_output_shape(attrs, lhs, rhs));
    }

    SUBCASE("leading dimensions don't match") {
      TensorShape rhs = TensorShape{
          TensorDims{
              FFOrdered<positive_int>{
                  5_p,
                  6_p,
                  3_p,
                  8_p,
              },
          },
          DataType::FLOAT,
      };

      CHECK_THROWS(batch_matmul_get_output_shape(attrs, lhs, rhs));
    }
  }

  TEST_CASE("batch_matmul_get_output_parallel_dim_degrees") {
    BatchMatmulAttrs attrs = BatchMatmulAttrs{};

    auto mk_degrees = [](positive_int sum_degree,
                         positive_int discard_copy_degree,
                         positive_int batch_dim_degree,
                         positive_int row_degree,
                         positive_int col_degree) -> ParallelTensorDimDegrees {
      return ParallelTensorDimDegrees{
          /*sum_degree=*/SumDegree{sum_degree},
          /*discard_copy_degree=*/DiscardCopyDegree{discard_copy_degree},
          /*shard_degrees=*/
          FFOrdered<positive_int>{
              batch_dim_degree,
              row_degree,
              col_degree,
          },
      };
    };

    SUBCASE("data parallelism") {
      SUBCASE("degrees match") {
        ParallelTensorDimDegrees result =
            batch_matmul_get_output_parallel_dim_degrees(
                /*attrs=*/attrs,
                /*lhs=*/mk_degrees(1_p, 1_p, 3_p, 1_p, 1_p),
                /*rhs=*/mk_degrees(1_p, 1_p, 3_p, 1_p, 1_p));

        ParallelTensorDimDegrees correct = mk_degrees(1_p, 1_p, 3_p, 1_p, 1_p);

        CHECK(result == correct);
      }

      SUBCASE("degrees don't match") {
        CHECK_THROWS(batch_matmul_get_output_parallel_dim_degrees(
            /*attrs=*/attrs,
            /*lhs=*/mk_degrees(1_p, 1_p, 3_p, 1_p, 1_p),
            /*rhs=*/mk_degrees(1_p, 1_p, 4_p, 1_p, 1_p)));
      }
    }

    SUBCASE("reduction parallelism") {
      SUBCASE("degrees match") {
        ParallelTensorDimDegrees result =
            batch_matmul_get_output_parallel_dim_degrees(
                /*attrs=*/attrs,
                /*lhs=*/mk_degrees(1_p, 1_p, 1_p, 1_p, 5_p),
                /*rhs=*/mk_degrees(1_p, 1_p, 1_p, 5_p, 1_p));

        ParallelTensorDimDegrees correct = mk_degrees(5_p, 1_p, 1_p, 1_p, 1_p);

        CHECK(result == correct);
      }

      SUBCASE("degrees don't match") {
        CHECK_THROWS(batch_matmul_get_output_parallel_dim_degrees(
            /*attrs=*/attrs,
            /*lhs=*/mk_degrees(1_p, 1_p, 1_p, 1_p, 5_p),
            /*rhs=*/mk_degrees(1_p, 1_p, 1_p, 4_p, 1_p)));
      }
    }

    SUBCASE("lhs row parallelism") {
      SUBCASE("degrees match") {
        ParallelTensorDimDegrees result =
            batch_matmul_get_output_parallel_dim_degrees(
                /*attrs=*/attrs,
                /*lhs=*/mk_degrees(1_p, 1_p, 1_p, 2_p, 1_p),
                /*rhs=*/mk_degrees(1_p, 2_p, 1_p, 1_p, 1_p));

        ParallelTensorDimDegrees correct = mk_degrees(1_p, 1_p, 1_p, 2_p, 1_p);

        CHECK(result == correct);
      }

      SUBCASE("degrees don't match") {
        CHECK_THROWS(batch_matmul_get_output_parallel_dim_degrees(
            /*attrs=*/attrs,
            /*lhs=*/mk_degrees(1_p, 1_p, 1_p, 2_p, 1_p),
            /*rhs=*/mk_degrees(1_p, 3_p, 1_p, 1_p, 1_p)));
      }
    }

    SUBCASE("rhs column parallelism") {
      SUBCASE("degrees match") {
        ParallelTensorDimDegrees result =
            batch_matmul_get_output_parallel_dim_degrees(
                /*attrs=*/attrs,
                /*lhs=*/mk_degrees(1_p, 3_p, 1_p, 1_p, 1_p),
                /*rhs=*/mk_degrees(1_p, 1_p, 1_p, 1_p, 3_p));

        ParallelTensorDimDegrees correct = mk_degrees(1_p, 1_p, 1_p, 1_p, 3_p);

        CHECK(result == correct);
      }

      SUBCASE("degrees don't match") {
        CHECK_THROWS(batch_matmul_get_output_parallel_dim_degrees(
            /*attrs=*/attrs,
            /*lhs=*/mk_degrees(1_p, 4_p, 1_p, 1_p, 1_p),
            /*rhs=*/mk_degrees(1_p, 1_p, 1_p, 1_p, 2_p)));
      }
    }

    SUBCASE("pre-existing lhs sum parallelism") {
      SUBCASE("degrees match") {
        ParallelTensorDimDegrees result =
            batch_matmul_get_output_parallel_dim_degrees(
                /*attrs=*/attrs,
                /*lhs=*/mk_degrees(3_p, 1_p, 1_p, 1_p, 1_p),
                /*rhs=*/mk_degrees(1_p, 3_p, 1_p, 1_p, 1_p));

        ParallelTensorDimDegrees correct = mk_degrees(3_p, 1_p, 1_p, 1_p, 1_p);

        CHECK(result == correct);
      }

      SUBCASE("degrees don't match") {
        CHECK_THROWS(batch_matmul_get_output_parallel_dim_degrees(
            /*attrs=*/attrs,
            /*lhs=*/mk_degrees(3_p, 1_p, 1_p, 1_p, 1_p),
            /*rhs=*/mk_degrees(1_p, 2_p, 1_p, 1_p, 1_p)));
      }
    }

    SUBCASE("pre-existing rhs sum parallelism") {
      SUBCASE("degrees match") {
        ParallelTensorDimDegrees result =
            batch_matmul_get_output_parallel_dim_degrees(
                /*attrs=*/attrs,
                /*lhs=*/mk_degrees(1_p, 5_p, 1_p, 1_p, 1_p),
                /*rhs=*/mk_degrees(5_p, 1_p, 1_p, 1_p, 1_p));

        ParallelTensorDimDegrees correct = mk_degrees(5_p, 1_p, 1_p, 1_p, 1_p);

        CHECK(result == correct);
      }

      SUBCASE("degrees don't match") {
        CHECK_THROWS(batch_matmul_get_output_parallel_dim_degrees(
            /*attrs=*/attrs,
            /*lhs=*/mk_degrees(1_p, 4_p, 1_p, 1_p, 1_p),
            /*rhs=*/mk_degrees(5_p, 1_p, 1_p, 1_p, 1_p)));
      }
    }

    SUBCASE("pre-existing lhs and rhs sum parallelism") {
      SUBCASE("degrees match") {
        ParallelTensorDimDegrees result =
            batch_matmul_get_output_parallel_dim_degrees(
                /*attrs=*/attrs,
                /*lhs=*/mk_degrees(3_p, 2_p, 1_p, 1_p, 1_p),
                /*rhs=*/mk_degrees(2_p, 3_p, 1_p, 1_p, 1_p));

        ParallelTensorDimDegrees correct = mk_degrees(6_p, 1_p, 1_p, 1_p, 1_p);

        CHECK(result == correct);
      }

      SUBCASE("degrees don't match") {
        CHECK_THROWS(batch_matmul_get_output_parallel_dim_degrees(
            /*attrs=*/attrs,
            /*lhs=*/mk_degrees(2_p, 3_p, 1_p, 1_p, 1_p),
            /*rhs=*/mk_degrees(2_p, 3_p, 1_p, 1_p, 1_p)));
      }
    }

    SUBCASE("all the degrees at once") {
      ParallelTensorDimDegrees result =
          batch_matmul_get_output_parallel_dim_degrees(
              /*attrs=*/attrs,
              /*lhs=*/mk_degrees(13_p, 11_p * 3_p, 2_p, 7_p, 5_p),
              /*rhs=*/mk_degrees(11_p, 13_p * 7_p, 2_p, 5_p, 3_p));

      ParallelTensorDimDegrees correct =
          mk_degrees(13_p * 11_p * 5_p, 1_p, 2_p, 7_p, 3_p);

      CHECK(result == correct);
    }
  }

  TEST_CASE("batch_matmul_get_output_parallel_shape") {
    // since most of the edge cases are already tested in
    // batch_matmul_get_output_shape and batch_matmul_get_output_parallel_dim_degrees,
    // here we just do a basic check that they compose

    BatchMatmulAttrs attrs = BatchMatmulAttrs{};

    ParallelTensorShape lhs_shape = ParallelTensorShape{
        ParallelTensorDims{
            FFOrdered<ShardParallelDim>{
                ShardParallelDim{4_p, 2_p},
                ShardParallelDim{14_p, 7_p},
                ShardParallelDim{10_p, 5_p},
            },
            ReplicaParallelDimSet{
                SumDegree{13_p},
                DiscardCopyDegree{11_p * 3_p},
            },
        },
        DataType::FLOAT,
    };

    ParallelTensorShape rhs_shape = ParallelTensorShape{
        ParallelTensorDims{
            FFOrdered<ShardParallelDim>{
                ShardParallelDim{4_p, 2_p},
                ShardParallelDim{10_p, 5_p},
                ShardParallelDim{6_p, 3_p},
            },
            ReplicaParallelDimSet{
                SumDegree{11_p},
                DiscardCopyDegree{13_p * 7_p},
            },
        },
        DataType::FLOAT,
    };

    ParallelTensorShape result =
        batch_matmul_get_output_parallel_shape(attrs, lhs_shape, rhs_shape);

    ParallelTensorShape correct = ParallelTensorShape{
        ParallelTensorDims{
            FFOrdered<ShardParallelDim>{
                ShardParallelDim{4_p, 2_p},
                ShardParallelDim{14_p, 7_p},
                ShardParallelDim{6_p, 3_p},
            },
            ReplicaParallelDimSet{
                SumDegree{13_p * 11_p * 5_p},
                DiscardCopyDegree{1_p},
            },
        },
        DataType::FLOAT,
    };

    CHECK(result == correct);
  }

  TEST_CASE("batch_matmul_get_operator_to_lhs_input_mapping") {
    BatchMatmulAttrs attrs = BatchMatmulAttrs{};

    auto mk_degrees = [](positive_int sum_degree,
                         positive_int discard_copy_degree,
                         positive_int batch_dim_degree,
                         positive_int row_degree,
                         positive_int col_degree) -> ParallelTensorDimDegrees {
      return ParallelTensorDimDegrees{
          /*sum_degree=*/SumDegree{sum_degree},
          /*discard_copy_degree=*/DiscardCopyDegree{discard_copy_degree},
          /*shard_degrees=*/
          FFOrdered<positive_int>{
              batch_dim_degree,
              row_degree,
              col_degree,
          },
      };
    };

    OperatorSpaceToParallelTensorSpaceBiuniqueMapping result =
        batch_matmul_get_operator_to_lhs_input_mapping(
            /*attrs=*/attrs,
            /*lhs=*/mk_degrees(13_p, 11_p * 3_p, 2_p, 7_p, 5_p),
            /*rhs=*/mk_degrees(11_p, 13_p * 7_p, 2_p, 5_p, 3_p));

    // for now just check that it doesn't crash
  }

  TEST_CASE("batch_matmul_get_parallel_task_signatures") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    auto mk_degrees = [](positive_int sum_degree,
                         positive_int discard_copy_degree,
                         positive_int batch_dim_degree,
                         positive_int row_degree,
                         positive_int col_degree) -> ParallelTensorDimDegrees {
      return ParallelTensorDimDegrees{
          /*sum_degree=*/SumDegree{sum_degree},
          /*discard_copy_degree=*/DiscardCopyDegree{discard_copy_degree},
          /*shard_degrees=*/
          FFOrdered<positive_int>{
              batch_dim_degree,
              row_degree,
              col_degree,
          },
      };
    };

    TensorShape lhs_shard_shape = TensorShape{
      /*dims=*/TensorDims{
        FFOrdered<positive_int>{
          4_p,
          2_p,
          3_p,
        },
      }
      /*data_type=*/DataType::FLOAT,
    };

    TensorShape rhs_shard_shape = TensorShape{
      /*dims=*/TensorDims{
        FFOrdered<positive_int>{
          4_p,
          3_p,
          1_p,
        },
      },
      /*data_type=*/DataType::FLOAT,
    };

    BatchMatmulAttrs attrs = BatchMatmulAttrs{};

    // TODO(@lockshaw)(#pr): more of this should be factored out as it will be shared between operators
    auto run_test_for_degrees = [&](ParallelTensorDimDegrees const &lhs_degrees, rhs_degrees) {
      std::set<OperatorAtomicTaskShardBinding> bindings = 
          batch_matmul_get_parallel_task_signatures(
            attrs, 
            lhs_degrees,
            rhs_degrees);

      ParallelTensorShape lhs_shape = lift_to_parallel_with_degrees(lhs_shard_shape, lhs_degrees);
      ParallelTensorShape rhs_shape = lift_to_parallel_with_degrees(rhs_shard_shape, rhs_degrees);

      EmulatedParallelTensor lhs_input = random_emulated_parallel_tensor_of_shape(lhs_shape, 1);
      EmulatedParallelTensor rhs_input = random_emulated_parallel_tensor_of_shape(rhs_shape, 2);

      std::map<TensorSlotName, EmulatedParallelTensor> inputs = {
        {
          TensorSlotName::LHS_INPUT,
          lhs_input,
        }
        {
          TensorSlotName::RHS_INPUT,
          rhs_input,
        }
      };

      auto run_op = [&](std::map<TensorSlotName, GenericTensorAccessorR> const &input_shards) 
        -> std::map<TensorSlotName, GenericTensorAccessorR>
      {
        GenericTensorAccessorR lhs_input_shard = input_shards.at(TensorSlotName::LHS_INPUT);
        GenericTensorAccessorR rhs_input_shard = input_shards.at(TensorSlotName::RHS_INPUT);
        TensorShape output_shard_shape = 
          batch_matmul_get_output_shape(attrs,
                                        lhs_input_shape.shape,
                                        rhs_input_shape.shape);
        GenericTensorAccessorW output_shard = create_zero_filled_accessor_w(output_shard_shape, cpu_allocator);

        batch_matmul_cpu_forward_kernel(
          /*input_lhs=*/lhs_input_shard,
          /*input_rhs=*/rhs_input_shard,
          /*output=*/output_shard,
        );

        return output_shard;
      };

      auto unparallelize = [&](std::map<TensorSlotName, EmulatedParallelTensor> const &parallel_tensors) 
        -> std::map<TensorSlotName, GenericTensorAccessorR>
      {
        return map_values(
          parallel_tensors,
          [&](EmulatedParallelTensor const &ptensor) -> GenericTensorAccessorR
          {
            return unparallelize_parallel_tensor(ptensor, cpu_allocator);
          });
      };

      GenericTensorAccessorR op_then_unpar = 
        require_only_key(
          unparallelize(
            parallelize_tensor_operation(
              inputs,
              bindings,
              run_op)),
          TensorSlotName::OUTPUT);

      GenericTensorAccessorR unpar_then_op = 
        require_only_key(
          run_op(unparallelize(inputs)),
          TensorSlotName::OUTPUT):

      CHECK_MESSAGE(
          accessors_are_equal(op_then_unpar, unpar_then_op),
          check_kv("op_then_unpar", format_accessor_r_contents(op_then_unpar),
          check_kv("unpar_then_op", format_accessor_r_contents(unpar_then_op));
    };

    ParallelTensorDimDegrees lhs_degrees = mk_degrees(1, 1, 2, 1, 1);
    ParallelTensorDimDegrees rhs_degrees = mk_degrees(1, 1, 2, 1, 1);

    run_test_for_degrees(lhs_degrees, rhs_degrees);
  }
}
