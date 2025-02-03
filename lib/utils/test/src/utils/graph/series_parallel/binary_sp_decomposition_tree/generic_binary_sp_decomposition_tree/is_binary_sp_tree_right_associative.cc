#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/is_binary_sp_tree_right_associative.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/binary_sp_decomposition_tree.dtg.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/binary_sp_decomposition_tree.h"
#include <catch2/catch_test_macros.hpp>

using namespace ::FlexFlow;


  TEST_CASE("is_binary_sp_tree_right_associative") {
    Node n1 = Node{1};
    Node n2 = Node{2};
    Node n3 = Node{3};
    Node n4 = Node{4};

    GenericBinarySPDecompositionTreeImplementation<BinarySPDecompositionTree,
                                                   BinarySeriesSplit,
                                                   BinaryParallelSplit,
                                                   Node>
        impl = generic_impl_for_binary_sp_tree();

    auto make_series_split = [](BinarySPDecompositionTree const &lhs,
                                BinarySPDecompositionTree const &rhs) {
      return BinarySPDecompositionTree{BinarySeriesSplit{lhs, rhs}};
    };

    auto make_parallel_split = [](BinarySPDecompositionTree const &lhs,
                                  BinarySPDecompositionTree const &rhs) {
      return BinarySPDecompositionTree{BinaryParallelSplit{lhs, rhs}};
    };

    auto make_leaf = [](Node const &n) { return BinarySPDecompositionTree{n}; };

    SECTION("input is actually right associative") {
      SECTION("just node") {
        BinarySPDecompositionTree input = make_leaf(n1);

        bool result = is_binary_sp_tree_right_associative(input);
        bool correct = true;

        CHECK(result == correct);
      }

      SECTION("just series") {
        BinarySPDecompositionTree input = make_series_split(
            make_leaf(n1), make_series_split(make_leaf(n2), make_leaf(n3)));

        bool result = is_binary_sp_tree_right_associative(input);
        bool correct = true;

        CHECK(result == correct);
      }

      SECTION("just parallel") {
        BinarySPDecompositionTree input = make_parallel_split(
            make_leaf(n1), make_parallel_split(make_leaf(n2), make_leaf(n3)));

        bool result = is_binary_sp_tree_right_associative(input);
        bool correct = true;

        CHECK(result == correct);
      }

      SECTION("nested") {
        BinarySPDecompositionTree input = make_series_split(
            make_parallel_split(make_leaf(n1), make_leaf(n2)),
            make_parallel_split(make_leaf(n3), make_leaf(n4)));

        bool result = is_binary_sp_tree_right_associative(input);
        bool correct = true;

        CHECK(result == correct);
      }
    }

    SECTION("input is not right associative") {
      SECTION("just series") {
        BinarySPDecompositionTree input = make_series_split(
            make_series_split(make_leaf(n1), make_leaf(n2)), make_leaf(n3));

        bool result = is_binary_sp_tree_right_associative(input);
        bool correct = false;

        CHECK(result == correct);
      }

      SECTION("just parallel") {
        BinarySPDecompositionTree input = make_parallel_split(
            make_parallel_split(make_leaf(n1), make_leaf(n2)), make_leaf(n3));

        bool result = is_binary_sp_tree_right_associative(input);
        bool correct = false;

        CHECK(result == correct);
      }
    }
  }
