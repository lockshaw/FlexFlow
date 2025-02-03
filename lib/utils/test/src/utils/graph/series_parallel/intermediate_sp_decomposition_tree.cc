#include "utils/graph/series_parallel/intermediate_sp_decomposition_tree.h"
#include "utils/fmt/variant.h"
#include <catch2/catch_test_macros.hpp>

using namespace ::FlexFlow;


  TEST_CASE("flatten_ast") {
    std::variant<IntermediateSpDecompositionTree, Node> input =
        IntermediateSpDecompositionTree{
            SplitType::SERIES,
            {
                Node{1},
                IntermediateSpDecompositionTree{
                    SplitType::SERIES,
                    {
                        Node{2},
                        Node{3},
                    },
                },
            },
        };

    std::variant<IntermediateSpDecompositionTree, Node> result =
        flatten_ast(input);
    std::variant<IntermediateSpDecompositionTree, Node> correct =
        IntermediateSpDecompositionTree{
            SplitType::SERIES,
            {
                Node{1},
                Node{2},
                Node{3},
            },
        };

    CHECK(result == correct);
  }
