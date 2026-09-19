#include <doctest/doctest.h>
#include "utils/containers/multiset_intersection.h"
#include "test/utils/doctest/fmt/multiset.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("multiset_intersection") {
    std::multiset<int> lhs = {1, 1, 2, 2, 3, 3, 3, 4};
    std::multiset<int> rhs = {1, 2, 2, 4, 4, 5};

    std::multiset<int> result = multiset_intersection(lhs, rhs);
    std::multiset<int> correct = {1, 2, 2, 4};

    CHECK(result == correct);
  }
}
