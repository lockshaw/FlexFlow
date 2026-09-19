#include <doctest/doctest.h>
#include "utils/containers/prime_factorization.h"
#include "test/utils/doctest/fmt/multiset.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("prime_factorization") {
    std::multiset<int_ge_two> result = prime_factorization(13_ge2 * 2_ge2 * 2_ge2 * 3_ge2);
    std::multiset<int_ge_two> correct = {
      13_ge2,
      2_ge2,
      2_ge2,
      3_ge2,
    };

    CHECK(result == correct);
  }
}
