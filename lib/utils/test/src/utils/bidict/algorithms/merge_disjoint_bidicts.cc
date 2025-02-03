#include "utils/bidict/algorithms/merge_disjoint_bidicts.h"
#include <catch2/catch_test_macros.hpp>

using namespace ::FlexFlow;



  TEST_CASE("merge_disjoint_bidicts") {

    SECTION("disjoint keys and values") {
      bidict<int, std::string> bd1 = {{1, "one"}, {2, "two"}};
      bidict<int, std::string> bd2 = {{3, "three"}, {4, "four"}};

      bidict<int, std::string> result = merge_disjoint_bidicts(bd1, bd2);
      bidict<int, std::string> correct = {
          {1, "one"}, {2, "two"}, {3, "three"}, {4, "four"}};

      CHECK(result == correct);
    }

    SECTION("overlapping key, different associated value") {
      bidict<int, std::string> bd1 = {{1, "one"}, {2, "two"}};
      bidict<int, std::string> bd2 = {{2, "three"}, {3, "four"}};

      CHECK_THROWS(merge_disjoint_bidicts(bd1, bd2));
    }

    SECTION("overlapping key, same associated value") {
      bidict<int, std::string> bd1 = {{1, "one"}, {2, "two"}};
      bidict<int, std::string> bd2 = {{2, "two"}, {3, "three"}};

      CHECK_THROWS(merge_disjoint_bidicts(bd1, bd2));
    }

    SECTION("overlapping values") {
      bidict<int, std::string> bd1 = {{1, "one"}, {2, "two"}};
      bidict<int, std::string> bd2 = {{3, "two"}, {4, "four"}};

      CHECK_THROWS(merge_disjoint_bidicts(bd1, bd2));
    }
  }
