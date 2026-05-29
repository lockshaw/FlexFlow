#include <doctest/doctest.h>
#include "utils/relation/hemiunique_binary_relation.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("HemiuniqueBinaryRelation") {
    SUBCASE("normalizes empty relation") {
      HemiuniqueBinaryRelation<int, std::string> r1
        = HemiuniqueBinaryRelation<int, std::string>{
          bidict<int, std::string>{},
        };

      HemiuniqueBinaryRelation<int, std::string> r2
        = HemiuniqueBinaryRelation<int, std::string>{
          OneToMany<int, std::string>{},
        };

      HemiuniqueBinaryRelation<int, std::string> r3
        = HemiuniqueBinaryRelation<int, std::string>{
          ManyToOne<int, std::string>{},
        };

      CHECK(r1 == r2);
      CHECK(r2 == r3);
      CHECK(r3 == r1);
    }

    SUBCASE("normalizes biunique relations") {
      HemiuniqueBinaryRelation<int, std::string> r1
        = HemiuniqueBinaryRelation<int, std::string>{
          bidict<int, std::string>{
            {1, "one"},
            {2, "two"},
            {4, "four"},
          },
        };

      HemiuniqueBinaryRelation<int, std::string> r2
        = HemiuniqueBinaryRelation<int, std::string>{
          OneToMany<int, std::string>{
            {1, {"one"}},
            {2, {"two"}},
            {4, {"four"}},
          },
        };

      HemiuniqueBinaryRelation<int, std::string> r3
        = HemiuniqueBinaryRelation<int, std::string>{
          ManyToOne<int, std::string>{
            {{1}, "one"},
            {{2}, "two"},
            {{4}, "four"},
          },
        };

      CHECK(r1 == r2);
      CHECK(r2 == r3);
      CHECK(r3 == r1);
    }
  }
}
