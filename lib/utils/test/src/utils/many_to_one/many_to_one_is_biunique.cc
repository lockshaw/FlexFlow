#include <doctest/doctest.h>
#include "utils/many_to_one/many_to_one_is_biunique.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("many_to_one_is_biunique") {
    SUBCASE("ManyToOne is empty") {
      ManyToOne<int, std::string> mto;

      bool correct = many_to_one_is_biunique(mto);
      bool result = true;

      CHECK(result == correct);
    }

    SUBCASE("ManyToOne is nonempty and biunique") {
      ManyToOne<int, std::string> mto;
      mto.insert({1, "a"});
      mto.insert({2, "aa"});
      mto.insert({4, "aaaaa"});

      bool correct = many_to_one_is_biunique(mto);
      bool result = true;

      CHECK(result == correct);
    }

    SUBCASE("ManyToOne is nonempty and not biunique") {
      ManyToOne<int, std::string> mto;
      mto.insert({10, "1"});
      mto.insert({2, "2"});
      mto.insert({20, "2"});
      mto.insert({4, "4"});

      bool correct = many_to_one_is_biunique(mto);
      bool result = false;

      CHECK(result == correct);
    }
  }
}
