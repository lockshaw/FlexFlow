#include <doctest/doctest.h>
#include "utils/one_to_many/one_to_many_is_biunique.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("one_to_many_is_biunique") {
    SUBCASE("OneToMany is empty") {
      OneToMany<int, std::string> otm;

      bool result = one_to_many_is_biunique(otm);
      bool correct = true;

      CHECK(result == correct);
    }

    SUBCASE("OneToMany is nonempty and biunique") {
      OneToMany<int, std::string> otm;
      otm.insert(std::pair<int, std::string>{1, "one"});
      otm.insert(std::pair<int, std::string>{2, "two"});
      otm.insert(std::pair<int, std::string>{4, "four"});

      bool result = one_to_many_is_biunique(otm);
      bool correct = true;

      CHECK(result == correct);
    }

    SUBCASE("OneToMany is nonempty and not biunique") {
      OneToMany<int, std::string> otm;
      otm.insert(std::pair<int, std::string>{1, "one"});
      otm.insert(std::pair<int, std::string>{2, "two"});
      otm.insert(std::pair<int, std::string>{2, "TWO"});
      otm.insert(std::pair<int, std::string>{4, "four"});

      bool result = one_to_many_is_biunique(otm);
      bool correct = false;

      CHECK(result == correct);
    }
  }
}
