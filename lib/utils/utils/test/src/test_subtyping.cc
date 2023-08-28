#include "test/utils/doctest.h"
#include "utils/subtyping.h"

using namespace FlexFlow;

struct GreatGrandparent {};
struct Grandparent {
  operator GreatGrandparent() const {
    return GreatGrandparent{};
  }
};

template <typename T>
struct Parent { 
  operator Grandparent() const {
    return Grandparent{};
  }
};

template <typename T>
struct Child {
  operator Parent<T>() const {
    return Parent<T>{};
  }
};
template <typename T, typename TT>
struct Grandchild {
  operator Child<T>() const {
    return Child<T>{};
  }
};

MAKE_SUBTYPING_FOREST(test_forest);
MAKE_SUBTYPING_FOREST_ROOT(test_forest, GreatGrandparent);
MAKE_SUBTYPING_RELATION(test_forest, Grandparent, GreatGrandparent);
MAKE_SUBTYPING_RELATION(test_forest, Parent, 1, Grandparent);
MAKE_SUBTYPING_RELATION(test_forest, Child, 1, Parent, 1);
MAKE_SUBTYPING_RELATION(test_forest, Grandchild, 2, Child, 1);

enum which_called_t { PARENT_INT, PARENT_FLOAT, CHILD_INT, CHILD_FLOAT, GRANDCHILD_INT_INT, GRANDCHILD_INT_FLOAT, GRANDCHILD_FLOAT_INT, GRANDCHILD_FLOAT_FLOAT };

template <typename T>
which_called_t test_coerce_to_self(T const &t, test_forest<Parent<int>>) {
  return PARENT_INT;
}

template <typename T>
which_called_t test_coerce_to_self(T const &t, test_forest<Parent<float>>) {
  return PARENT_FLOAT;
}

template <typename T>
which_called_t test_coerce_to_self(T const &t, test_forest<Child<int>>) {
  return CHILD_INT;
}

template <typename T>
which_called_t test_coerce_to_self(T const &t, test_forest<Child<float>>) {
  return CHILD_FLOAT;
}

template <typename T>
which_called_t test_coerce_to_self(T const &t, test_forest<Grandchild<int, int>>) {
  return GRANDCHILD_INT_INT;
}

template <typename T>
which_called_t test_coerce_to_self(T const &t, test_forest<Grandchild<int, float>>) {
  return GRANDCHILD_INT_FLOAT;
}

template <typename T>
which_called_t test_coerce_to_self(T const &t, test_forest<Grandchild<float, int>>) {
  return GRANDCHILD_FLOAT_INT;
}

template <typename T>
which_called_t test_coerce_to_self(T const &t, test_forest<Grandchild<float, float>>) {
  return GRANDCHILD_FLOAT_FLOAT;
}

template <typename T>
which_called_t test_coerce_up_one_level(T const &t, test_forest<Parent<int>>) {
  return PARENT_INT;
}

template <typename T>
which_called_t test_coerce_up_one_level(T const &t, test_forest<Parent<float>>) {
  return PARENT_FLOAT;
}

template <typename T>
which_called_t test_coerce_up_one_level(T const &t, test_forest<Child<int>>) {
  return CHILD_INT;
}

template <typename T>
which_called_t test_coerce_up_one_level(T const &t, test_forest<Child<float>>) {
  return CHILD_FLOAT;
}

template <typename T>
which_called_t test_coerce_up_two_levels(T const &t, test_forest<Parent<int>>) {
  return PARENT_INT;
}

template <typename T>
which_called_t test_coerce_up_two_levels(T const &t, test_forest<Parent<float>>) {
  return PARENT_FLOAT;
}

template <typename T>
which_called_t test_coerce_to_self(T const &t) {
  return test_coerce_to_self(t, create_tag(t));
}

template <typename T>
which_called_t test_coerce_up_one_level(T const &t) {
  return test_coerce_up_one_level(t, create_tag(t));
}

template <typename T>
which_called_t test_coerce_up_two_levels(T const &t) {
  return test_coerce_up_two_levels(t, create_tag(t));
}

TEST_CASE("subtyping - coerce") {
  Grandchild<int, int> g_int_int; 
  Grandchild<int, float> g_int_float;
  Grandchild<float, int> g_float_int;
  Grandchild<float, float> g_float_float;

  CHECK(test_coerce_to_self(g_int_int) == GRANDCHILD_INT_INT);
  CHECK(test_coerce_to_self(g_int_float) == GRANDCHILD_INT_FLOAT);
  CHECK(test_coerce_to_self(g_float_int) == GRANDCHILD_FLOAT_INT);
  CHECK(test_coerce_to_self(g_float_float) == GRANDCHILD_FLOAT_FLOAT);

  CHECK(test_coerce_up_one_level(g_int_int) == CHILD_INT);
  CHECK(test_coerce_up_one_level(g_int_float) == CHILD_INT);
  CHECK(test_coerce_up_one_level(g_float_int) == CHILD_FLOAT);
  CHECK(test_coerce_up_one_level(g_float_float) == CHILD_FLOAT);

  CHECK(test_coerce_up_two_levels(g_int_int) == PARENT_INT);
  CHECK(test_coerce_up_two_levels(g_int_float) == PARENT_INT);
  CHECK(test_coerce_up_two_levels(g_float_int) == PARENT_FLOAT);
  CHECK(test_coerce_up_two_levels(g_float_float) == PARENT_FLOAT);

  CHECK(test_coerce_to_self(coerce<test_forest<Child<int>>>(g_int_int)) == CHILD_INT);
  CHECK(test_coerce_to_self(coerce<test_forest<Child<int>>>(g_int_float)) == CHILD_INT);
  CHECK(test_coerce_to_self(coerce<test_forest<Child<float>>>(g_float_int)) == CHILD_FLOAT);
  CHECK(test_coerce_to_self(coerce<test_forest<Child<float>>>(g_float_float)) == CHILD_FLOAT);

  CHECK(test_coerce_to_self(coerce<test_forest<Parent<int>>>(g_int_int)) == PARENT_INT);
  CHECK(test_coerce_to_self(coerce<test_forest<Parent<int>>>(g_int_float)) == PARENT_INT);
  CHECK(test_coerce_to_self(coerce<test_forest<Parent<float>>>(g_float_int)) == PARENT_FLOAT);
  CHECK(test_coerce_to_self(coerce<test_forest<Parent<float>>>(g_float_float)) == PARENT_FLOAT);
}
