#include "doctest/doctest.h"
#include "utils/fmt/expected.h"
#include <fmt/format.h>
#include <sstream>
#include <tl/expected.hpp>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define CHECK_WITHOUT_STRINGIFY(...)                                           \
  do {                                                                         \
    bool result = __VA_ARGS__;                                                 \
    CHECK(result);                                                             \
  } while (0);
