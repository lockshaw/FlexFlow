#include "./random_set.h"

namespace FlexFlow {

std::unordered_set<int> random_set(nonnegative_int size) {
  srand(0);
  std::unordered_set<int> result;
  while (result.size() < size.unwrap_nonnegative()) {
    result.insert(std::rand());
  }
  return result;
};

}
