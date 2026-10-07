#include "utils/benchmark_utils/random_hex_string.h"
#include "utils/random_utils.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

std::string random_hex_string(std::mt19937 &gen, nonnegative_int size) {
  std::string result;

  std::vector<char> hex_chars = {
      '0',
      '1',
      '2',
      '3',
      '4',
      '5',
      '6',
      '7',
      '8',
      '9',
      'a',
      'b',
      'c',
      'd',
      'e',
      'f',
  };

  for (int i = 0; i < size; i++) {
    result.push_back(select_random(gen, hex_chars));
  }

  ASSERT(result.size() == size);

  return result;
}

} // namespace FlexFlow
