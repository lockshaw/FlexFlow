#include "utils/not_implemented.h"
#include <fmt/format.h>

namespace FlexFlow {

not_implemented::not_implemented(std::string const &function_name,
                                 std::string const &file_name,
                                 int line)
    : std::logic_error(fmt::format("Function '{}' not yet implemented at {}:{}",
                                   function_name,
                                   file_name,
                                   line)){};

} // namespace FlexFlow
