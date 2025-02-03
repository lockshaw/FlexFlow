include(aliasing)

if(FF_USE_EXTERNAL_CATCH2)
  find_package(Catch2 3 REQUIRED)
else()
  message(FATAL_ERROR "Currently FF_USE_EXTERNAL_CATCH2 is required")
endif()

# target_compile_definitions(
#   Catch2::Catch2
#   INTERFACE 
#     CATCH_CONFIG_FALLBACK_STRINGIFIER="fallbackStringifier"
#     CATCH_CONFIG_ENABLE_ALL_STRINGMAKERS
# )

# target_compile_definitions(
#   Catch2::Catch2WithMain
#   INTERFACE 
#     CATCH_CONFIG_FALLBACK_STRINGIFIER="fallbackStringifier"
#     CATCH_CONFIG_ENABLE_ALL_STRINGMAKERS
# )

include(CTest)
include(Catch)

alias_library(catch2 Catch2::Catch2)
alias_library(catch2-main Catch2::Catch2WithMain)
