ff_add_test_executable(
  NAME
    utils-tests
  SRC_PATTERNS
    src/utils/*.cc
  PRIVATE_INCLUDE
    src/
  DEPS
    utils
    catch2-main
    utils-test-common
)
# target_compile_definitions(
#   utils-tests
#   PRIVATE
#     CATCH_CONFIG_FALLBACK_STRINGIFIER=fmt::to_string
# )

add_subdirectory(common)
