if (FF_USE_EXTERNAL_RAPIDCHECK)
  find_package(rapidcheck REQUIRED)
else()
  include(FetchContent)

  FetchContent_Declare(
    rapidcheckdl
    URL https://github.com/emil-e/rapidcheck/archive/ff6af6fc683159deb51c543b065eba14dfcf329b.zip
    DOWNLOAD_EXTRACT_TIMESTAMP ON
  )
  FetchContent_MakeAvailable(rapidcheckdl)
endif()
