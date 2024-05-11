set(project_target kernels)

# find_package(rccl)

project(${project_target} 
	LANGUAGES CXX)

find_package(hip REQUIRED)
find_package(miopen REQUIRED)

file(GLOB_RECURSE SRC
     CONFIGURE_DEPENDS
     LIST_DIRECTORIES False
     src/*.cc
     src/hip/concat_kernels.cpp
     )

add_library(
  ${project_target}
  SHARED
  ${SRC}
)
target_include_directories(
  ${project_target}
  PRIVATE
    src/
  PUBLIC
    include/
)
target_link_libraries(
  ${project_target}
  op-attrs
  # cuda
  miopen
  hip::host
  rccl
)

define_ff_vars(${project_target})

set_target_properties(
  ${project_target} 
  PROPERTIES 
  HIP_STANDARD 17
)
