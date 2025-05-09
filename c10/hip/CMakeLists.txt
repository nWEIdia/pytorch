# Build file for the C10 HIP.  This directory doesn't actually contain
# any files; these files are copied over by the HIPIFY script in tools/amd_build
#
# At the moment, this cmake is NOT standalone

include(../../cmake/public/utils.cmake)

if(BUILD_LIBTORCHLESS)
  find_library(C10_HIP_LIB c10_hip PATHS $ENV{LIBTORCH_LIB_PATH} NO_DEFAULT_PATH)
endif()

# ---[ Configure macro file.
set(C10_HIP_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS}) # used in cmake_macros.h.in
configure_file(
    ${CMAKE_CURRENT_LIST_DIR}/impl/hip_cmake_macros.h.in
    ${CMAKE_BINARY_DIR}/c10/hip/impl/hip_cmake_macros.h)

# NB: All previous cu files are renamed into cc files.  This isn't tested at the
# moment.
file(GLOB C10_HIP_SRCS
        *.cpp
        *.cc
        impl/*.cpp
        impl/*.cc
        )

# Mark the cc files as HIP files, so we call the compiler.  (They have to be
# suffixed with cc, because the hcc compiler won't accept them otherwise.)
file(GLOB __c10_hip_srcs_cpp *.cc impl/*.cc)
set_source_files_properties(${__c10_hip_srcs_cpp} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)

file(GLOB_RECURSE C10_HIP_HEADERS *.h)

if(NOT BUILD_LIBTORCHLESS)
  hip_add_library(c10_hip ${C10_HIP_SRCS} ${C10_HIP_HEADERS})
  torch_compile_options(c10_hip)

  # Propagate HIP_CXX_FLAGS that were set from Dependencies.cmake
  target_compile_options(c10_hip PRIVATE ${HIP_CXX_FLAGS})

  # caffe2_hip adds a bunch of dependencies like rocsparse, but c10/hip is supposed to be
  # minimal.  I'm not sure if we need hip_hcc or not; for now leave it out

  # If building shared library, set dllimport/dllexport proper.
  target_compile_options(c10_hip PRIVATE "-DC10_HIP_BUILD_MAIN_LIB")
  # Enable hidden visibility if compiler supports it.
  if(${COMPILER_SUPPORTS_HIDDEN_VISIBILITY})
    target_compile_options(c10_hip PRIVATE "-fvisibility=hidden")
  endif()

  # ---[ Dependency of c10_hip
  target_link_libraries(c10_hip PUBLIC ${C10_LIB} hip::amdhip64)

  target_include_directories(
      c10_hip PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
      $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
      $<INSTALL_INTERFACE:include>)
  install(TARGETS c10_hip EXPORT Caffe2Targets DESTINATION lib)
  set(C10_HIP_LIB c10_hip)
endif()

add_subdirectory(test)

# ---[ Installation
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        DESTINATION include
        FILES_MATCHING PATTERN "*.h")
install(FILES ${CMAKE_BINARY_DIR}/c10/hip/impl/hip_cmake_macros.h
  DESTINATION include/c10/hip/impl)
