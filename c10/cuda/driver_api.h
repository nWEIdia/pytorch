// clang-format off
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// clang-format on
#pragma once

#include <cuda.h>

// Lazy loading system for CUDA driver APIs in PyTorch
//
// This system allows PyTorch to use CUDA driver APIs without directly linking
// against them, providing better compatibility across different CUDA versions.
// 
// Usage: Just call the driver API functions normally (e.g., cuGetErrorName).
// The first call will lazily load the function, subsequent calls go directly
// to the driver API with zero overhead.

namespace at {
namespace cuda {
namespace driver {

#define DECLARE_DRIVER_API_WRAPPER(funcName, version) \
  extern decltype(::funcName)* funcName

// List of driver APIs with their minimum required CUDA versions.
// For maximum compatibility, versions should be as low as possible
// while supporting required capabilities.
//
// PyTorch supports CUDA_VERSION >= 11000
#define ALL_DRIVER_API_WRAPPER_CUDA(fn) \
  fn(cuDeviceGetAttribute, 11000);      \
  fn(cuDeviceGetName, 11000);           \
  fn(cuDriverGetVersion, 11000);        \
  fn(cuFuncGetAttribute, 11000);        \
  fn(cuFuncSetAttribute, 11000);        \
  fn(cuGetErrorName, 11000);            \
  fn(cuGetErrorString, 11000);          \
  fn(cuInit, 11000);                    \
  fn(cuLaunchCooperativeKernel, 11000); \
  fn(cuLaunchKernel, 11000);            \
  fn(cuModuleGetFunction, 11000);       \
  fn(cuModuleLoadDataEx, 11000);        \
  fn(cuModuleUnload, 11000);            \
  fn(cuMemGetAddressRange, 11000);      \
  fn(cuMemAlloc, 11000);                \
  fn(cuMemFree, 11000);                 \
  fn(cuMemcpyDtoH, 11000);              \
  fn(cuMemcpyHtoD, 11000);              \
  fn(cuMemcpyDtoD, 11000);              \
  fn(cuOccupancyMaxActiveBlocksPerMultiprocessor, 11000); \
  fn(cuStreamCreate, 11000);            \
  fn(cuStreamDestroy, 11000);           \
  fn(cuStreamSynchronize, 11000);       \
  fn(cuCtxGetCurrent, 11000);           \
  fn(cuCtxSetCurrent, 11000)

// Stream memory operations handling for different CUDA versions
// CUDA 12+ integrates v2 APIs into vanilla APIs and removes the
// NVreg_EnableStreamMemOPs=1 requirement
#if (CUDA_VERSION >= 12000)
#define ALL_DRIVER_API_WRAPPER(fn) \
  ALL_DRIVER_API_WRAPPER_CUDA(fn); \
  fn(cuStreamWaitValue32, 12000);  \
  fn(cuStreamWriteValue32, 12000); \
  fn(cuTensorMapEncodeTiled, 12000); \
  fn(cuTensorMapReplaceAddress, 12000)
#elif (CUDA_VERSION >= 11000)
#define ALL_DRIVER_API_WRAPPER(fn) \
  ALL_DRIVER_API_WRAPPER_CUDA(fn); \
  fn(cuStreamWaitValue32, 11000);  \
  fn(cuStreamWriteValue32, 11000)
#else
#error "CUDA_VERSION < 11000 isn't supported by PyTorch."
#endif

ALL_DRIVER_API_WRAPPER(DECLARE_DRIVER_API_WRAPPER);

#undef DECLARE_DRIVER_API_WRAPPER

// Utility functions for driver API management
namespace detail {
  // Initialize the driver API system (call once during PyTorch initialization)
  void initializeDriverAPI();
  
  // Check if a specific driver API is available at runtime
  bool isDriverAPIAvailable(const char* funcName, unsigned int minVersion);
  
  // Get the loaded CUDA driver version
  int getDriverVersion();
}

} // namespace driver
} // namespace cuda
} // namespace at 