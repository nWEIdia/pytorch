// clang-format off
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// clang-format on
#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
#include <c10/cuda/driver_api.h>
#include <iostream>
#include <mutex>
#include <unordered_map>

#include <cuda.h>
#include <cuda_runtime.h>

// PyTorch includes (adjust paths based on actual PyTorch structure)
#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>


// How does the lazy loading magic work?
//
// This is adapted from nvFuser's implementation with PyTorch-specific improvements:
//
// 1. Name Shadowing: When you write cuGetErrorName() in PyTorch code within the
//    at::cuda::driver namespace, you're actually calling at::cuda::driver::cuGetErrorName,
//    not the global CUDA driver API, due to C++ name lookup rules.
//
// 2. Function Pointer Swapping: Each driver API function starts as a pointer to a
//    lazilyLoadAndInvoke function. On first call:
//    - Loads the real CUDA driver API using cudaGetDriverEntryPoint
//    - Replaces the function pointer with the real driver API
//    - Calls the real function
//    - Subsequent calls go directly to the real driver API (zero overhead)
//
// 3. Template Magic: Uses C++ templates with CTAD (Class Template Argument Deduction)
//    to automatically handle different function signatures generically.
//
// 4. PyTorch Integration: Added proper error handling, thread safety, and utility
//    functions for PyTorch's needs.

namespace {

// Thread-safe driver API loading
std::mutex driver_api_mutex;
std::unordered_map<std::string, void*> loaded_apis;
int cached_driver_version = -1;

void getDriverEntryPoint(
    const char* symbol,
    unsigned int version,
    void** entry_point) {
  try {
#if (CUDA_VERSION >= 12050)
    C10_CUDA_CHECK(cudaGetDriverEntryPointByVersion(
        symbol, entry_point, version, cudaEnableDefault));
#else
    (void)version; // Suppress unused parameter warning
    C10_CUDA_CHECK(
        cudaGetDriverEntryPoint(symbol, entry_point, cudaEnableDefault));
#endif
  } catch (const c10::Error& e) {
    TORCH_WARN("Failed to load CUDA driver API '", symbol, "': ", e.what());
    *entry_point = nullptr;
  }
}

// Enhanced loader with thread safety and error handling
template <typename ReturnType, typename... Args>
struct DriverAPILoader {
  static ReturnType lazilyLoadAndInvoke(Args... args) {
    // Thread-safe static initialization (C++11 guarantees this)
    static auto* entry_point = []() {
      std::lock_guard<std::mutex> lock(driver_api_mutex);
      
      decltype(::ReturnType(*)(Args...))* func_ptr = nullptr;
      
      // You'll need to pass the symbol name and version somehow
      // This is a limitation of the template approach - we need the function name
      // For now, we'll use a runtime error, but in practice you'd need to 
      // specialize this for each function or use a different approach
      
      TORCH_CHECK(false, "Direct template instantiation not supported. Use DEFINE_DRIVER_API_WRAPPER macro.");
      return func_ptr;
    }();
    
    TORCH_CHECK(entry_point != nullptr, "CUDA driver API function not available");
    return entry_point(args...);
  }
  
  // CTAD helper constructor
  DriverAPILoader(ReturnType(Args...)) {}
};

// CTAD deduction guide
template <typename ReturnType, typename... Args>
DriverAPILoader(ReturnType(Args...)) -> DriverAPILoader<ReturnType, Args...>;

} // anonymous namespace

// Macro for defining driver API wrappers with proper error handling
#define DEFINE_DRIVER_API_WRAPPER(funcName, version)                          \
  namespace {                                                                 \
  template <typename ReturnType, typename... Args>                           \
  struct funcName##Loader {                                                  \
    static ReturnType lazilyLoadAndInvoke(Args... args) {                    \
      static auto* entry_point = []() {                                      \
        std::lock_guard<std::mutex> lock(driver_api_mutex);                  \
                                                                              \
        auto it = loaded_apis.find(#funcName);                               \
        if (it != loaded_apis.end()) {                                       \
          return reinterpret_cast<decltype(::funcName)*>(it->second);        \
        }                                                                     \
                                                                              \
        decltype(::funcName)* func_ptr = nullptr;                            \
        getDriverEntryPoint(                                                 \
            #funcName, version, reinterpret_cast<void**>(&func_ptr));        \
                                                                              \
        if (func_ptr != nullptr) {                                           \
          loaded_apis[#funcName] = reinterpret_cast<void*>(func_ptr);        \
        }                                                                     \
                                                                              \
        return func_ptr;                                                      \
      }();                                                                    \
                                                                              \
      TORCH_CHECK(entry_point != nullptr,                                    \
                  "CUDA driver API '", #funcName, "' is not available. "     \
                  "This may indicate an incompatible CUDA driver version.");  \
                                                                              \
      return entry_point(args...);                                           \
    }                                                                         \
                                                                              \
    funcName##Loader(ReturnType(Args...)) {}                                 \
  };                                                                          \
                                                                              \
  template <typename ReturnType, typename... Args>                           \
  funcName##Loader(ReturnType(Args...)) -> funcName##Loader<ReturnType, Args...>; \
  } /* anonymous namespace */                                                \
                                                                              \
  decltype(::funcName)* funcName =                                           \
      decltype(funcName##Loader(::funcName))::lazilyLoadAndInvoke

namespace at {
namespace cuda {
namespace driver {

// Define all the driver API wrappers
ALL_DRIVER_API_WRAPPER(DEFINE_DRIVER_API_WRAPPER);

// Utility function implementations
namespace detail {

void initializeDriverAPI() {
  std::lock_guard<std::mutex> lock(driver_api_mutex);
  
  // Initialize CUDA driver if not already done
  try {
    CUresult result = cuInit(0);
    if (result != CUDA_SUCCESS) {
      const char* error_string = nullptr;
      cuGetErrorString(result, &error_string);
      TORCH_WARN("Failed to initialize CUDA driver: ", 
                error_string ? error_string : "Unknown error");
    }
  } catch (...) {
    TORCH_WARN("Exception during CUDA driver initialization");
  }
}

bool isDriverAPIAvailable(const char* funcName, unsigned int minVersion) {
  std::lock_guard<std::mutex> lock(driver_api_mutex);
  
  // Check if already loaded
  auto it = loaded_apis.find(funcName);
  if (it != loaded_apis.end()) {
    return it->second != nullptr;
  }
  
  // Try to load it
  void* entry_point = nullptr;
  getDriverEntryPoint(funcName, minVersion, &entry_point);
  
  loaded_apis[funcName] = entry_point;
  return entry_point != nullptr;
}

int getDriverVersion() {
  if (cached_driver_version >= 0) {
    return cached_driver_version;
  }
  
  try {
    int version = 0;
    C10_CUDA_CHECK(cudaDriverGetVersion(&version));
    cached_driver_version = version;
    return version;
  } catch (const c10::Error& e) {
    TORCH_WARN("Failed to get CUDA driver version: ", e.what());
    return -1;
  }
}

} // namespace detail

} // namespace driver
} // namespace cuda
} // namespace at

#undef DEFINE_DRIVER_API_WRAPPER 
#endif