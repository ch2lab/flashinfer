/*
 * Copyright (c) 2024 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_FASTDIV_CUH_
#define FLASHINFER_FASTDIV_CUH_
#include <cstdint>
#include <cuda/cmath>

namespace flashinfer {

// API-compatible wrapper around cuda::fast_mod_div<uint32_t>.
// Preserves the default constructor, implicit conversions, and divmod()
// method expected by existing call sites throughout the attention kernels.
// cuda::fast_mod_div requires SM75+; on SM70 targets fall back to plain
// division. __CUDA_ARCH_LIST__ reflects the target in both host and device
// passes; __CUDA_ARCH__ covers pure device passes.
#if (defined(__CUDA_ARCH_LIST__) && (__CUDA_ARCH_LIST__ < 750)) || \
    (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 750) && !defined(__CUDA_ARCH_LIST__))
#define FLASHINFER_FASTDIV_PLAIN 1
#endif
struct uint_fastdiv {
#ifndef FLASHINFER_FASTDIV_PLAIN
  __host__ __device__ uint_fastdiv() : impl_(1), d_(0) {}

  __host__ uint_fastdiv(uint32_t d) : impl_(d ? d : 1), d_(d) {}

  __host__ __device__ __forceinline__ operator unsigned int() const { return d_; }

  __host__ __device__ __forceinline__ void divmod(uint32_t n, uint32_t& q, uint32_t& r) const {
    q = n / impl_;
    r = n - q * d_;
  }

 private:
  cuda::fast_mod_div<uint32_t> impl_;
  uint32_t d_;
#else
  __host__ __device__ uint_fastdiv() : d_(1) {}

  __host__ uint_fastdiv(uint32_t d) : d_(d ? d : 1) {}

  __host__ __device__ __forceinline__ operator unsigned int() const { return d_; }

  __host__ __device__ __forceinline__ void divmod(uint32_t n, uint32_t& q, uint32_t& r) const {
    q = n / d_;
    r = n - q * d_;
  }

 private:
  uint32_t d_;
#endif
};

__host__ __device__ __forceinline__ uint32_t operator/(const uint32_t n,
                                                       const uint_fastdiv& divisor) {
  uint32_t q, r;
  divisor.divmod(n, q, r);
  return q;
}

__host__ __device__ __forceinline__ uint32_t operator%(const uint32_t n,
                                                       const uint_fastdiv& divisor) {
  uint32_t q, r;
  divisor.divmod(n, q, r);
  return r;
}

}  // namespace flashinfer

#endif  // FLASHINFER_FASTDIV_CUH_
