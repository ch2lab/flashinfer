/*
 * Copyright (c) 2023 by FlashInfer team.
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
#ifndef FLASHINFER_MMA_CUH_
#define FLASHINFER_MMA_CUH_

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cstring>
#include <type_traits>

namespace flashinfer {

namespace mma {

#if (__CUDACC_VER_MAJOR__ * 10000 + __CUDACC_VER_MINOR__ * 100 >= 120400)
#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 890))
#define FLASHINFER_MMA_F8F8F32_M16N8K32_ENABLED
#endif
#endif

#if (__CUDACC_VER_MAJOR__ >= 11)
#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 900))
#define FLASHINFER_STMATRIX_M8N8X4_ENABLED
#endif
#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800))
#define FLASHINFER_MMA_F16F16F32_M16N8K16_ENABLED
#define FLASHINFER_MMA_F16F16F16_M16N8K16_ENABLED
#endif
#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 750))
#define FLASHINFER_MMA_F16F16F32_M16N8K8_ENABLED
#define FLASHINFER_MMA_F16F16F16_M16N8K8_ENABLED
#define FLASHINFER_LDMATRIX_M8N8X4_ENABLED
#endif
#endif

#if defined(__CUDA_ARCH__)
#define FLASHINFER_RUNTIME_ASSERT(x) __brkpt()
#else
#define FLASHINFER_RUNTIME_ASSERT(x) assert(0 && x)
#endif

enum class MMAMode {
  kInit = 0U,
  kInplaceUpdate = 1U,
};

/*!
 * \brief Wrapper of PTX ldmatrix m8n8.x4 instruction, loads data from shared memory
 *   to fragment
 * \tparam T data type of the fragment
 * \param R pointer to the fragment
 * \param smem_ptr pointer to the shared memory
 */
template <typename T>
__device__ __forceinline__ void ldmatrix_m8n8x4(uint32_t* R, T* smem_ptr) {
#ifdef FLASHINFER_LDMATRIX_M8N8X4_ENABLED
  uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(R[0]), "=r"(R[1]), "=r"(R[2]), "=r"(R[3])
               : "r"(smem_int_ptr));
#else
  // sm_70 (V100) fallback: no ldmatrix. Manual wmma m8n8 fragment load.
  // Each of the 32 lanes supplies the swizzled row-base of one 8x8 tile row;
  // lanes 0-7/8-15/16-23/24-31 -> matrices 0/1/2/3. Thread t holds, per
  // matrix Mj, the pair {Mj[r][c0], Mj[r][c0+1]} with r=t/4, c0=2*(t&3).
  {
    const uint32_t lane = threadIdx.x & 0x1f;
    const uint32_t r = lane >> 2;
    const uint64_t gptr = reinterpret_cast<uint64_t>(smem_ptr);
    const uint32_t lo_this = (uint32_t)gptr;
    const uint32_t hi_this = (uint32_t)(gptr >> 32);
    auto rowbase = [&](int j) -> T* {
      const uint32_t lo_j = __shfl_sync(0xffffffff, lo_this, j);
      const uint32_t hi_j = __shfl_sync(0xffffffff, hi_this, j);
      return reinterpret_cast<T*>(((uint64_t)hi_j << 32) | lo_j);
    };
    auto pair = [&](int j) -> uint32_t {
      const T* p = rowbase(j);
      uint32_t lo, hi2;
      memcpy(&lo, p + 2 * (lane & 3u), sizeof(T));
      memcpy(&hi2, p + 2 * (lane & 3u) + 1, sizeof(T));
      return (sizeof(T) == 2) ? (lo | (hi2 << 16)) : lo;
    };
    R[0] = pair((int)r);
    R[1] = pair((int)(8 + r));
    R[2] = pair((int)(16 + r));
    R[3] = pair((int)(24 + r));
  }
#endif
}

/*!
 * \brief Wrapper of PTX ldmatrix m8n8.x4 instruction, loads data from shared memory
 *   to fragment
 * \tparam T data type of the fragment
 * \param R pointer to the fragment
 * \param smem_ptr pointer to the shared memory
 */
template <typename T>
__device__ __forceinline__ void ldmatrix_m8n8x4_left_half(uint32_t* R, T* smem_ptr) {
#ifdef FLASHINFER_LDMATRIX_M8N8X4_ENABLED
  uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, _, %1, _}, [%2];\n"
               : "=r"(R[0]), "=r"(R[1])
               : "r"(smem_int_ptr));
#else
  FLASHINFER_RUNTIME_ASSERT("Unsupported CUDA architecture for ldmatrix instruction");
#endif
}

/*!
 * \brief Wrapper of PTX ldmatrix m8n8.x4 instruction, loads data from shared memory
 *   to fragment
 * \tparam T data type of the fragment
 * \param R pointer to the fragment
 * \param smem_ptr pointer to the shared memory
 */
template <typename T>
__device__ __forceinline__ void ldmatrix_m8n8x4_right_half(uint32_t* R, T* smem_ptr) {
#ifdef FLASHINFER_LDMATRIX_M8N8X4_ENABLED
  uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {_, %0, _, %1}, [%2];\n"
               : "=r"(R[0]), "=r"(R[1])
               : "r"(smem_int_ptr));
#else
  FLASHINFER_RUNTIME_ASSERT("Unsupported CUDA architecture for ldmatrix instruction");
#endif
}

// sm_70 (no ldmatrix): m16n16k16 fragment loads via wmma (correct by construction) on
// LINEAR (kLinear) shared memory. smem_ptr = sub-tile's first element (b128-aligned);
// LEAD_B128 = smem row width in b128 units (row width in elements = LEAD_B128*16/sizeof(T)).
template <typename T>
__device__ __forceinline__ void wmma_load_m16n16k16_a(uint32_t lead_b128, uint32_t* R, T* smem_ptr) {
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, T, nvcuda::wmma::row_major> fa;
  nvcuda::wmma::load_matrix_sync(fa, smem_ptr, lead_b128 * (16u / sizeof(T)));
  // The sm_70 wmma m16n16k16 fragment is 32 bytes (16 storage slots); copy it whole into the
  // caller's uint32_t[8]. A per-element (4x) copy only transfers half the fragment.
  memcpy(R, &fa.x, sizeof(fa.x));
}
template <typename T>
__device__ __forceinline__ void wmma_load_m16n16k16_b_col(uint32_t lead_b128, uint32_t* R, T* smem_ptr) {
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, T, nvcuda::wmma::col_major> fb;
  nvcuda::wmma::load_matrix_sync(fb, smem_ptr, lead_b128 * (16u / sizeof(T)));
  // Copy the full 32-byte fragment into the caller's uint32_t[8] (see wmma_load_m16n16k16_a).
  memcpy(R, &fb.x, sizeof(fb.x));
}
template <typename T>
__device__ __forceinline__ void wmma_load_m16n16k16_b_row(uint32_t lead_b128, uint32_t* R, T* smem_ptr) {
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, T, nvcuda::wmma::row_major> fb;
  nvcuda::wmma::load_matrix_sync(fb, smem_ptr, lead_b128 * (16u / sizeof(T)));
  // Copy the full 32-byte fragment into the caller's uint32_t[8] (see wmma_load_m16n16k16_a).
  memcpy(R, &fb.x, sizeof(fb.x));
}

// NOTE: a smem-based wmma m16n8k8 helper (wmma_mma_m16n16k16_sm) was tried but wmma m16n8k8
// fragments are "incomplete type" on sm_70 (V100 wmma only supports m16n16k16 + m32n8k8).
// The s_frag (QK C) from wmma m16n16k16 has a thread->(row,col) mapping that differs from the
// PTX m16n8x2 s_frag layout the kernel's mdo/pv/store expect; a warp-shfl C-permutation
// (wmma C -> PTX s_frag) is the remaining path (see memory 340).

/*!
 * \brief Wrapper of PTX ldmatrix m8n8.x4 transposed instruction, loads data from
 *   shared memory to fragment and transposes the fragment
 * \tparam T data type of the fragment
 * \param R pointer to the fragment
 * \param smem_ptr pointer to the shared memory
 */
template <typename T>
__device__ __forceinline__ void ldmatrix_m8n8x4_trans(uint32_t* R, T* smem_ptr) {
#ifdef FLASHINFER_LDMATRIX_M8N8X4_ENABLED
  uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.trans.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(R[0]), "=r"(R[1]), "=r"(R[2]), "=r"(R[3])
               : "r"(smem_int_ptr));
#else
  // sm_70 (V100) fallback: transposed wmma m8n8 fragment load. Thread t holds
  // {Mj[2c][r], Mj[2c+1][r]} (c=t&3, r=t/4) for matrices 0..3 at lane groups.
  {
    const uint32_t lane = threadIdx.x & 0x1f;
    const uint32_t r = lane >> 2;
    const uint32_t c2 = 2 * (lane & 3u);
    const uint64_t gptr = reinterpret_cast<uint64_t>(smem_ptr);
    const uint32_t lo_this = (uint32_t)gptr;
    const uint32_t hi_this = (uint32_t)(gptr >> 32);
    auto rowbase = [&](int j) -> T* {
      const uint32_t lo_j = __shfl_sync(0xffffffff, lo_this, j);
      const uint32_t hi_j = __shfl_sync(0xffffffff, hi_this, j);
      return reinterpret_cast<T*>(((uint64_t)hi_j << 32) | lo_j);
    };
    auto pack2 = [&](const T* a, const T* b) -> uint32_t {
      uint32_t x, y;
      memcpy(&x, a, sizeof(T));
      memcpy(&y, b, sizeof(T));
      return (sizeof(T) == 2) ? (x | (y << 16)) : x;
    };
    auto tpair = [&](int m8) -> uint32_t {
      // trans m8n8: thread t holds {M[2c][r], M[2c+1][r]}; row (2c) base + col r.
      const T* a = rowbase(m8 + c2) + r;
      const T* b = rowbase(m8 + c2 + 1) + r;
      return pack2(a, b);
    };
    R[0] = tpair(0);
    R[1] = tpair(8);
    R[2] = tpair(16);
    R[3] = tpair(24);
  }
#endif
}

/*!
 * \brief Wrapper of PTX ldmatrix m8n8.x4 transposed instruction, loads data from
 *   shared memory to fragment and transposes the fragment
 * \tparam T data type of the fragment
 * \param R pointer to the fragment
 * \param smem_ptr pointer to the shared memory
 */
template <typename T>
__device__ __forceinline__ void ldmatrix_m8n8x4_trans_left_half(uint32_t* R, T* smem_ptr) {
#ifdef FLASHINFER_LDMATRIX_M8N8X4_ENABLED
  uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.trans.m8n8.x4.shared.b16 {%0, %1, _, _}, [%2];\n"
               : "=r"(R[0]), "=r"(R[1])
               : "r"(smem_int_ptr));
#else
  FLASHINFER_RUNTIME_ASSERT("Unsupported CUDA architecture for ldmatrix instruction");
#endif
}

/*!
 * \brief Wrapper of PTX ldmatrix m8n8.x4 transposed instruction, loads data from
 *   shared memory to fragment and transposes the fragment
 * \tparam T data type of the fragment
 * \param R pointer to the fragment
 * \param smem_ptr pointer to the shared memory
 */
template <typename T>
__device__ __forceinline__ void ldmatrix_m8n8x4_trans_right_half(uint32_t* R, T* smem_ptr) {
#ifdef FLASHINFER_LDMATRIX_M8N8X4_ENABLED
  uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.trans.m8n8.x4.shared.b16 {_, _, %0, %1}, [%2];\n"
               : "=r"(R[0]), "=r"(R[1])
               : "r"(smem_int_ptr));
#else
  FLASHINFER_RUNTIME_ASSERT("Unsupported CUDA architecture for ldmatrix instruction");
#endif
}

/*!
 * \brief Wrapper of PTX stmatrix m8n8.x4 instruction, stores data from fragment
 *   to shared memory
 * \tparam T data type of the fragment
 * \param R pointer to the fragment
 * \param smem_ptr pointer to the shared memory
 */
template <typename T>
__device__ __forceinline__ void stmatrix_m8n8x4(uint32_t* R, T* smem_ptr) {
#ifdef FLASHINFER_STMATRIX_M8N8X4_ENABLED
  uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
               :
               : "r"(smem_int_ptr), "r"(R[0]), "r"(R[1]), "r"(R[2]), "r"(R[3]));
#else
  // Fallback implementation, slower than PTX instruction
  const uint32_t tx = threadIdx.x;
  uint4 word;
#pragma unroll
  for (uint32_t reg_id = 0; reg_id < 4; ++reg_id) {
    word.x = __shfl_sync(0xffffffff, R[reg_id], (tx % 8) * 4);
    word.y = __shfl_sync(0xffffffff, R[reg_id], (tx % 8) * 4 + 1);
    word.z = __shfl_sync(0xffffffff, R[reg_id], (tx % 8) * 4 + 2);
    word.w = __shfl_sync(0xffffffff, R[reg_id], (tx % 8) * 4 + 3);
    if (tx / 8 == reg_id) {
      *(uint4*)smem_ptr = word;
    }
  }
#endif
}

/*!
 * \brief Wrapper of two mma m16n8k32 instructions for row major and column major f8 matrix
 *   multiplication, accumulated in f32.
 * \tparam T data type of the fragment
 * \tparam mma_mode whether we are initializing the accumulator or updating it
 * \param C pointer to the accumulator
 * \param A pointer to the fragment of matrix A
 * \param B pointer to the fragment of matrix B
 */
template <typename T, MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n16k32_row_col_f8f8f32(float* C, uint32_t* A,
                                                                   uint32_t* B) {
  static_assert(sizeof(T) == 1, "DType must be 8bit floating data type");
#if defined(FLASHINFER_MMA_F8F8F32_M16N8K32_ENABLED)
  if constexpr (mma_mode == MMAMode::kInit) {
    if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
      asm volatile(
          "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(0.f), "f"(0.f),
            "f"(0.f), "f"(0.f));
      asm volatile(
          "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(C[4]), "=f"(C[5]), "=f"(C[6]), "=f"(C[7])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "f"(0.f), "f"(0.f),
            "f"(0.f), "f"(0.f));
    } else {  // e5m2
      asm volatile(
          "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(0.f), "f"(0.f),
            "f"(0.f), "f"(0.f));
      asm volatile(
          "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(C[4]), "=f"(C[5]), "=f"(C[6]), "=f"(C[7])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "f"(0.f), "f"(0.f),
            "f"(0.f), "f"(0.f));
    }
  } else {
    if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
      asm volatile(
          "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(C[0]), "f"(C[1]),
            "f"(C[2]), "f"(C[3]));
      asm volatile(
          "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(C[4]), "=f"(C[5]), "=f"(C[6]), "=f"(C[7])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "f"(C[4]), "f"(C[5]),
            "f"(C[6]), "f"(C[7]));
    } else {  // e5m2
      asm volatile(
          "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(C[0]), "f"(C[1]),
            "f"(C[2]), "f"(C[3]));
      asm volatile(
          "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(C[4]), "=f"(C[5]), "=f"(C[6]), "=f"(C[7])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "f"(C[4]), "f"(C[5]),
            "f"(C[6]), "f"(C[7]));
    }
  }
#else
  FLASHINFER_RUNTIME_ASSERT(
      "fp8 mma instruction is only available for sm89, PTX 8.4+ and CUDA 12.4+");
#endif
}

/*!
 * \brief Wrapper of two mma m16n8k16 instructions for row major and column major f16 matrix
 *   multiplication, accumulated in f32.
 * \tparam T data type of the fragment
 * \tparam mma_mode whether we are initializing the accumulator or updating it
 * \param C pointer to the accumulator
 * \param A pointer to the fragment of matrix A
 * \param B pointer to the fragment of matrix B
 */
template <typename T, MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n16k16_row_col_f16f16f32(float* C, uint32_t* A,
                                                                     uint32_t* B) {
#if defined(FLASHINFER_MMA_F16F16F32_M16N8K16_ENABLED)
  if constexpr (mma_mode == MMAMode::kInit) {
    if constexpr (std::is_same_v<T, half>) {
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(0.f), "f"(0.f),
            "f"(0.f), "f"(0.f));
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(C[4]), "=f"(C[5]), "=f"(C[6]), "=f"(C[7])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "f"(0.f), "f"(0.f),
            "f"(0.f), "f"(0.f));
    } else {
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(0.f), "f"(0.f),
            "f"(0.f), "f"(0.f));
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(C[4]), "=f"(C[5]), "=f"(C[6]), "=f"(C[7])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "f"(0.f), "f"(0.f),
            "f"(0.f), "f"(0.f));
    }
  } else {
    if constexpr (std::is_same_v<T, half>) {
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(C[0]), "f"(C[1]),
            "f"(C[2]), "f"(C[3]));
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(C[4]), "=f"(C[5]), "=f"(C[6]), "=f"(C[7])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "f"(C[4]), "f"(C[5]),
            "f"(C[6]), "f"(C[7]));
    } else {
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(C[0]), "f"(C[1]),
            "f"(C[2]), "f"(C[3]));
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(C[4]), "=f"(C[5]), "=f"(C[6]), "=f"(C[7])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "f"(C[4]), "f"(C[5]),
            "f"(C[6]), "f"(C[7]));
    }
  }
#elif defined(FLASHINFER_MMA_F16F16F32_M16N8K8_ENABLED)
  if constexpr (std::is_same_v<T, half>) {
    if constexpr (mma_mode == MMAMode::kInit) {
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5},"
          "{%6},"
          "{%7, %8, %9, %10};\n"
          : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
          : "r"(A[0]), "r"(A[1]), "r"(B[0]), "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f));
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5},"
          "{%6},"
          "{%7, %8, %9, %10};\n"
          : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
          : "r"(A[2]), "r"(A[3]), "r"(B[1]), "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5},"
          "{%6},"
          "{%7, %8, %9, %10};\n"
          : "=f"(C[4]), "=f"(C[5]), "=f"(C[6]), "=f"(C[7])
          : "r"(A[0]), "r"(A[1]), "r"(B[2]), "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f));
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5},"
          "{%6},"
          "{%7, %8, %9, %10};\n"
          : "=f"(C[4]), "=f"(C[5]), "=f"(C[6]), "=f"(C[7])
          : "r"(A[2]), "r"(A[3]), "r"(B[3]), "f"(C[4]), "f"(C[5]), "f"(C[6]), "f"(C[7]));
    } else {
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5},"
          "{%6},"
          "{%7, %8, %9, %10};\n"
          : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
          : "r"(A[0]), "r"(A[1]), "r"(B[0]), "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5},"
          "{%6},"
          "{%7, %8, %9, %10};\n"
          : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
          : "r"(A[2]), "r"(A[3]), "r"(B[1]), "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5},"
          "{%6},"
          "{%7, %8, %9, %10};\n"
          : "=f"(C[4]), "=f"(C[5]), "=f"(C[6]), "=f"(C[7])
          : "r"(A[0]), "r"(A[1]), "r"(B[2]), "f"(C[4]), "f"(C[5]), "f"(C[6]), "f"(C[7]));
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5},"
          "{%6},"
          "{%7, %8, %9, %10};\n"
          : "=f"(C[4]), "=f"(C[5]), "=f"(C[6]), "=f"(C[7])
          : "r"(A[2]), "r"(A[3]), "r"(B[3]), "f"(C[4]), "f"(C[5]), "f"(C[6]), "f"(C[7]));
    }
  } else {
    FLASHINFER_RUNTIME_ASSERT("Unsupported CUDA architecture for mma instruction");
  }
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700) && (__CUDA_ARCH__ < 750)
  // sm_70 (V100): no PTX mma.sync; use wmma m16n16k16 (HMMA.16816).
  if constexpr (std::is_same_v<T, half>) {
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> a;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> b;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c;
    // A/B are the full 32-byte wmma m16n16k16 fragments (uint32_t[8]); copy them whole.
    memcpy(&a.x, A, sizeof(a.x));
    memcpy(&b.x, B, sizeof(b.x));
    if constexpr (mma_mode == MMAMode::kInit) {
      nvcuda::wmma::fill_fragment(c, 0.0f);
    } else {
      memcpy(&c.x, C, sizeof(float) * 8);
    }
    nvcuda::wmma::mma_sync(c, a, b, c);
    memcpy(C, &c.x, sizeof(float) * 8);
  } else {
    FLASHINFER_RUNTIME_ASSERT("Unsupported data type for sm_70 mma");
  }
#else
  FLASHINFER_RUNTIME_ASSERT("Unsupported CUDA architecture for mma instruction");
#endif
}

// A row-major / B row-major. For the V-multiply: O[m][n] = P[m][k] * V[k][n]; V is stored
// row-major [kv][head_dim] (n contiguous) => B row-major. sm_75+/80: B-fragment comes from
// ldmatrix.trans, so the ptx instruction is identical to row_col (delegate). sm_70: wmma.
template <typename T = half, MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n16k16_row_row_f16f16f32(float* C, uint32_t* A,
                                                                      uint32_t* B) {
#if defined(FLASHINFER_MMA_F16F16F32_M16N8K16_ENABLED) || \
    defined(FLASHINFER_MMA_F16F16F32_M16N8K8_ENABLED)
  mma_sync_m16n16k16_row_col_f16f16f32<T, mma_mode>(C, A, B);
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700) && (__CUDA_ARCH__ < 750)
  if constexpr (std::is_same_v<T, half>) {
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> a;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::row_major> b;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c;
    // A/B are the full 32-byte wmma m16n16k16 fragments (uint32_t[8]); copy them whole.
    memcpy(&a.x, A, sizeof(a.x));
    memcpy(&b.x, B, sizeof(b.x));
    if constexpr (mma_mode == MMAMode::kInit) {
      nvcuda::wmma::fill_fragment(c, 0.0f);
    } else {
      memcpy(&c.x, C, sizeof(float) * 8);
    }
    nvcuda::wmma::mma_sync(c, a, b, c);
    memcpy(C, &c.x, sizeof(float) * 8);
  } else {
    FLASHINFER_RUNTIME_ASSERT("Unsupported data type for sm_70 mma");
  }
#else
  FLASHINFER_RUNTIME_ASSERT("Unsupported CUDA architecture for mma instruction");
#endif
}

/*!
 * \brief Use mma instructions to compute rowsum.
 */
template <typename DType>
__device__ __forceinline__ void m16k32_rowsum_f8f8f32(float* d, DType* s) {
  static_assert(sizeof(DType) == 1, "DType must be 8bit floating data type");
  uint32_t* s_u32 = (uint32_t*)(s);
#if defined(FLASHINFER_MMA_F8F8F32_M16N8K32_ENABLED)
  if constexpr (std::is_same_v<DType, __nv_fp8_e4m3>) {
    asm volatile(
        "{\n"
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,  _,  %1,  _},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  0.,  %9,  0.};\n"
        "}\n"
        : "=f"(d[0]), "=f"(d[1])
        : "r"(s_u32[0]), "r"(s_u32[1]), "r"(s_u32[2]), "r"(s_u32[3]), "r"(943208504),
          "r"(943208504), "f"(d[0]), "f"(d[1]));
  } else {  // e5m2
    asm volatile(
        "{\n"
        "mma.sync.aligned.m16n8k16.row.col.f32.e5m2.e5m2.f32 "
        "{%0,  _,  %1,  _},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  0.,  %9,  0.};\n"
        "}\n"
        : "=f"(d[0]), "=f"(d[1])
        : "r"(s_u32[0]), "r"(s_u32[1]), "r"(s_u32[2]), "r"(s_u32[3]), "r"(1010580540),
          "r"(1010580540), "f"(d[0]), "f"(d[1]));
  }
#else
  FLASHINFER_RUNTIME_ASSERT(
      "fp8 mma instruction is only available for sm89, PTX 8.4+ and CUDA 12.4+");
#endif
}

/*!
 * \brief Use mma instructions to compute rowsum.
 */
template <typename DType>
__device__ __forceinline__ void m16k16_rowsum_f16f16f32(float* d, DType* s) {
  static_assert(sizeof(DType) == 2, "DType must be 16bit floating data type");
  uint32_t* s_u32 = (uint32_t*)(s);
#if defined(FLASHINFER_MMA_F16F16F32_M16N8K16_ENABLED)
  if constexpr (std::is_same_v<DType, half>) {
    asm volatile(
        "{\n"
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,  _,  %1,  _},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  0.,  %9,  0.};\n"
        "}\n"
        : "=f"(d[0]), "=f"(d[1])
        : "r"(s_u32[0]), "r"(s_u32[1]), "r"(s_u32[2]), "r"(s_u32[3]), "r"(1006648320),
          "r"(1006648320), "f"(d[0]), "f"(d[1]));
  } else {
    asm volatile(
        "{\n"
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,  _,  %1,  _},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  0.,  %9,  0.};\n"
        "}\n"
        : "=f"(d[0]), "=f"(d[1])
        : "r"(s_u32[0]), "r"(s_u32[1]), "r"(s_u32[2]), "r"(s_u32[3]), "r"(1065369472),
          "r"(1065369472), "f"(d[0]), "f"(d[1]));
  }
#elif defined(FLASHINFER_MMA_F16F16F32_M16N8K8_ENABLED)
  if constexpr (std::is_same_v<DType, half>) {
    asm volatile(
        "{\n"
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,  _,  %1,  _},"
        "{%2,  %3},"
        "{%4},"
        "{%5,  0.,  %6,  0.};\n"
        "}\n"
        : "=f"(d[0]), "=f"(d[1])
        : "r"(s_u32[0]), "r"(s_u32[1]), "r"(1006648320), "f"(d[0]), "f"(d[1]));
    asm volatile(
        "{\n"
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,  _,  %1,  _},"
        "{%2,  %3},"
        "{%4},"
        "{%5,  0.,  %6,  0.};\n"
        "}\n"
        : "=f"(d[0]), "=f"(d[1])
        : "r"(s_u32[2]), "r"(s_u32[3]), "r"(1006648320), "f"(d[0]), "f"(d[1]));
  } else {
    FLASHINFER_RUNTIME_ASSERT("Unsupported CUDA architecture for mma instruction");
  }
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700) && (__CUDA_ARCH__ < 750)
  // sm_70 (V100) wmma rowsum: mma with an all-ones B fragment -> per-row sums.
  {
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> a;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> b;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c;
    memcpy(&a.x, s, sizeof(uint32_t) * 4);
    nvcuda::wmma::fill_fragment(b, (__half)1.0f);
    nvcuda::wmma::fill_fragment(c, 0.0f);
    nvcuda::wmma::mma_sync(c, a, b, c);
    d[0] = c.x[0];
    d[1] = c.x[4];
  }
#else
  FLASHINFER_RUNTIME_ASSERT("Unsupported CUDA architecture for mma instruction");
#endif
}

/*!
 * \brief Wrapper of two mma m16n8k16 instructions for row major and column major f16 matrix
 *   multiplication, accumulated in f16.
 * \tparam mma_mode whether we are initializing the accumulator or updating it
 * \param C pointer to the accumulator
 * \param A pointer to the fragment of matrix A
 * \param B pointer to the fragment of matrix B
 */
template <MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n16k16_row_col_f16f16f16(uint32_t* C, uint32_t* A,
                                                                     uint32_t* B) {
#if defined(FLASHINFER_MMA_F16F16F16_M16N8K16_ENABLED)
  if constexpr (mma_mode == MMAMode::kInit) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  %9};\n"
        : "=r"(C[0]), "=r"(C[1])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(0), "r"(0));
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  %9};\n"
        : "=r"(C[2]), "=r"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "r"(0), "r"(0));
  } else {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  %9};\n"
        : "=r"(C[0]), "=r"(C[1])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(C[0]), "r"(C[1]));
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  %9};\n"
        : "=r"(C[2]), "=r"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "r"(C[2]), "r"(C[3]));
  }
#elif defined(FLASHINFER_MMA_F16F16F16_M16N8K8_ENABLED)
  if constexpr (mma_mode == MMAMode::kInit) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3},"
        "{%4},"
        "{%5, %6};\n"
        : "=r"(C[0]), "=r"(C[1])
        : "r"(A[0]), "r"(A[1]), "r"(B[0]), "r"(0), "r"(0));
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3},"
        "{%4},"
        "{%5, %6};\n"
        : "=r"(C[0]), "=r"(C[1])
        : "r"(A[2]), "r"(A[3]), "r"(B[1]), "r"(0), "r"(0));
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3},"
        "{%4},"
        "{%5, %6};\n"
        : "=r"(C[2]), "=r"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(B[2]), "r"(0), "r"(0));
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3},"
        "{%4},"
        "{%5, %6};\n"
        : "=r"(C[2]), "=r"(C[3])
        : "r"(A[2]), "r"(A[3]), "r"(B[3]), "r"(0), "r"(0));
  } else {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3},"
        "{%4},"
        "{%5, %6};\n"
        : "=r"(C[0]), "=r"(C[1])
        : "r"(A[0]), "r"(A[1]), "r"(B[0]), "r"(C[0]), "r"(C[1]));
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3},"
        "{%4},"
        "{%5, %6};\n"
        : "=r"(C[0]), "=r"(C[1])
        : "r"(A[2]), "r"(A[3]), "r"(B[1]), "r"(C[0]), "r"(C[1]));
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3},"
        "{%4},"
        "{%5, %6};\n"
        : "=r"(C[2]), "=r"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(B[2]), "r"(C[2]), "r"(C[3]));
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3},"
        "{%4},"
        "{%5, %6};\n"
        : "=r"(C[2]), "=r"(C[3])
        : "r"(A[2]), "r"(A[3]), "r"(B[3]), "r"(C[2]), "r"(C[3]));
  }
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700) && (__CUDA_ARCH__ < 750)
  // sm_70 (V100) wmma m16n16k16, f16 accumulator.
  {
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> a;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> b;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, __half> c;
    // A/B are the full 32-byte wmma m16n16k16 fragments (uint32_t[8]); copy them whole.
    // C is the 16-byte half accumulator (8 halfs); keep the 4x uint32 copy.
    memcpy(&a.x, A, sizeof(a.x));
    memcpy(&b.x, B, sizeof(b.x));
    if constexpr (mma_mode == MMAMode::kInit) {
      nvcuda::wmma::fill_fragment(c, (__half)0.0f);
    } else {
      memcpy(&c.x, C, sizeof(uint32_t) * 4);
    }
    nvcuda::wmma::mma_sync(c, a, b, c);
    memcpy(C, &c.x, sizeof(uint32_t) * 4);
  }
#else
  FLASHINFER_RUNTIME_ASSERT("Unsupported CUDA architecture for mma instruction");
#endif
}

}  // namespace mma

}  // namespace flashinfer

#endif  // FLASHINFER_MMA_CUH_
