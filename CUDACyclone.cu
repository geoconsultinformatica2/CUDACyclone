
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <thread>
#include <chrono>
#include <cmath>
#include <csignal>
#include <atomic>
#include <algorithm>
#include <vector>
#ifdef RTX5090_OPT
#include <cerrno>
#include <filesystem>
#include <fstream>
#include <map>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include "CUDAMath.h"
#include "sha256.h"
#include "CUDAHash.cuh"
#include "CUDAUtils.h"
#include "CUDAStructures.h"

static volatile sig_atomic_t g_sigint = 0;
static void handle_sigint(int) { g_sigint = 1; }

__device__ __forceinline__ int load_found_flag_relaxed(const int* p) {
    return *((const volatile int*)p);
}
__device__ __forceinline__ bool warp_found_ready(const int* __restrict__ d_found_flag,
                                                 unsigned full_mask,
                                                 unsigned lane)
{
    int f = 0;
    if (lane == 0) f = load_found_flag_relaxed(d_found_flag);
    f = __shfl_sync(full_mask, f, 0);
    return f == FOUND_READY;
}

#ifndef MAX_BATCH_SIZE
#define MAX_BATCH_SIZE 1024
#endif
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

__constant__ uint64_t c_Gx[(MAX_BATCH_SIZE/2) * 4];
__constant__ uint64_t c_Gy[(MAX_BATCH_SIZE/2) * 4];
__constant__ uint64_t c_Jx[4];
__constant__ uint64_t c_Jy[4];

#ifdef RTX5090_OPT
static_assert(RTX5090_BATCH == 64 || RTX5090_BATCH == 128 || RTX5090_BATCH == 256 ||
              RTX5090_BATCH == 512 || RTX5090_BATCH == 1024,
              "RTX5090_BATCH must be one of 64, 128, 256, 512, 1024");
static_assert(RTX5090_TPB >= 32 && RTX5090_TPB <= 1024 && (RTX5090_TPB % 32) == 0,
              "RTX5090_TPB must be a valid whole-warp block size");

static __device__ __forceinline__ bool hash160_words_match_5090(
    const uint32_t h[5], const uint32_t target_prefix)
{
    if (h[0] != target_prefix) return false;
    return h[1] == c_target_hash160_words[1]
        && h[2] == c_target_hash160_words[2]
        && h[3] == c_target_hash160_words[3]
        && h[4] == c_target_hash160_words[4];
}

template<int B>
__launch_bounds__(RTX5090_TPB, RTX5090_MIN_BLOCKS)
#else
__launch_bounds__(256, 2)
#endif
__global__ void kernel_point_add_and_check_oneinv(
    const uint64_t* __restrict__ Px,
    const uint64_t* __restrict__ Py,
    uint64_t* __restrict__ Rx,
    uint64_t* __restrict__ Ry,
    uint64_t* __restrict__ start_scalars,
    uint64_t* __restrict__ counts256,
    uint64_t threadsTotal,
#ifndef RTX5090_OPT
    uint32_t batch_size,
#endif
    uint32_t max_batches_per_launch,
    int* __restrict__ d_found_flag,
    FoundResult* __restrict__ d_found_result,
    unsigned long long* __restrict__ hashes_accum,
    unsigned int* __restrict__ d_any_left
)
{
#ifdef RTX5090_OPT
    constexpr int half = B / 2;
#else
    const int B = (int)batch_size;
    if (B <= 0 || (B & 1) || B > MAX_BATCH_SIZE) return;
    const int half = B >> 1;
#endif

    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= threadsTotal) return;

    const unsigned lane      = (unsigned)(threadIdx.x & (WARP_SIZE - 1));
    const unsigned full_mask = 0xFFFFFFFFu;
    if (warp_found_ready(d_found_flag, full_mask, lane)) return;

    const uint32_t target_prefix = c_target_prefix;

    unsigned int local_hashes = 0;
    #define FLUSH_THRESHOLD 65536u
    #define WARP_FLUSH_HASHES() do { \
        unsigned long long v = warp_reduce_add_ull((unsigned long long)local_hashes); \
        if (lane == 0 && v) atomicAdd(hashes_accum, v); \
        local_hashes = 0; \
    } while (0)
    #define MAYBE_WARP_FLUSH() do { if ((local_hashes & (FLUSH_THRESHOLD - 1u)) == 0u) WARP_FLUSH_HASHES(); } while (0)

    uint64_t x1[4], y1[4], S[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint64_t idx = gid * 4 + i;
        x1[i] = Px[idx];
        y1[i] = Py[idx];
        S[i]  = start_scalars[idx];   
    }
    uint64_t rem[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) rem[i] = counts256[gid*4 + i];

    if ((rem[0]|rem[1]|rem[2]|rem[3]) == 0ull) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { Rx[gid*4+i] = x1[i]; Ry[gid*4+i] = y1[i]; }
        WARP_FLUSH_HASHES(); return;
    }

    uint32_t batches_done = 0;

    while (batches_done < max_batches_per_launch && ge256_u64(rem, (uint64_t)B)) {
        if (warp_found_ready(d_found_flag, full_mask, lane)) { WARP_FLUSH_HASHES(); return; }

        {
#ifdef RTX5090_OPT
            uint32_t h160[5];
            const uint8_t prefix = (y1[0] & 1ULL) ? 0x03 : 0x02;
            getHash160_33_from_limbs_5090(prefix, x1, h160);
#else
            uint8_t h20[20];
            uint8_t prefix = (uint8_t)(y1[0] & 1ULL) ? 0x03 : 0x02;
            getHash160_33_from_limbs(prefix, x1, h20);
#endif
            ++local_hashes; MAYBE_WARP_FLUSH();

#ifdef RTX5090_OPT
            bool pref = h160[0] == target_prefix;
#else
            bool pref = hash160_prefix_equals(h20, target_prefix);
#endif
#ifdef RTX5090_OPT
            const bool full_match = pref && hash160_words_match_5090(h160, target_prefix);
            if (__any_sync(full_mask, full_match)) {
                if (full_match) {
#else
            if (__any_sync(full_mask, pref)) {
                if (pref && hash160_matches_prefix_then_full(h20, c_target_hash160, target_prefix)) {
#endif
                    if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                        d_found_result->threadId = (int)gid;
                        d_found_result->iter     = 0;
#pragma unroll
                        for (int k=0;k<4;++k) d_found_result->scalar[k]=S[k];
#pragma unroll
                        for (int k=0;k<4;++k) d_found_result->Rx[k]=x1[k];
#pragma unroll
                        for (int k=0;k<4;++k) d_found_result->Ry[k]=y1[k];
                        __threadfence_system();
                        atomicExch(d_found_flag, FOUND_READY);
                    }
                }
                __syncwarp(full_mask); WARP_FLUSH_HASHES(); return;
            }
        }

        uint64_t subp[MAX_BATCH_SIZE/2][4];
        uint64_t acc[4], tmp[4];

#pragma unroll
        for (int j=0;j<4;++j) acc[j] = c_Jx[j];
        ModSub256(acc, acc, x1);
#pragma unroll
        for (int j=0;j<4;++j) subp[half-1][j] = acc[j];

        for (int i = half - 2; i >= 0; --i) {
#pragma unroll
            for (int j=0;j<4;++j) tmp[j] = c_Gx[(size_t)(i+1)*4 + j];
            ModSub256(tmp, tmp, x1);
            _ModMult(acc, acc, tmp);
#pragma unroll
            for (int j=0;j<4;++j) subp[i][j] = acc[j];
        }

        uint64_t d0[4], inverse[5];
#pragma unroll
        for (int j=0;j<4;++j) d0[j] = c_Gx[0*4 + j];
        ModSub256(d0, d0, x1);
#pragma unroll
        for (int j=0;j<4;++j) inverse[j] = d0[j];
        _ModMult(inverse, subp[0]);
        inverse[4] = 0ull;
        _ModInv(inverse);

        uint64_t sy_neg[4], sx_neg[4];
        ModNeg256(sy_neg, y1);
        ModNeg256(sx_neg, x1);

        for (int i = 0; i < half - 1; ++i) {
            if (warp_found_ready(d_found_flag, full_mask, lane)) { WARP_FLUSH_HASHES(); return; }

            uint64_t dx_inv_i[4];
            _ModMult(dx_inv_i, subp[i], inverse);

            {
                uint64_t px3[4], s[4], lam[4];
                uint64_t px_i[4], py_i[4];
#pragma unroll
                for (int j=0;j<4;++j) { px_i[j]=c_Gx[(size_t)i*4+j]; py_i[j]=c_Gy[(size_t)i*4+j]; }

                ModSub256(s, py_i, y1);
                _ModMult(lam, s, dx_inv_i);

                _ModSqr(px3, lam);     
                ModSub256(px3, px3, x1);
                ModSub256(px3, px3, px_i);

                ModSub256(s, x1, px3); 
                _ModMult(s, s, lam);
                uint8_t odd; ModSub256isOdd(s, y1, &odd);

#ifdef RTX5090_OPT
                uint32_t h160[5]; getHash160_33_from_limbs_5090(odd?0x03:0x02, px3, h160);
#else
                uint8_t h20[20]; getHash160_33_from_limbs(odd?0x03:0x02, px3, h20);
#endif
                ++local_hashes; MAYBE_WARP_FLUSH();

#ifdef RTX5090_OPT
                bool pref = h160[0] == target_prefix;
#else
                bool pref = hash160_prefix_equals(h20, target_prefix);
#endif
#ifdef RTX5090_OPT
                const bool full_match = pref && hash160_words_match_5090(h160, target_prefix);
                if (__any_sync(full_mask, full_match)) {
                    if (full_match) {
#else
                if (__any_sync(full_mask, pref)) {
                    if (pref && hash160_matches_prefix_then_full(h20, c_target_hash160, target_prefix)) {
#endif
                        if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                            uint64_t fs[4]; for (int k=0;k<4;++k) fs[k]=S[k];
                            uint64_t addv=(uint64_t)(i+1);
                            for (int k=0;k<4 && addv;++k){ uint64_t old=fs[k]; fs[k]=old+addv; addv=(fs[k]<old)?1ull:0ull; }
#pragma unroll
                            for (int k=0;k<4;++k) d_found_result->scalar[k]=fs[k];
#pragma unroll
                            for (int k=0;k<4;++k) d_found_result->Rx[k]=px3[k];
                           
                            uint64_t y3[4]; uint64_t t[4]; ModSub256(t, x1, px3); _ModMult(y3, t, lam); ModSub256(y3, y3, y1);
#pragma unroll
                            for (int k=0;k<4;++k) d_found_result->Ry[k]=y3[k];
                            d_found_result->threadId = (int)gid;
                            d_found_result->iter     = 0;
                            __threadfence_system();
                            atomicExch(d_found_flag, FOUND_READY);
                        }
                    }
                    __syncwarp(full_mask); WARP_FLUSH_HASHES(); return;
                }
            }

            {
                uint64_t px3[4], s[4], lam[4];
                uint64_t px_i[4], py_i[4];
#pragma unroll
                for (int j=0;j<4;++j) { px_i[j]=c_Gx[(size_t)i*4+j]; py_i[j]=c_Gy[(size_t)i*4+j]; }
                ModNeg256(py_i, py_i); 

                ModSub256(s, py_i, y1);
                _ModMult(lam, s, dx_inv_i);

                _ModSqr(px3, lam);
                ModSub256(px3, px3, x1);
                ModSub256(px3, px3, px_i);

                ModSub256(s, x1, px3);
                _ModMult(s, s, lam);
                uint8_t odd; ModSub256isOdd(s, y1, &odd);

#ifdef RTX5090_OPT
                uint32_t h160[5]; getHash160_33_from_limbs_5090(odd?0x03:0x02, px3, h160);
#else
                uint8_t h20[20]; getHash160_33_from_limbs(odd?0x03:0x02, px3, h20);
#endif
                ++local_hashes; MAYBE_WARP_FLUSH();

#ifdef RTX5090_OPT
                bool pref = h160[0] == target_prefix;
#else
                bool pref = hash160_prefix_equals(h20, target_prefix);
#endif
#ifdef RTX5090_OPT
                const bool full_match = pref && hash160_words_match_5090(h160, target_prefix);
                if (__any_sync(full_mask, full_match)) {
                    if (full_match) {
#else
                if (__any_sync(full_mask, pref)) {
                    if (pref && hash160_matches_prefix_then_full(h20, c_target_hash160, target_prefix)) {
#endif
                        if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                            uint64_t fs[4]; for (int k=0;k<4;++k) fs[k]=S[k];
                            uint64_t sub=(uint64_t)(i+1);
                            for (int k=0;k<4 && sub;++k){ uint64_t old=fs[k]; fs[k]=old-sub; sub=(old<sub)?1ull:0ull; }
#pragma unroll
                            for (int k=0;k<4;++k) d_found_result->scalar[k]=fs[k];
#pragma unroll
                            for (int k=0;k<4;++k) d_found_result->Rx[k]=px3[k];
                            uint64_t y3[4]; uint64_t t[4]; ModSub256(t, x1, px3); _ModMult(y3, t, lam); ModSub256(y3, y3, y1);
#pragma unroll
                            for (int k=0;k<4;++k) d_found_result->Ry[k]=y3[k];
                            d_found_result->threadId = (int)gid;
                            d_found_result->iter     = 0;
                            __threadfence_system();
                            atomicExch(d_found_flag, FOUND_READY);
                        }
                    }
                    __syncwarp(full_mask); WARP_FLUSH_HASHES(); return;
                }
            }

            uint64_t gxmi[4];
#pragma unroll
            for (int j=0;j<4;++j) gxmi[j] = c_Gx[(size_t)i*4 + j];
            ModSub256(gxmi, gxmi, x1);
            _ModMult(inverse, inverse, gxmi);
        }

        {
            const int i = half - 1;
            uint64_t dx_inv_i[4];
            _ModMult(dx_inv_i, subp[i], inverse);

            uint64_t px3[4], s[4], lam[4];
            uint64_t px_i[4], py_i[4];
#pragma unroll
            for (int j=0;j<4;++j) { px_i[j]=c_Gx[(size_t)i*4+j]; py_i[j]=c_Gy[(size_t)i*4+j]; }
            ModNeg256(py_i, py_i);

            ModSub256(s, py_i, y1);
            _ModMult(lam, s, dx_inv_i);

            _ModSqr(px3, lam);
            ModSub256(px3, px3, x1);
            ModSub256(px3, px3, px_i);

            ModSub256(s, x1, px3);
            _ModMult(s, s, lam);
            uint8_t odd; ModSub256isOdd(s, y1, &odd);

#ifdef RTX5090_OPT
            uint32_t h160[5]; getHash160_33_from_limbs_5090(odd?0x03:0x02, px3, h160);
#else
            uint8_t h20[20]; getHash160_33_from_limbs(odd?0x03:0x02, px3, h20);
#endif
            ++local_hashes; MAYBE_WARP_FLUSH();

#ifdef RTX5090_OPT
            bool pref = h160[0] == target_prefix;
#else
            bool pref = hash160_prefix_equals(h20, target_prefix);
#endif
#ifdef RTX5090_OPT
            const bool full_match = pref && hash160_words_match_5090(h160, target_prefix);
            if (__any_sync(full_mask, full_match)) {
                if (full_match) {
#else
            if (__any_sync(full_mask, pref)) {
                if (pref && hash160_matches_prefix_then_full(h20, c_target_hash160, target_prefix)) {
#endif
                    if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                        uint64_t fs[4]; for (int k=0;k<4;++k) fs[k]=S[k];
                        uint64_t sub=(uint64_t)half;
                        for (int k=0;k<4 && sub;++k){ uint64_t old=fs[k]; fs[k]=old-sub; sub=(old<sub)?1ull:0ull; }
#pragma unroll
                        for (int k=0;k<4;++k) d_found_result->scalar[k]=fs[k];
#pragma unroll
                        for (int k=0;k<4;++k) d_found_result->Rx[k]=px3[k];
                        uint64_t y3[4]; uint64_t t[4]; ModSub256(t, x1, px3); _ModMult(y3, t, lam); ModSub256(y3, y3, y1);
#pragma unroll
                        for (int k=0;k<4;++k) d_found_result->Ry[k]=y3[k];
                        d_found_result->threadId = (int)gid;
                        d_found_result->iter     = 0;
                        __threadfence_system();
                        atomicExch(d_found_flag, FOUND_READY);
                    }
                }
                __syncwarp(full_mask); WARP_FLUSH_HASHES(); return;
            }

            uint64_t last_dx[4];
#pragma unroll
            for (int j=0;j<4;++j) last_dx[j] = c_Gx[(size_t)i*4 + j];
            ModSub256(last_dx, last_dx, x1);
            _ModMult(inverse, inverse, last_dx);
        }

        {
            uint64_t lam[4], s[4], x3[4], y3[4];

            uint64_t Jy_minus_y1[4];
#pragma unroll
            for (int j=0;j<4;++j) Jy_minus_y1[j] = c_Jy[j];
            ModSub256(Jy_minus_y1, Jy_minus_y1, y1);

            _ModMult(lam, Jy_minus_y1, inverse);
            _ModSqr(x3, lam);
            ModSub256(x3, x3, x1);
            uint64_t Jx_local[4]; for (int j=0;j<4;++j) Jx_local[j]=c_Jx[j];
            ModSub256(x3, x3, Jx_local);

            ModSub256(s, x1, x3);
            _ModMult(y3, s, lam);
            ModSub256(y3, y3, y1);

#pragma unroll
            for (int j=0;j<4;++j) { x1[j] = x3[j]; y1[j] = y3[j]; }
        }

        {
            uint64_t addv=(uint64_t)B;
            for (int k=0;k<4 && addv;++k){ uint64_t old=S[k]; S[k]=old+addv; addv=(S[k]<old)?1ull:0ull; }
            sub256_u64_inplace(rem, (uint64_t)B);
        }
        ++batches_done;
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        Rx[gid*4+i] = x1[i];
        Ry[gid*4+i] = y1[i];
        counts256[gid*4+i] = rem[i];
        start_scalars[gid*4+i] = S[i];
    }
    if ((rem[0] | rem[1] | rem[2] | rem[3]) != 0ull) {
        atomicAdd(d_any_left, 1u);
    }

    WARP_FLUSH_HASHES();
    #undef MAYBE_WARP_FLUSH
    #undef WARP_FLUSH_HASHES
    #undef FLUSH_THRESHOLD
}

extern bool hexToLE64(const std::string& h_in, uint64_t w[4]);
extern bool hexToHash160(const std::string& h, uint8_t hash160[20]);
extern std::string formatHex256(const uint64_t limbs[4]);
extern long double ld_from_u256(const uint64_t v[4]);
extern bool decode_p2pkh_address(const std::string& addr, uint8_t out20[20]);
extern std::string formatCompressedPubHex(const uint64_t X[4], const uint64_t Y[4]);
__global__ void scalarMulKernelBase(const uint64_t* scalars_in, uint64_t* outX, uint64_t* outY, int N);

#ifdef RTX5090_OPT
struct HashSelfTestSummary {
    unsigned int pubkey_mismatches;
    unsigned int sha256_mismatches;
    unsigned int ripemd160_mismatches;
    unsigned int match_mismatches;
    unsigned int prefix02_count;
    unsigned int prefix03_count;
    uint32_t first_hash160[5];
};

__global__ void hash_self_test_kernel_5090(
    const uint64_t* __restrict__ x,
    const uint64_t* __restrict__ y,
    int count,
    HashSelfTestSummary* summary)
{
    const int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (gid >= count) return;
    uint64_t lx[4], ly[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        lx[i] = x[(size_t)gid * 4 + i];
        ly[i] = y[(size_t)gid * 4 + i];
    }
    const uint8_t prefix = (ly[0] & 1ull) ? 0x03 : 0x02;
    if (prefix == 0x02) atomicAdd(&summary->prefix02_count, 1u);
    else atomicAdd(&summary->prefix03_count, 1u);

    uint8_t pubkey[33];
    pubkey[0] = prefix;
#pragma unroll
    for (int limb = 0; limb < 4; ++limb) {
        const uint64_t value = lx[3 - limb];
        const int offset = 1 + limb * 8;
#pragma unroll
        for (int byte = 0; byte < 8; ++byte) {
            pubkey[offset + byte] = (uint8_t)(value >> (56 - byte * 8));
        }
    }
    uint8_t pubkey_opt[33];
    pubkey_opt[0] = prefix;
#pragma unroll
    for (int limb = 0; limb < 4; ++limb) storeU64BE(pubkey_opt + 1 + limb * 8, lx[3 - limb]);
    bool pubkey_bad = false;
#pragma unroll
    for (int i = 0; i < 33; ++i) pubkey_bad |= pubkey[i] != pubkey_opt[i];
    if (pubkey_bad) atomicAdd(&summary->pubkey_mismatches, 1u);

    uint8_t sha_ref[32], ripemd_ref[20];
    getSHA256_33bytes(pubkey, sha_ref);
    getRIPEMD160_32bytes(sha_ref, ripemd_ref);

    uint32_t sha_opt[8], ripemd_opt[5];
    getSHA256_33_from_limbs_5090(prefix, lx, sha_opt);
    getHash160_33_from_limbs_5090(prefix, lx, ripemd_opt);

    bool sha_bad = false;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        sha_bad |= sha_ref[4*i + 0] != (uint8_t)(sha_opt[i] >> 24);
        sha_bad |= sha_ref[4*i + 1] != (uint8_t)(sha_opt[i] >> 16);
        sha_bad |= sha_ref[4*i + 2] != (uint8_t)(sha_opt[i] >> 8);
        sha_bad |= sha_ref[4*i + 3] != (uint8_t)sha_opt[i];
    }
    bool ripemd_bad = false;
#pragma unroll
    for (int i = 0; i < 5; ++i) {
        const uint32_t ref = (uint32_t)ripemd_ref[4*i + 0]
                           | ((uint32_t)ripemd_ref[4*i + 1] << 8)
                           | ((uint32_t)ripemd_ref[4*i + 2] << 16)
                           | ((uint32_t)ripemd_ref[4*i + 3] << 24);
        ripemd_bad |= ref != ripemd_opt[i];
    }
    if (sha_bad) atomicAdd(&summary->sha256_mismatches, 1u);
    if (ripemd_bad) atomicAdd(&summary->ripemd160_mismatches, 1u);

    const bool ref_match = load_u32_le(ripemd_ref) == 0xe8761e75u
                        && ripemd_ref[4] == 0x19u;
    const bool opt_match = ripemd_opt[0] == 0xe8761e75u
                        && (ripemd_opt[1] & 0xffu) == 0x19u;
    if (ref_match != opt_match) atomicAdd(&summary->match_mismatches, 1u);
    if (gid == 0) {
#pragma unroll
        for (int i = 0; i < 5; ++i) summary->first_hash160[i] = ripemd_opt[i];
    }
}

static int run_hash_self_test_5090(int count) {
    if (count < 2) count = 2;
    std::vector<uint64_t> scalars((size_t)count * 4, 0ull);
    scalars[0] = 1ull;
    uint64_t rng = 0x71c0ffee5090128ull;
    auto next_u64 = [&]() {
        rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27;
        return rng * 0x2545F4914F6CDD1Dull;
    };
    for (int i = 1; i < count; ++i) {
        scalars[(size_t)i*4 + 0] = next_u64();
        scalars[(size_t)i*4 + 1] = next_u64() & 0x7full;
    }

    uint64_t *d_scalars = nullptr, *d_x = nullptr, *d_y = nullptr;
    HashSelfTestSummary *d_summary = nullptr, summary{};
    auto check = [](cudaError_t error, const char* where) {
        if (error != cudaSuccess) {
            std::cerr << "Self-test CUDA error at " << where << ": " << cudaGetErrorString(error) << "\n";
            std::exit(EXIT_FAILURE);
        }
    };
    const size_t bytes = scalars.size() * sizeof(uint64_t);
    check(cudaMalloc(&d_scalars, bytes), "cudaMalloc scalars");
    check(cudaMalloc(&d_x, bytes), "cudaMalloc x");
    check(cudaMalloc(&d_y, bytes), "cudaMalloc y");
    check(cudaMalloc(&d_summary, sizeof(summary)), "cudaMalloc summary");
    check(cudaMemcpy(d_scalars, scalars.data(), bytes, cudaMemcpyHostToDevice), "copy scalars");
    check(cudaMemset(d_summary, 0, sizeof(summary)), "clear summary");
    const int threads = 256;
    scalarMulKernelBase<<<(count + threads - 1) / threads, threads>>>(d_scalars, d_x, d_y, count);
    check(cudaDeviceSynchronize(), "scalar multiplication");
    hash_self_test_kernel_5090<<<(count + threads - 1) / threads, threads>>>(d_x, d_y, count, d_summary);
    check(cudaDeviceSynchronize(), "hash comparison");
    check(cudaMemcpy(&summary, d_summary, sizeof(summary), cudaMemcpyDeviceToHost), "copy summary");
    uint64_t first_x[4], first_y[4];
    check(cudaMemcpy(first_x, d_x, sizeof(first_x), cudaMemcpyDeviceToHost), "copy first x");
    check(cudaMemcpy(first_y, d_y, sizeof(first_y), cudaMemcpyDeviceToHost), "copy first y");
    cudaFree(d_scalars); cudaFree(d_x); cudaFree(d_y); cudaFree(d_summary);

    const uint32_t expected_hash[5] = {0xe8761e75u, 0xd4969119u, 0x451c9454u, 0x23a3b3d1u, 0xd63b43f1u};
    bool known_hash_ok = true;
    for (int i = 0; i < 5; ++i) known_hash_ok &= summary.first_hash160[i] == expected_hash[i];
    const std::string expected_pub = "0279BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798";
    const bool known_pub_ok = formatCompressedPubHex(first_x, first_y) == expected_pub;
    const bool ok = known_pub_ok && known_hash_ok
                 && summary.pubkey_mismatches == 0
                 && summary.sha256_mismatches == 0
                 && summary.ripemd160_mismatches == 0
                 && summary.match_mismatches == 0
                 && summary.prefix02_count != 0
                 && summary.prefix03_count != 0;
    std::cout << "HASH160 self-test: " << (ok ? "PASS" : "FAIL") << "\n"
              << "Private keys       : " << count << "\n"
              << "Prefix 02 / 03     : " << summary.prefix02_count << " / " << summary.prefix03_count << "\n"
              << "Pubkey differences : " << summary.pubkey_mismatches << "\n"
              << "SHA256 differences : " << summary.sha256_mismatches << "\n"
              << "RIPEMD differences : " << summary.ripemd160_mismatches << "\n"
              << "Match differences  : " << summary.match_mismatches << "\n"
              << "Known pubkey/hash  : " << (known_pub_ok ? "PASS" : "FAIL")
              << " / " << (known_hash_ok ? "PASS" : "FAIL") << "\n";
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

struct RandomBlockOptions5090 {
    bool enabled = false;
    bool child = false;
    bool seed_given = false;
    uint64_t seed = 0;
    unsigned int block_bits = 37;
    std::string checkpoint = "cudacyclone-p71.checkpoint";
};

static bool write_checkpoint_atomic_5090(
    const std::string& path,
    const std::string& range_start,
    const std::string& range_end,
    unsigned int block_bits,
    uint64_t seed,
    uint64_t next_counter)
{
    std::ostringstream contents;
    contents << "version=1\n"
             << "range_start=" << range_start << "\n"
             << "range_end=" << range_end << "\n"
             << "block_bits=" << block_bits << "\n"
             << "seed=" << seed << "\n"
             << "next_counter=" << next_counter << "\n";
    const std::string data = contents.str();
    const std::string temporary = path + ".tmp." + std::to_string((unsigned long long)getpid());
    const int fd = open(temporary.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) return false;
    size_t written = 0;
    while (written < data.size()) {
        const ssize_t amount = write(fd, data.data() + written, data.size() - written);
        if (amount <= 0) { close(fd); unlink(temporary.c_str()); return false; }
        written += (size_t)amount;
    }
    if (fsync(fd) != 0 || close(fd) != 0) { unlink(temporary.c_str()); return false; }
    if (rename(temporary.c_str(), path.c_str()) != 0) { unlink(temporary.c_str()); return false; }
    std::filesystem::path checkpoint_path(path);
    std::filesystem::path parent = checkpoint_path.parent_path();
    if (parent.empty()) parent = ".";
    const int directory_fd = open(parent.c_str(), O_RDONLY | O_DIRECTORY);
    if (directory_fd >= 0) { (void)fsync(directory_fd); close(directory_fd); }
    return true;
}

static bool read_checkpoint_5090(const std::string& path, std::map<std::string,std::string>& values) {
    std::ifstream input(path);
    if (!input) return false;
    std::string line;
    while (std::getline(input, line)) {
        const size_t equals = line.find('=');
        if (equals != std::string::npos) values[line.substr(0, equals)] = line.substr(equals + 1);
    }
    return values["version"] == "1";
}

static void shifted_u64_256_5090(uint64_t value, unsigned int shift, uint64_t out[4]) {
    out[0] = out[1] = out[2] = out[3] = 0ull;
    const unsigned int word = shift >> 6;
    const unsigned int bits = shift & 63u;
    if (word < 4) out[word] = value << bits;
    if (bits && word + 1 < 4) out[word + 1] = value >> (64u - bits);
}

static int run_random_blocks_5090(
    const char* executable,
    const std::string& canonical_start,
    const std::string& canonical_end,
    const uint64_t range_start[4],
    const uint64_t range_len[4],
    const std::string& target_hash_hex,
    const std::string& address_b58,
    uint32_t batch,
    uint32_t batches_per_sm,
    uint32_t slices,
    RandomBlockOptions5090 options)
{
    int range_bits = -1;
    for (int limb = 0; limb < 4; ++limb) {
        if (!range_len[limb]) continue;
        if ((range_len[limb] & (range_len[limb] - 1ull)) != 0ull || range_bits != -1) {
            std::cerr << "Error: random-block range length must be a power of two.\n";
            return EXIT_FAILURE;
        }
        range_bits = limb * 64 + __builtin_ctzll(range_len[limb]);
    }
    if (range_bits < 0 || options.block_bits > (unsigned int)range_bits || options.block_bits < 20) {
        std::cerr << "Error: --block-bits must be in 20..range_bits (" << range_bits << ").\n";
        return EXIT_FAILURE;
    }
    const unsigned int index_bits = (unsigned int)range_bits - options.block_bits;
    if (index_bits >= 64) {
        std::cerr << "Error: random-block index does not fit in uint64.\n";
        return EXIT_FAILURE;
    }
    const uint64_t total_blocks = index_bits == 0 ? 1ull : (1ull << index_bits);
    const uint64_t index_mask = total_blocks - 1ull;
    uint64_t next_counter = 0ull;

    std::map<std::string,std::string> saved;
    if (read_checkpoint_5090(options.checkpoint, saved)) {
        try {
            if (saved["range_start"] != canonical_start || saved["range_end"] != canonical_end
                || std::stoul(saved["block_bits"]) != options.block_bits) {
                std::cerr << "Error: checkpoint range or block size does not match this invocation.\n";
                return EXIT_FAILURE;
            }
            const uint64_t saved_seed = std::stoull(saved["seed"]);
            if (options.seed_given && options.seed != saved_seed) {
                std::cerr << "Error: --seed does not match checkpoint seed.\n";
                return EXIT_FAILURE;
            }
            options.seed = saved_seed;
            next_counter = std::stoull(saved["next_counter"]);
        } catch (...) {
            std::cerr << "Error: invalid checkpoint contents.\n";
            return EXIT_FAILURE;
        }
    } else {
        if (std::filesystem::exists(options.checkpoint)) {
            std::cerr << "Error: checkpoint exists but is invalid.\n";
            return EXIT_FAILURE;
        }
        if (!options.seed_given) {
            options.seed = (uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count()
                         ^ ((uint64_t)getpid() << 32);
        }
        if (!write_checkpoint_atomic_5090(options.checkpoint, canonical_start, canonical_end,
                                          options.block_bits, options.seed, next_counter)) {
            std::cerr << "Error: cannot create checkpoint " << options.checkpoint << ": " << std::strerror(errno) << "\n";
            return EXIT_FAILURE;
        }
    }
    if (next_counter > total_blocks) {
        std::cerr << "Error: checkpoint counter exceeds total block count.\n";
        return EXIT_FAILURE;
    }

    uint64_t mix = options.seed;
    mix ^= mix >> 30; mix *= 0xbf58476d1ce4e5b9ull;
    mix ^= mix >> 27; mix *= 0x94d049bb133111ebull;
    mix ^= mix >> 31;
    const uint64_t multiplier = mix | 1ull;
    mix += 0x9e3779b97f4a7c15ull;
    mix ^= mix >> 30; mix *= 0xbf58476d1ce4e5b9ull;
    mix ^= mix >> 27; mix *= 0x94d049bb133111ebull;
    mix ^= mix >> 31;
    const uint64_t addend = mix;

    std::cout << "Random blocks mode  : enabled\n"
              << "Seed                : " << options.seed << "\n"
              << "Block bits          : " << options.block_bits << "\n"
              << "Total blocks        : " << total_blocks << "\n"
              << "Resume counter      : " << next_counter << "\n"
              << "Checkpoint          : " << options.checkpoint << "\n";

    for (uint64_t counter = next_counter; counter < total_blocks; ++counter) {
        const uint64_t block_index = (multiplier * counter + addend) & index_mask;
        uint64_t offset[4], block_start[4], block_end[4], block_mask[4];
        shifted_u64_256_5090(block_index, options.block_bits, offset);
        add256(range_start, offset, block_start);
        shifted_u64_256_5090(1ull, options.block_bits, block_mask);
        uint64_t borrow = 1ull;
        for (int i = 0; i < 4; ++i) {
            const uint64_t old = block_mask[i];
            block_mask[i] = old - borrow;
            borrow = old < borrow ? 1ull : 0ull;
        }
        add256(block_start, block_mask, block_end);
        const std::string child_range = formatHex256(block_start) + ":" + formatHex256(block_end);
        std::cout << "\n[block " << (counter + 1) << "/" << total_blocks << "] permutation index "
                  << block_index << " range " << child_range << "\n";
        std::cout.flush();

        std::vector<std::string> child_args = {
            executable, "--range", child_range, "--grid",
            std::to_string(batch) + "," + std::to_string(batches_per_sm),
            "--slices", std::to_string(slices), "--random-child"
        };
        if (!target_hash_hex.empty()) { child_args.push_back("--target-hash160"); child_args.push_back(target_hash_hex); }
        else { child_args.push_back("--address"); child_args.push_back(address_b58); }
        std::vector<char*> child_argv;
        for (std::string& argument : child_args) child_argv.push_back(argument.data());
        child_argv.push_back(nullptr);

        const pid_t child = fork();
        if (child == 0) { execv(executable, child_argv.data()); _exit(127); }
        if (child < 0) { std::cerr << "Error: fork failed.\n"; return EXIT_FAILURE; }
        int status = 0;
        while (waitpid(child, &status, 0) < 0 && errno == EINTR) {}
        if (!WIFEXITED(status)) return 130;
        const int child_status = WEXITSTATUS(status);
        if (child_status == 10) {
            std::cout << "Match found; current block remains uncommitted in the checkpoint.\n";
            return EXIT_SUCCESS;
        }
        if (child_status != 0) {
            std::cerr << "Block interrupted or failed (exit " << child_status << "); checkpoint not advanced.\n";
            return child_status;
        }
        if (!write_checkpoint_atomic_5090(options.checkpoint, canonical_start, canonical_end,
                                          options.block_bits, options.seed, counter + 1ull)) {
            std::cerr << "Error: cannot advance checkpoint atomically.\n";
            return EXIT_FAILURE;
        }
    }
    std::cout << "All random-order blocks completed exactly once.\n";
    return EXIT_SUCCESS;
}

static std::string decimal_u128_5090(unsigned __int128 value) {
    if (value == 0) return "0";
    std::string result;
    while (value) {
        result.push_back((char)('0' + value % 10));
        value /= 10;
    }
    std::reverse(result.begin(), result.end());
    return result;
}
#endif

int main(int argc, char** argv) {
    std::signal(SIGINT, handle_sigint);

    std::string target_hash_hex, range_hex, address_b58;
    uint32_t runtime_points_batch_size =
#ifdef RTX5090_OPT
        RTX5090_BATCH;
#else
        128;
#endif
    uint32_t runtime_batches_per_sm    = 8;
    uint32_t slices_per_launch         =
#ifdef RTX5090_OPT
        16;
#else
        64;
#endif
#ifdef RTX5090_OPT
    int self_test_count = 0;
    RandomBlockOptions5090 random_blocks;
#endif

    auto parse_grid = [](const std::string& s, uint32_t& a_out, uint32_t& b_out)->bool {
        size_t comma = s.find(',');
        if (comma == std::string::npos) return false;
        auto trim = [](std::string& z){
            size_t p1 = z.find_first_not_of(" \t");
            size_t p2 = z.find_last_not_of(" \t");
            if (p1 == std::string::npos) { z.clear(); return; }
            z = z.substr(p1, p2 - p1 + 1);
        };
        std::string a_str = s.substr(0, comma);
        std::string b_str = s.substr(comma + 1);
        trim(a_str); trim(b_str);
        if (a_str.empty() || b_str.empty()) return false;
        char* endp=nullptr;
        unsigned long aa = std::strtoul(a_str.c_str(), &endp, 10); if (*endp) return false;
        endp=nullptr;
        unsigned long bb = std::strtoul(b_str.c_str(), &endp, 10); if (*endp) return false;
        if (aa == 0ul || bb == 0ul) return false;
        if (aa > (1ul<<20) || bb > (1ul<<20)) return false;
        a_out=(uint32_t)aa; b_out=(uint32_t)bb; return true;
    };

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--target-hash160" && i + 1 < argc) target_hash_hex = argv[++i];
        else if (arg == "--address"        && i + 1 < argc) address_b58     = argv[++i];
        else if (arg == "--range"          && i + 1 < argc) range_hex       = argv[++i];
        else if (arg == "--grid"           && i + 1 < argc) {
            uint32_t a=0,b=0;
            if (!parse_grid(argv[++i], a, b)) {
                std::cerr << "Error: --grid expects \"A,B\" (positive integers).\n";
                return EXIT_FAILURE;
            }
            runtime_points_batch_size = a;
            runtime_batches_per_sm    = b;
        }
        else if (arg == "--slices" && i + 1 < argc) {
            char* endp=nullptr;
            unsigned long v = std::strtoul(argv[++i], &endp, 10);
            if (*endp != '\0' || v == 0ul || v > (1ul<<20)) {
                std::cerr << "Error: --slices must be in 1.." << (1u<<20) << "\n";
                return EXIT_FAILURE;
            }
            slices_per_launch = (uint32_t)v;
        }
#ifdef RTX5090_OPT
        else if (arg == "--self-test" && i + 1 < argc) {
            char* endp = nullptr;
            long v = std::strtol(argv[++i], &endp, 10);
            if (*endp != '\0' || v < 2 || v > 1000000) {
                std::cerr << "Error: --self-test must be in 2..1000000\n";
                return EXIT_FAILURE;
            }
            self_test_count = (int)v;
        }
        else if (arg == "--random-blocks") random_blocks.enabled = true;
        else if (arg == "--random-child") random_blocks.child = true;
        else if (arg == "--seed" && i + 1 < argc) {
            char* endp = nullptr;
            random_blocks.seed = std::strtoull(argv[++i], &endp, 0);
            if (*endp != '\0') { std::cerr << "Error: invalid --seed.\n"; return EXIT_FAILURE; }
            random_blocks.seed_given = true;
        }
        else if (arg == "--checkpoint" && i + 1 < argc) random_blocks.checkpoint = argv[++i];
        else if (arg == "--block-bits" && i + 1 < argc) {
            char* endp = nullptr;
            unsigned long value = std::strtoul(argv[++i], &endp, 10);
            if (*endp != '\0' || value > 255) { std::cerr << "Error: invalid --block-bits.\n"; return EXIT_FAILURE; }
            random_blocks.block_bits = (unsigned int)value;
        }
#endif
    }

#ifdef RTX5090_OPT
    if (self_test_count) return run_hash_self_test_5090(self_test_count);
#endif

    if (range_hex.empty() || (target_hash_hex.empty() && address_b58.empty())) {
        std::cerr << "Usage: " << argv[0]
                  << " --range <start_hex>:<end_hex> (--address <base58> | --target-hash160 <hash160_hex>) [--grid A,B] [--slices N]\n";
        return EXIT_FAILURE;
    }
    if (!target_hash_hex.empty() && !address_b58.empty()) {
        std::cerr << "Error: provide either --address or --target-hash160, not both.\n";
        return EXIT_FAILURE;
    }

    size_t colon_pos = range_hex.find(':');
    if (colon_pos == std::string::npos) { std::cerr << "Error: range format must be start:end\n"; return EXIT_FAILURE; }
    std::string start_hex = range_hex.substr(0, colon_pos);
    std::string end_hex   = range_hex.substr(colon_pos + 1);

    uint64_t range_start[4]{0}, range_end[4]{0};
    if (!hexToLE64(start_hex, range_start) || !hexToLE64(end_hex, range_end)) {
        std::cerr << "Error: invalid range hex\n"; return EXIT_FAILURE;
    }

    uint8_t target_hash160[20];
    if (!address_b58.empty()) {
        if (!decode_p2pkh_address(address_b58, target_hash160)) {
            std::cerr << "Error: invalid P2PKH address\n"; return EXIT_FAILURE;
        }
    } else {
        if (!hexToHash160(target_hash_hex, target_hash160)) {
            std::cerr << "Error: invalid target hash160 hex\n"; return EXIT_FAILURE;
        }
    }

    auto is_pow2 = [](uint32_t v)->bool { return v && ((v & (v-1)) == 0); };
#ifdef RTX5090_OPT
    if (runtime_points_batch_size != RTX5090_BATCH) {
        std::cerr << "Error: CUDACyclone-5090 was compiled for batch size " << RTX5090_BATCH << ".\n";
        return EXIT_FAILURE;
    }
#endif
    if (!is_pow2(runtime_points_batch_size) || (runtime_points_batch_size & 1u)) {
        std::cerr << "Error: batch size must be even and a power of two.\n";
        return EXIT_FAILURE;
    }
    if (runtime_points_batch_size > MAX_BATCH_SIZE) {
        std::cerr << "Error: batch size must be <= " << MAX_BATCH_SIZE << " (kernel limit).\n";
        return EXIT_FAILURE;
    }

    uint64_t range_len[4]; sub256(range_end, range_start, range_len); add256_u64(range_len, 1ull, range_len);

    auto is_zero_256 = [](const uint64_t a[4])->bool { return (a[0]|a[1]|a[2]|a[3]) == 0ull; };
    auto is_power_of_two_256 = [&](const uint64_t a[4])->bool {
        if (is_zero_256(a)) return false;
        uint64_t am1[4]; uint64_t borrow = 1ull;
        for (int i=0;i<4;++i) {
            uint64_t v = a[i] - borrow; borrow = (a[i] < borrow) ? 1ull : 0ull; am1[i] = v;
            if (!borrow && i+1<4) { for (int k=i+1;k<4;++k) am1[k] = a[k]; break; }
        }
        uint64_t and0=a[0]&am1[0], and1=a[1]&am1[1], and2=a[2]&am1[2], and3=a[3]&am1[3];
        return (and0|and1|and2|and3)==0ull;
    };
    if (!is_power_of_two_256(range_len)) {
        std::cerr << "Error: range length (end - start + 1) must be a power of two.\n"; return EXIT_FAILURE;
    }
    uint64_t len_minus1[4];
    {   uint64_t borrow=1ull;
        for (int i=0;i<4;++i) {
            uint64_t v=range_len[i]-borrow; borrow=(range_len[i]<borrow)?1ull:0ull; len_minus1[i]=v;
            if (!borrow && i+1<4) { for (int k=i+1;k<4;++k) len_minus1[k]=range_len[k]; break; }
        }
    }
    {   uint64_t and0 = range_start[0] & len_minus1[0];
        uint64_t and1 = range_start[1] & len_minus1[1];
        uint64_t and2 = range_start[2] & len_minus1[2];
        uint64_t and3 = range_start[3] & len_minus1[3];
        if ((and0|and1|and2|and3) != 0ull) {
            std::cerr << "Error: start must be aligned to the range length.\n"; return EXIT_FAILURE;
        }
    }

#ifdef RTX5090_OPT
    if (random_blocks.enabled) {
        return run_random_blocks_5090(argv[0], formatHex256(range_start), formatHex256(range_end),
                                      range_start, range_len, target_hash_hex, address_b58,
                                      runtime_points_batch_size, runtime_batches_per_sm,
                                      slices_per_launch, random_blocks);
    }
#endif

    int device=0; cudaDeviceProp prop{};
    if (cudaGetDevice(&device)!=cudaSuccess || cudaGetDeviceProperties(&prop, device)!=cudaSuccess) {
        std::cerr<<"CUDA init error\n"; return EXIT_FAILURE;
    }

    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

#ifdef RTX5090_OPT
    int threadsPerBlock=RTX5090_TPB;
#else
    int threadsPerBlock=256;
#endif
    if (threadsPerBlock > (int)prop.maxThreadsPerBlock) threadsPerBlock=prop.maxThreadsPerBlock;
    if (threadsPerBlock < 32) threadsPerBlock=32;
#ifdef RTX5090_OPT
    const int scalarThreadsPerBlock = threadsPerBlock > 256 ? 256 : threadsPerBlock;
#endif

    const uint64_t bytesPerThread = 2ull*4ull*sizeof(uint64_t);
    size_t totalGlobalMem = prop.totalGlobalMem;
    const uint64_t reserveBytes = 64ull * 1024 * 1024;
    uint64_t usableMem = (totalGlobalMem > reserveBytes) ? (totalGlobalMem - reserveBytes) : (totalGlobalMem / 2);
    uint64_t maxThreadsByMem = usableMem / bytesPerThread;

    uint64_t q_div_batch[4], r_div_batch = 0ull;
    divmod_256_by_u64(range_len, (uint64_t)runtime_points_batch_size, q_div_batch, r_div_batch);
    if (r_div_batch != 0ull) {
        std::cerr << "Error: range length must be divisible by batch size (" << runtime_points_batch_size << ").\n";
        return EXIT_FAILURE;
    }
    bool q_fits_u64 = (q_div_batch[3]|q_div_batch[2]|q_div_batch[1]) == 0ull;
    uint64_t total_batches_u64 = q_fits_u64 ? q_div_batch[0] : 0ull;
    if (!q_fits_u64) { std::cerr << "Error: total batches too large for u64.\n"; return EXIT_FAILURE; }

    uint64_t userUpper = (uint64_t)prop.multiProcessorCount * (uint64_t)runtime_batches_per_sm * (uint64_t)threadsPerBlock;
    if (userUpper == 0ull) userUpper = UINT64_MAX;

    auto pick_threads_total = [&](uint64_t upper)->uint64_t {
        if (upper < (uint64_t)threadsPerBlock) return 0ull;
        uint64_t t = upper - (upper % (uint64_t)threadsPerBlock);
        uint64_t q = total_batches_u64;
        while (t >= (uint64_t)threadsPerBlock) {
            if ((q % t) == 0ull) return t;
            t -= (uint64_t)threadsPerBlock;
        }
        return 0ull;
    };

    uint64_t upper = maxThreadsByMem;
    if (total_batches_u64 < upper) upper = total_batches_u64;
    if (userUpper         < upper) upper = userUpper;

    uint64_t threadsTotal = pick_threads_total(upper);
#ifdef RTX5090_OPT
    bool uneven_partition = false;
    if (threadsTotal == 0ull) {
        threadsTotal = upper - (upper % (uint64_t)threadsPerBlock);
        uneven_partition = true;
    }
#endif
    if (threadsTotal == 0ull) {
        std::cerr << "Error: failed to pick threadsTotal satisfying divisibility.\n";
        return EXIT_FAILURE;
    }
    int blocks = (int)(threadsTotal / (uint64_t)threadsPerBlock);

    uint64_t per_thread_cnt[4]{0,0,0,0}; uint64_t r_u64 = 0ull;
#ifdef RTX5090_OPT
    uint64_t base_batches_per_thread = 0ull;
    uint64_t extra_batch_threads = 0ull;
    if (uneven_partition) {
        base_batches_per_thread = total_batches_u64 / threadsTotal;
        extra_batch_threads = total_batches_u64 % threadsTotal;
    } else
#endif
    {
    divmod_256_by_u64(range_len, threadsTotal, per_thread_cnt, r_u64);
    if (r_u64 != 0ull) { std::cerr << "Internal error: range_len not divisible by threadsTotal.\n"; return EXIT_FAILURE; }
    {   uint64_t qq[4], rr=0ull;
        divmod_256_by_u64(per_thread_cnt, (uint64_t)runtime_points_batch_size, qq, rr);
        if (rr != 0ull) { std::cerr << "Internal error: per-thread count is not a multiple of batch size.\n"; return EXIT_FAILURE; }
    }
    }

    uint64_t* h_counts256     = nullptr;
    uint64_t* h_start_scalars = nullptr;
    cudaHostAlloc(&h_counts256,     threadsTotal * 4 * sizeof(uint64_t), cudaHostAllocWriteCombined | cudaHostAllocMapped);
    cudaHostAlloc(&h_start_scalars, threadsTotal * 4 * sizeof(uint64_t), cudaHostAllocWriteCombined | cudaHostAllocMapped);

    const uint32_t B = runtime_points_batch_size;
    for (uint64_t i = 0; i < threadsTotal; ++i) {
#ifdef RTX5090_OPT
        if (uneven_partition) {
            const uint64_t batches = base_batches_per_thread + (i < extra_batch_threads ? 1ull : 0ull);
            h_counts256[i*4+0] = batches * (uint64_t)B;
            h_counts256[i*4+1] = 0ull;
            h_counts256[i*4+2] = 0ull;
            h_counts256[i*4+3] = 0ull;
            continue;
        }
#endif
        h_counts256[i*4+0] = per_thread_cnt[0];
        h_counts256[i*4+1] = per_thread_cnt[1];
        h_counts256[i*4+2] = per_thread_cnt[2];
        h_counts256[i*4+3] = per_thread_cnt[3];
    }

    const uint32_t half = B >> 1;
    {
        uint64_t cur[4] = { range_start[0], range_start[1], range_start[2], range_start[3] };
        for (uint64_t i = 0; i < threadsTotal; ++i) {
            uint64_t Sc[4]; add256_u64(cur, (uint64_t)half, Sc); 
            h_start_scalars[i*4+0] = Sc[0];
            h_start_scalars[i*4+1] = Sc[1];
            h_start_scalars[i*4+2] = Sc[2];
            h_start_scalars[i*4+3] = Sc[3];

#ifdef RTX5090_OPT
            uint64_t next[4];
            if (uneven_partition) add256_u64(cur, h_counts256[i*4+0], next);
            else add256(cur, per_thread_cnt, next);
#else
            uint64_t next[4]; add256(cur, per_thread_cnt, next);
#endif
            cur[0]=next[0]; cur[1]=next[1]; cur[2]=next[2]; cur[3]=next[3];
        }
    }

    {
        uint32_t prefix_le = (uint32_t)target_hash160[0]
                           | ((uint32_t)target_hash160[1] << 8)
                           | ((uint32_t)target_hash160[2] << 16)
                           | ((uint32_t)target_hash160[3] << 24);
        cudaMemcpyToSymbol(c_target_prefix, &prefix_le, sizeof(prefix_le));
        cudaMemcpyToSymbol(c_target_hash160, target_hash160, 20);
#ifdef RTX5090_OPT
        uint32_t target_words[5];
        for (int i = 0; i < 5; ++i) {
            target_words[i] = (uint32_t)target_hash160[4*i + 0]
                            | ((uint32_t)target_hash160[4*i + 1] << 8)
                            | ((uint32_t)target_hash160[4*i + 2] << 16)
                            | ((uint32_t)target_hash160[4*i + 3] << 24);
        }
        cudaMemcpyToSymbol(c_target_hash160_words, target_words, sizeof(target_words));
#endif
    }

    uint64_t *d_start_scalars=nullptr, *d_Px=nullptr, *d_Py=nullptr, *d_Rx=nullptr, *d_Ry=nullptr, *d_counts256=nullptr;
    int *d_found_flag=nullptr; FoundResult *d_found_result=nullptr;
    unsigned long long *d_hashes_accum=nullptr; unsigned int *d_any_left=nullptr;

    auto ck = [](cudaError_t e, const char* msg){
        if (e != cudaSuccess) {
            std::cerr << msg << ": " << cudaGetErrorString(e) << "\n";
            std::exit(EXIT_FAILURE);
        }
    };

    ck(cudaMalloc(&d_start_scalars, threadsTotal * 4 * sizeof(uint64_t)), "cudaMalloc(d_start_scalars)");
    ck(cudaMalloc(&d_Px,           threadsTotal * 4 * sizeof(uint64_t)), "cudaMalloc(d_Px)");
    ck(cudaMalloc(&d_Py,           threadsTotal * 4 * sizeof(uint64_t)), "cudaMalloc(d_Py)");
    ck(cudaMalloc(&d_Rx,           threadsTotal * 4 * sizeof(uint64_t)), "cudaMalloc(d_Rx)");
    ck(cudaMalloc(&d_Ry,           threadsTotal * 4 * sizeof(uint64_t)), "cudaMalloc(d_Ry)");
    ck(cudaMalloc(&d_counts256,    threadsTotal * 4 * sizeof(uint64_t)), "cudaMalloc(d_counts256)");
    ck(cudaMalloc(&d_found_flag,   sizeof(int)),                         "cudaMalloc(d_found_flag)");
    ck(cudaMalloc(&d_found_result, sizeof(FoundResult)),                 "cudaMalloc(d_found_result)");
    ck(cudaMalloc(&d_hashes_accum, sizeof(unsigned long long)),          "cudaMalloc(d_hashes_accum)");
    ck(cudaMalloc(&d_any_left,     sizeof(unsigned int)),                "cudaMalloc(d_any_left)");

    ck(cudaMemcpy(d_start_scalars, h_start_scalars, threadsTotal * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice), "cpy start_scalars");
    ck(cudaMemcpy(d_counts256,     h_counts256,     threadsTotal * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice), "cpy counts256");
    { int zero = FOUND_NONE; unsigned long long zero64=0ull;
      ck(cudaMemcpy(d_found_flag, &zero,   sizeof(int),                cudaMemcpyHostToDevice), "init found_flag");
      ck(cudaMemcpy(d_hashes_accum, &zero64, sizeof(unsigned long long), cudaMemcpyHostToDevice), "init hashes_accum"); }

    {
#ifdef RTX5090_OPT
        int blocks_scal = (int)((threadsTotal + scalarThreadsPerBlock - 1) / scalarThreadsPerBlock);
        scalarMulKernelBase<<<blocks_scal, scalarThreadsPerBlock>>>(d_start_scalars, d_Px, d_Py, (int)threadsTotal);
#else
        int blocks_scal = (int)((threadsTotal + threadsPerBlock - 1) / threadsPerBlock);
        scalarMulKernelBase<<<blocks_scal, threadsPerBlock>>>(d_start_scalars, d_Px, d_Py, (int)threadsTotal);
#endif
        ck(cudaDeviceSynchronize(), "scalarMulKernelBase sync");
        ck(cudaGetLastError(), "scalarMulKernelBase launch");
    }

    {
        uint64_t* h_scalars_half = nullptr;
        cudaHostAlloc(&h_scalars_half, (size_t)half * 4 * sizeof(uint64_t), cudaHostAllocWriteCombined | cudaHostAllocMapped);
        std::memset(h_scalars_half, 0, (size_t)half * 4 * sizeof(uint64_t));
        for (uint32_t k = 0; k < half; ++k) h_scalars_half[(size_t)k*4 + 0] = (uint64_t)(k + 1);

        uint64_t *d_scalars_half=nullptr, *d_Gx_half=nullptr, *d_Gy_half=nullptr;
        ck(cudaMalloc(&d_scalars_half, (size_t)half * 4 * sizeof(uint64_t)), "cudaMalloc(d_scalars_half)");
        ck(cudaMalloc(&d_Gx_half,      (size_t)half * 4 * sizeof(uint64_t)), "cudaMalloc(d_Gx_half)");
        ck(cudaMalloc(&d_Gy_half,      (size_t)half * 4 * sizeof(uint64_t)), "cudaMalloc(d_Gy_half)");
        ck(cudaMemcpy(d_scalars_half, h_scalars_half, (size_t)half * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice), "cpy half scalars");

#ifdef RTX5090_OPT
        int blocks_scal = (int)((half + scalarThreadsPerBlock - 1) / scalarThreadsPerBlock);
        scalarMulKernelBase<<<blocks_scal, scalarThreadsPerBlock>>>(d_scalars_half, d_Gx_half, d_Gy_half, (int)half);
#else
        int blocks_scal = (int)((half + threadsPerBlock - 1) / threadsPerBlock);
        scalarMulKernelBase<<<blocks_scal, threadsPerBlock>>>(d_scalars_half, d_Gx_half, d_Gy_half, (int)half);
#endif
        ck(cudaDeviceSynchronize(), "scalarMulKernelBase(half) sync");
        ck(cudaGetLastError(), "scalarMulKernelBase(half) launch");

        uint64_t* h_Gx_half = (uint64_t*)std::malloc((size_t)half * 4 * sizeof(uint64_t));
        uint64_t* h_Gy_half = (uint64_t*)std::malloc((size_t)half * 4 * sizeof(uint64_t));
        ck(cudaMemcpy(h_Gx_half, d_Gx_half, (size_t)half * 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost), "D2H Gx_half");
        ck(cudaMemcpy(h_Gy_half, d_Gy_half, (size_t)half * 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost), "D2H Gy_half");
        ck(cudaMemcpyToSymbol(c_Gx, h_Gx_half, (size_t)half * 4 * sizeof(uint64_t)), "ToSymbol c_Gx");
        ck(cudaMemcpyToSymbol(c_Gy, h_Gy_half, (size_t)half * 4 * sizeof(uint64_t)), "ToSymbol c_Gy");

        cudaFree(d_scalars_half); cudaFree(d_Gx_half); cudaFree(d_Gy_half);
        cudaFreeHost(h_scalars_half);
        std::free(h_Gx_half); std::free(h_Gy_half);
    }
    {
        uint64_t* h_scalarB = nullptr;
        cudaHostAlloc(&h_scalarB, 4 * sizeof(uint64_t), cudaHostAllocWriteCombined | cudaHostAllocMapped);
        std::memset(h_scalarB, 0, 4 * sizeof(uint64_t));
        h_scalarB[0] = (uint64_t)B;

        uint64_t *d_scalarB=nullptr, *d_Jx=nullptr, *d_Jy=nullptr;
        ck(cudaMalloc(&d_scalarB, 4 * sizeof(uint64_t)), "cudaMalloc(d_scalarB)");
        ck(cudaMalloc(&d_Jx,      4 * sizeof(uint64_t)), "cudaMalloc(d_Jx)");
        ck(cudaMalloc(&d_Jy,      4 * sizeof(uint64_t)), "cudaMalloc(d_Jy)");
        ck(cudaMemcpy(d_scalarB, h_scalarB, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice), "cpy scalarB");

        scalarMulKernelBase<<<1, 1>>>(d_scalarB, d_Jx, d_Jy, 1);
        ck(cudaDeviceSynchronize(), "scalarMulKernelBase(B) sync");
        ck(cudaGetLastError(), "scalarMulKernelBase(B) launch");

        uint64_t hJx[4], hJy[4];
        ck(cudaMemcpy(hJx, d_Jx, 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost), "D2H Jx");
        ck(cudaMemcpy(hJy, d_Jy, 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost), "D2H Jy");
        ck(cudaMemcpyToSymbol(c_Jx, hJx, 4 * sizeof(uint64_t)), "ToSymbol c_Jx");
        ck(cudaMemcpyToSymbol(c_Jy, hJy, 4 * sizeof(uint64_t)), "ToSymbol c_Jy");

        cudaFree(d_scalarB); cudaFree(d_Jx); cudaFree(d_Jy);
        cudaFreeHost(h_scalarB);
    }

    size_t freeB=0,totalB=0; cudaMemGetInfo(&freeB,&totalB);
    size_t usedB = totalB - freeB;
    double util = totalB ? (double)usedB * 100.0 / (double)totalB : 0.0;
#ifdef RTX5090_OPT
    int activeBlocksPerSm = 0;
    ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
           &activeBlocksPerSm, kernel_point_add_and_check_oneinv<RTX5090_BATCH>, threadsPerBlock, 0),
       "cudaOccupancyMaxActiveBlocksPerMultiprocessor");
#endif

    std::cout << "======== PrePhase: GPU Information ====================\n";
    std::cout << std::left << std::setw(20) << "Device"            << " : " << prop.name << " (compute " << prop.major << "." << prop.minor << ")\n";
    std::cout << std::left << std::setw(20) << "SM"                << " : " << prop.multiProcessorCount << "\n";
    std::cout << std::left << std::setw(20) << "ThreadsPerBlock"   << " : " << threadsPerBlock << "\n";
    std::cout << std::left << std::setw(20) << "Blocks"            << " : " << (int)(threadsTotal / (uint64_t)threadsPerBlock) << "\n";
    std::cout << std::left << std::setw(20) << "Points batch size" << " : " << B << "\n";
    std::cout << std::left << std::setw(20) << "Batches/SM"        << " : " << runtime_batches_per_sm << "\n";
    std::cout << std::left << std::setw(20) << "Batches/launch"    << " : " << slices_per_launch << " (per thread)\n";
    std::cout << std::left << std::setw(20) << "Memory utilization"<< " : "
              << std::fixed << std::setprecision(1) << util << "% ("
              << human_bytes((double)usedB) << " / " << human_bytes((double)totalB) << ")\n";
    std::cout << "------------------------------------------------------- \n";
    std::cout << std::left << std::setw(20) << "Total threads"     << " : " << (uint64_t)threadsTotal << "\n\n";
#ifdef RTX5090_OPT
    std::cout << std::left << std::setw(20) << "Active blocks/SM"  << " : " << activeBlocksPerSm
              << " (theoretical, " << (activeBlocksPerSm * threadsPerBlock / WARP_SIZE) << " warps/SM)\n\n";
#endif
    std::cout << "======== Phase-1: BruteForce ==========================\n";

    cudaStream_t streamKernel;
    ck(cudaStreamCreateWithFlags(&streamKernel, cudaStreamNonBlocking), "create stream");

#ifdef RTX5090_OPT
    (void)cudaFuncSetCacheConfig(kernel_point_add_and_check_oneinv<RTX5090_BATCH>, cudaFuncCachePreferL1);
#else
    (void)cudaFuncSetCacheConfig(kernel_point_add_and_check_oneinv, cudaFuncCachePreferL1);
#endif

    auto t0 = std::chrono::high_resolution_clock::now();
    auto tLast = t0;
    unsigned long long lastHashes = 0ull;
#ifdef RTX5090_OPT
    unsigned long long hashCounterWraps = 0ull;
#endif

    bool stop_all = false;
    bool completed_all = false;
    while (!stop_all) {
        if (g_sigint) std::cerr << "\n[Ctrl+C] Interrupt received. Finishing current kernel slice and exiting...\n";

        unsigned int zeroU = 0u;
        ck(cudaMemcpyAsync(d_any_left, &zeroU, sizeof(unsigned int), cudaMemcpyHostToDevice, streamKernel), "zero d_any_left");

#ifdef RTX5090_OPT
        kernel_point_add_and_check_oneinv<RTX5090_BATCH><<<blocks, threadsPerBlock, 0, streamKernel>>>(
#else
        kernel_point_add_and_check_oneinv<<<blocks, threadsPerBlock, 0, streamKernel>>>(
#endif
            d_Px, d_Py, d_Rx, d_Ry,
            d_start_scalars, d_counts256,
            threadsTotal,
#ifndef RTX5090_OPT
            B,
#endif
            slices_per_launch,
            d_found_flag, d_found_result,
            d_hashes_accum,
            d_any_left
        );
        cudaError_t launchErr = cudaGetLastError();
        if (launchErr != cudaSuccess) {
            std::cerr << "\nKernel launch error: " << cudaGetErrorString(launchErr) << "\n";
            stop_all = true;
        }

        while (!stop_all) {
            auto now = std::chrono::high_resolution_clock::now();
            double dt = std::chrono::duration<double>(now - tLast).count();
            if (dt >= 1.0) {
                unsigned long long h_hashes = 0ull;
                ck(cudaMemcpy(&h_hashes, d_hashes_accum, sizeof(unsigned long long), cudaMemcpyDeviceToHost), "read hashes");
                double delta = (double)(h_hashes - lastHashes);
                double mkeys = delta / (dt * 1e6);
                double elapsed = std::chrono::duration<double>(now - t0).count();
                long double total_keys_ld = ld_from_u256(range_len);
#ifdef RTX5090_OPT
                if (h_hashes < lastHashes) ++hashCounterWraps;
                const unsigned __int128 exact_hashes = ((unsigned __int128)hashCounterWraps << 64) | h_hashes;
                const long double exact_hashes_ld = std::ldexp((long double)hashCounterWraps, 64) + (long double)h_hashes;
                long double prog = total_keys_ld > 0.0L ? (exact_hashes_ld / total_keys_ld) * 100.0L : 0.0L;
#else
                long double prog = total_keys_ld > 0.0L ? ((long double)h_hashes / total_keys_ld) * 100.0L : 0.0L;
#endif
                if (prog > 100.0L) prog = 100.0L;
                std::cout << "\rTime: " << std::fixed << std::setprecision(1) << elapsed
                          << " s | Speed: " << std::fixed << std::setprecision(1) << mkeys
#ifdef RTX5090_OPT
                          << " Mkeys/s | Count: " << decimal_u128_5090(exact_hashes)
#else
                          << " Mkeys/s | Count: " << h_hashes
#endif
                          << " | Progress: " << std::fixed << std::setprecision(2) << (double)prog << " %";
                std::cout.flush();
                lastHashes = h_hashes; tLast = now;
            }

            int host_found = 0;
            ck(cudaMemcpy(&host_found, d_found_flag, sizeof(int), cudaMemcpyDeviceToHost), "read found_flag");
            if (host_found == FOUND_READY) { stop_all = true; break; }

            cudaError_t qs = cudaStreamQuery(streamKernel);
            if (qs == cudaSuccess) break;
            else if (qs != cudaErrorNotReady) { cudaGetLastError(); stop_all = true; break; }

            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        cudaStreamSynchronize(streamKernel);
        std::cout.flush();
        if (stop_all || g_sigint) break;

        unsigned int h_any = 0u;
        ck(cudaMemcpy(&h_any, d_any_left, sizeof(unsigned int), cudaMemcpyDeviceToHost), "read any_left");

        std::swap(d_Px, d_Rx);
        std::swap(d_Py, d_Ry);

        if (h_any == 0u) { completed_all = true; break; }
    }

    cudaDeviceSynchronize();
    std::cout << "\n";

#ifdef RTX5090_OPT
    bool counter_integrity_error = false;
    unsigned long long final_hashes = 0ull;
    ck(cudaMemcpy(&final_hashes, d_hashes_accum, sizeof(final_hashes), cudaMemcpyDeviceToHost), "final read hashes");
    if (final_hashes < lastHashes) ++hashCounterWraps;
    const unsigned __int128 exact_final_hashes = ((unsigned __int128)hashCounterWraps << 64) | final_hashes;
    std::cout << "Final count         : " << decimal_u128_5090(exact_final_hashes) << "\n";
    const bool range_count_fits_u128 = (range_len[2] | range_len[3]) == 0ull;
    const unsigned __int128 expected_hashes = ((unsigned __int128)range_len[1] << 64) | range_len[0];
    if (completed_all && range_count_fits_u128 && exact_final_hashes != expected_hashes) {
        std::cerr << "Counter integrity error: exhaustive range contains " << decimal_u128_5090(expected_hashes)
                  << " keys but device counted " << decimal_u128_5090(exact_final_hashes) << ".\n";
        counter_integrity_error = true;
    }
#endif

    int h_found_flag = 0;
    ck(cudaMemcpy(&h_found_flag, d_found_flag, sizeof(int), cudaMemcpyDeviceToHost), "final read found_flag");

#ifdef RTX5090_OPT
    int exit_code = counter_integrity_error ? EXIT_FAILURE : EXIT_SUCCESS;
#else
    int exit_code = EXIT_SUCCESS;
#endif

    if (h_found_flag == FOUND_READY) {
        FoundResult host_result{};
        ck(cudaMemcpy(&host_result, d_found_result, sizeof(FoundResult), cudaMemcpyDeviceToHost), "read found_result");
        std::cout << "\n======== FOUND MATCH! =================================\n";
        std::cout << "Private Key   : " << formatHex256(host_result.scalar) << "\n";
        std::cout << "Public Key    : " << formatCompressedPubHex(host_result.Rx, host_result.Ry) << "\n";
#ifdef RTX5090_OPT
        if (random_blocks.child) exit_code = 10;
#endif
    } else {
        if (g_sigint) {
            std::cout << "======== INTERRUPTED (Ctrl+C) ==========================\n";
            std::cout << "Search was interrupted by user. Partial progress above.\n";
            exit_code = 130;
        } else if (completed_all) {
            std::cout << "======== KEY NOT FOUND (exhaustive) ===================\n";
            std::cout << "Target hash160 was not found within the specified range.\n";
        } else {
            std::cout << "======== TERMINATED ===================================\n";
        }
    }

    cudaFree(d_start_scalars); cudaFree(d_Px); cudaFree(d_Py); cudaFree(d_Rx); cudaFree(d_Ry);
    cudaFree(d_counts256); cudaFree(d_found_flag); cudaFree(d_found_result); cudaFree(d_hashes_accum); cudaFree(d_any_left);
    cudaStreamDestroy(streamKernel);

    if (h_start_scalars) cudaFreeHost(h_start_scalars);
    if (h_counts256)     cudaFreeHost(h_counts256);

    return exit_code;
}
