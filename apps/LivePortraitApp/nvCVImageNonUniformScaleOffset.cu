/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "nvCVImageNonUniformScaleOffset.h"

#include <math.h>
#include <string.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

typedef int offset_t;
// Note: an explicit cast must be used to assure that signed*unsigned is executed as a signed operation,
// because the C default is    signed * unsigned --> unsigned
#define BYTE_OFFSET(x, y, pixBytes, rowBytes) ((offset_t)(y) * (offset_t)(rowBytes) + (offset_t)(x) * (offset_t)(pixBytes))
#define SRC_OFFSET(x, y) BYTE_OFFSET(x, y, p.sPixBytes, p.sRowBytes)
#define DST_OFFSET(x, y) BYTE_OFFSET(x, y, p.dPixBytes, p.dRowBytes)

#define BAIL_IF_ERR(err)            do { if ((err) != 0)          { goto bail;             } } while(0)
#define BAIL(err, code)             do {                            err = code; goto bail;   } while(0)

////////////////////////////////////////////////////////////////////////////////
/// Compute the dimBlock and dimGrid for CUDA.
/// \param[in]  dstWidth  the width of the image.
/// \param[in]  dstHeight the height of the image.
/// \param[out] dimBlock  the block size for CUDA computations.
/// \param[out] dimGrid   the grid of blocks for CUDA computations.
////////////////////////////////////////////////////////////////////////////////
#define DIM_BLOCK_X 32   // These could be adjusted to increase performance
#define DIM_BLOCK_Y 32

__host__ static void ComputeDimBlockGrid(unsigned dstWidth, unsigned dstHeight, dim3& dimBlock, dim3& dimGrid) {
  dimBlock.x = DIM_BLOCK_X;
  dimBlock.y = DIM_BLOCK_Y;
  dimBlock.z = 1;
  dimGrid.x  = (dstWidth  + dimBlock.x - 1) / dimBlock.x; /* ceil */
  dimGrid.y  = (dstHeight + dimBlock.y - 1) / dimBlock.y;
  dimGrid.z  = 1;
}


////////////////////////////////////////////////////////////////////////////////
/// Parameter for the scale offset.
////////////////////////////////////////////////////////////////////////////////

struct NonUniformScaleOffset {
  /// Set the parameters.
  /// @param[in]  A         the affine transformation.
  /// @param[in]  src_rect  the source rectangle, to optimize rendering (can be NULL).
  /// @param[out] dst       the result image.
  const unsigned char *sr, *sg, *sb, *sa;  // src pixel component pointers
  unsigned char       *dr, *dg, *db, *da;  // dst pixel component pointers
  offset_t            sPixBytes, sRowBytes, dPixBytes, dRowBytes;
  unsigned            width, height;
  float               scale[4], offset[4];
  __host__ void set(const NvCVImage *src, NvCVImage *dst, const float scale_in[4], const float offset_in[4]);
};


////////////////////////////////////////////////////////////////////////////////
// Set NonUniformScaleOffset
////////////////////////////////////////////////////////////////////////////////

void NonUniformScaleOffset::set(const NvCVImage *src, NvCVImage *dst, const float scale_in[4], const float offset_in[4]) {
  int ro, go, bo, ao;
  offset_t nextCompBytes;

  width  = dst->width;  // Set rect
  height = dst->height;
  for (int i = 0; i < 4; i++) {
    scale[i] = scale_in[i];
    offset[i] = offset_in[i];
  }

  NvCVImage_ComponentOffsets(src->pixelFormat, &ro, &go, &bo, &ao, nullptr);  // Set src
  if (!src->planar) {
    nextCompBytes = src->componentBytes;
    sPixBytes = src->pixelBytes;
  } else {
    nextCompBytes = (offset_t)src->pitch * (offset_t)src->height;
    sPixBytes = src->componentBytes;
  }
  sb = (const unsigned char*)src->pixels;
  sa = sb + nextCompBytes * ao;
  sr = sb + nextCompBytes * ro;
  sg = sb + nextCompBytes * go;
  sb = sb + nextCompBytes * bo;
  sRowBytes = src->pitch;

  NvCVImage_ComponentOffsets(dst->pixelFormat, &ro, &go, &bo, &ao, nullptr);  // Set dst
  if (!dst->planar) {
    nextCompBytes = dst->componentBytes;
    dPixBytes = dst->pixelBytes;
  } else {
    nextCompBytes = (offset_t)dst->pitch * (offset_t)dst->height;
    dPixBytes = dst->componentBytes;
  }
  db = (unsigned char*)dst->pixels;
  da = db + nextCompBytes * ao;
  dr = db + nextCompBytes * ro;
  dg = db + nextCompBytes * go;
  db = db + nextCompBytes * bo;
  dRowBytes = dst->pitch;
}


////////////////////////////////////////////////////////////////////////////////
/// Scale & offset RGBu8 --> RGBf32.
/// param[in] p the affine parameters.
////////////////////////////////////////////////////////////////////////////////

__host__ static void NonUniformScoff_RGBu8_RGBf32_CPU(const NonUniformScaleOffset &p) {
  const offset_t  sGap = p.sRowBytes - p.width * p.sPixBytes,
                  dGap = p.dRowBytes - p.width * p.dPixBytes,
                  dRow = p.width * p.dPixBytes;
  offset_t        sz, dz, dEndRow, dEndImg;
  for (sz = 0, dz = 0, dEndImg = dz + p.height * p.dRowBytes; dz != dEndImg; dz += dGap, sz += sGap) {
    for (dEndRow = dz + dRow; dz != dEndRow; dz += p.dPixBytes, sz += p.sPixBytes) {
      *((float*)(p.dr + dz)) = *((const unsigned char*)(p.sr + sz)) * p.scale[0] + p.offset[0];
      *((float*)(p.dg + dz)) = *((const unsigned char*)(p.sg + sz)) * p.scale[1] + p.offset[1];
      *((float*)(p.db + dz)) = *((const unsigned char*)(p.sb + sz)) * p.scale[2] + p.offset[2];
    }
  }
}

__global__ static void NonUniformScoff_RGBu8_RGBf32_GPU(const NonUniformScaleOffset p) {
  unsigned x = blockIdx.x * blockDim.x + threadIdx.x; /* Get destination coordinates */
  unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= p.width || y >= p.height) return;
  offset_t sz = SRC_OFFSET(x, y);
  offset_t dz = DST_OFFSET(x, y);
  *((float*)(p.dr + dz)) = *((const unsigned char*)(p.sr + sz)) * p.scale[0] + p.offset[0];
  *((float*)(p.dg + dz)) = *((const unsigned char*)(p.sg + sz)) * p.scale[1] + p.offset[1];
  *((float*)(p.db + dz)) = *((const unsigned char*)(p.sb + sz)) * p.scale[2] + p.offset[2];
}


////////////////////////////////////////////////////////////////////////////////
/// Scale & offset RGBf32 --> RGBu8.
/// param[in] p the affine parameters.
////////////////////////////////////////////////////////////////////////////////

__host__ static void NonUniformScoff_RGBf32_RGBu8_CPU(const NonUniformScaleOffset &p) {
  const offset_t  sGap = p.sRowBytes - p.width * p.sPixBytes,
                  dGap = p.dRowBytes - p.width * p.dPixBytes,
                  dRow = p.width * p.dPixBytes;
  offset_t        sz, dz, dEndRow, dEndImg;
  for (sz = 0, dz = 0, dEndImg = dz + p.height * p.dRowBytes; dz != dEndImg; dz += dGap, sz += sGap) {
    for (dEndRow = dz + dRow; dz != dEndRow; dz += p.dPixBytes, sz += p.sPixBytes) {
      float f;
      f = *((const float*)(p.sr + sz)) * p.scale[0] + p.offset[0];
      *((unsigned char*)(p.dr + dz)) = (f <= 0.f) ? 0 : (f >= 255.f) ? 255 : (unsigned char)(f + 0.5f);
      f = *((const float*)(p.sg + sz)) * p.scale[1] + p.offset[1];
      *((unsigned char*)(p.dg + dz)) = (f <= 0.f) ? 0 : (f >= 255.f) ? 255 : (unsigned char)(f + 0.5f);
      f = *((const float*)(p.sb + sz)) * p.scale[2] + p.offset[2];
      *((unsigned char*)(p.db + dz)) = (f <= 0.f) ? 0 : (f >= 255.f) ? 255 : (unsigned char)(f + 0.5f);
    }
  }
}

__global__ static void NonUniformScoff_RGBf32_RGBu8_GPU(const NonUniformScaleOffset p) {
  unsigned x = blockIdx.x * blockDim.x + threadIdx.x; /* Get destination coordinates */
  unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= p.width || y >= p.height) return;
  offset_t sz = SRC_OFFSET(x, y);
  offset_t dz = DST_OFFSET(x, y);
  float f;
  f = *((const float*)(p.sr + sz)) * p.scale[0] + p.offset[0];
  *((unsigned char*)(p.dr + dz)) = (f <= 0.f) ? 0 : (f >= 255.f) ? 255 : (unsigned char)(f + 0.5f);
  f = *((const float*)(p.sg + sz)) * p.scale[1] + p.offset[1];
  *((unsigned char*)(p.dg + dz)) = (f <= 0.f) ? 0 : (f >= 255.f) ? 255 : (unsigned char)(f + 0.5f);
  f = *((const float*)(p.sb + sz)) * p.scale[2] + p.offset[2];
  *((unsigned char*)(p.db + dz)) = (f <= 0.f) ? 0 : (f >= 255.f) ? 255 : (unsigned char)(f + 0.5f);
}



////////////////////////////////////////////////////////////////////////////////
/// Scale & offset RGBAu8 --> RGBAf32.
/// param[in] p the affine parameters.
////////////////////////////////////////////////////////////////////////////////

__host__ static void NonUniformScoff_RGBAu8_RGBAf32_CPU(const NonUniformScaleOffset &p) {
  const offset_t  sGap = p.sRowBytes - p.width * p.sPixBytes,
                  dGap = p.dRowBytes - p.width * p.dPixBytes,
                  dRow = p.width * p.dPixBytes;
  offset_t        sz, dz, dEndRow, dEndImg;
  for (sz = 0, dz = 0, dEndImg = dz + p.height * p.dRowBytes; dz != dEndImg; dz += dGap, sz += sGap) {
    for (dEndRow = dz + dRow; dz != dEndRow; dz += p.dPixBytes, sz += p.sPixBytes) {
      *((float*)(p.dr + dz)) = *((const unsigned char*)(p.sr + sz)) * p.scale[0] + p.offset[0];
      *((float*)(p.dg + dz)) = *((const unsigned char*)(p.sg + sz)) * p.scale[1] + p.offset[1];
      *((float*)(p.db + dz)) = *((const unsigned char*)(p.sb + sz)) * p.scale[2] + p.offset[2];
      *((float*)(p.da + dz)) = *((const unsigned char*)(p.sa + sz)) * p.scale[3] + p.offset[3];
    }
  }
}

__global__ static void NonUniformScoff_RGBAu8_RGBAf32_GPU(const NonUniformScaleOffset p) {
  unsigned x = blockIdx.x * blockDim.x + threadIdx.x; /* Get destination coordinates */
  unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= p.width || y >= p.height) return;
  offset_t sz = SRC_OFFSET(x, y);
  offset_t dz = DST_OFFSET(x, y);
  *((float*)(p.dr + dz)) = *((const unsigned char*)(p.sr + sz)) * p.scale[0] + p.offset[0];
  *((float*)(p.dg + dz)) = *((const unsigned char*)(p.sg + sz)) * p.scale[1] + p.offset[1];
  *((float*)(p.db + dz)) = *((const unsigned char*)(p.sb + sz)) * p.scale[2] + p.offset[2];
  *((float*)(p.da + dz)) = *((const unsigned char*)(p.sa + sz)) * p.scale[3] + p.offset[3];
}


////////////////////////////////////////////////////////////////////////////////
/// Scale & offset RGBAf32 --> RGBAu8.
/// param[in] p the affine parameters.
////////////////////////////////////////////////////////////////////////////////

__host__ static void NonUniformScoff_RGBAf32_RGBAu8_CPU(const NonUniformScaleOffset &p) {
  const offset_t  sGap = p.sRowBytes - p.width * p.sPixBytes,
                  dGap = p.dRowBytes - p.width * p.dPixBytes,
                  dRow = p.width * p.dPixBytes;
  offset_t        sz, dz, dEndRow, dEndImg;
  for (sz = 0, dz = 0, dEndImg = dz + p.height * p.dRowBytes; dz != dEndImg; dz += dGap, sz += sGap) {
    for (dEndRow = dz + dRow; dz != dEndRow; dz += p.dPixBytes, sz += p.sPixBytes) {
      float f;
      f = *((const float*)(p.sr + sz)) * p.scale[0] + p.offset[0];
      *((unsigned char*)(p.dr + dz)) = (f <= 0.f) ? 0 : (f >= 255.f) ? 255 : (unsigned char)(f + 0.5f);
      f = *((const float*)(p.sg + sz)) * p.scale[1] + p.offset[1];
      *((unsigned char*)(p.dg + dz)) = (f <= 0.f) ? 0 : (f >= 255.f) ? 255 : (unsigned char)(f + 0.5f);
      f = *((const float*)(p.sb + sz)) * p.scale[2] + p.offset[2];
      *((unsigned char*)(p.db + dz)) = (f <= 0.f) ? 0 : (f >= 255.f) ? 255 : (unsigned char)(f + 0.5f);
      f = *((const float*)(p.sa + sz)) * p.scale[3] + p.offset[3];
      *((unsigned char*)(p.da + dz)) = (f <= 0.f) ? 0 : (f >= 255.f) ? 255 : (unsigned char)(f + 0.5f);
    }
  }
}

__global__ static void NonUniformScoff_RGBAf32_RGBAu8_GPU(const NonUniformScaleOffset p) {
  unsigned x = blockIdx.x * blockDim.x + threadIdx.x; /* Get destination coordinates */
  unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= p.width || y >= p.height) return;
  offset_t sz = SRC_OFFSET(x, y);
  offset_t dz = DST_OFFSET(x, y);
  float f;
  f = *((const float*)(p.sr + sz)) * p.scale[0] + p.offset[0];
  *((unsigned char*)(p.dr + dz)) = (f <= 0.f) ? 0 : (f >= 255.f) ? 255 : (unsigned char)(f + 0.5f);
  f = *((const float*)(p.sg + sz)) * p.scale[1] + p.offset[1];
  *((unsigned char*)(p.dg + dz)) = (f <= 0.f) ? 0 : (f >= 255.f) ? 255 : (unsigned char)(f + 0.5f);
  f = *((const float*)(p.sb + sz)) * p.scale[2] + p.offset[2];
  *((unsigned char*)(p.db + dz)) = (f <= 0.f) ? 0 : (f >= 255.f) ? 255 : (unsigned char)(f + 0.5f);
  f = *((const float*)(p.sa + sz)) * p.scale[3] + p.offset[3];
  *((unsigned char*)(p.da + dz)) = (f <= 0.f) ? 0 : (f >= 255.f) ? 255 : (unsigned char)(f + 0.5f);
}


////////////////////////////////////////////////////////////////////////////////
// NvCVImageScaleOffset API
////////////////////////////////////////////////////////////////////////////////

NvCV_Status NvCVImageNonUniformScaleOffset(
    const NvCVImage *src, const float rgba_scale[4], const float rgba_offset[4], NvCVImage *dst, struct CUstream_st *stream, NvCVImage *tmp
) {
  NvCV_Status err = NVCV_SUCCESS;
  NvCVImage   loc;    // Use this if no tmp was given
  NonUniformScaleOffset params;

  if (!(NVCV_RGB <= src->pixelFormat && src->pixelFormat < NVCV_YUV420 &&
        NVCV_RGB <= dst->pixelFormat && dst->pixelFormat < NVCV_YUV420
  ))
    return NVCV_ERR_PIXELFORMAT;

  if (!((src->gpuMem & NVCV_GPU) | (dst->gpuMem & NVCV_GPU))) {                 // CPU --> CPU
    void (*cfunc)(const NonUniformScaleOffset& p);
    if (NVCV_U8 == src->componentType && NVCV_F32 == dst->componentType)
      cfunc = (dst->numComponents == 3) ? NonUniformScoff_RGBu8_RGBf32_CPU : NonUniformScoff_RGBAu8_RGBAf32_CPU;
    else if (NVCV_F32 == src->componentType && NVCV_U8 == dst->componentType)
      cfunc = (dst->numComponents == 3) ? NonUniformScoff_RGBf32_RGBu8_CPU : NonUniformScoff_RGBAf32_RGBAu8_CPU;
    else
      BAIL(err, NVCV_ERR_PIXELFORMAT);
    params.set(src, dst, rgba_scale, rgba_offset);
    (*cfunc)(params);
  }
  else {
    void (*gfunc)(const NonUniformScaleOffset p);
    if (NVCV_U8 == src->componentType && NVCV_F32 == dst->componentType)
      gfunc = (dst->numComponents == 3) ? NonUniformScoff_RGBu8_RGBf32_GPU : NonUniformScoff_RGBAu8_RGBAf32_GPU;
    else if (NVCV_F32 == src->componentType && NVCV_U8 == dst->componentType)
      gfunc = (dst->numComponents == 3) ? NonUniformScoff_RGBf32_RGBu8_GPU : NonUniformScoff_RGBAf32_RGBAu8_GPU;
    else
      BAIL(err, NVCV_ERR_PIXELFORMAT);

    dim3 dimBlock, dimGrid;
    ComputeDimBlockGrid(dst->width, dst->height, dimBlock, dimGrid);
    if (!tmp) tmp = &loc;

    if (!(src->gpuMem & NVCV_GPU)) {                                            // CPU --> GPU
      BAIL_IF_ERR(err = NvCVImage_Realloc(tmp, src->width, src->height, src->pixelFormat, src->componentType,
                src->planar, NVCV_CUDA, 0));                                    // Allocate GPU tmp src
      BAIL_IF_ERR(err = NvCVImage_Transfer(src, tmp, 1.f, stream, nullptr));    // tmp src --> GPU
      params.set(tmp, dst, rgba_scale, rgba_offset);
      (*gfunc)<<<dimGrid, dimBlock, 0, stream>>>(params);
    }

    else if (!(dst->gpuMem & NVCV_GPU)) {                                       // GPU --> CPU
      BAIL_IF_ERR(err = NvCVImage_Realloc(tmp, dst->width, dst->height, dst->pixelFormat, dst->componentType,
                dst->planar, NVCV_CUDA, 0));                                    // Allocate GPU tmp dst
      params.set(src, tmp, rgba_scale, rgba_offset);
      (*gfunc)<<<dimGrid, dimBlock, 0, stream>>>(params);                       // src --> GPU tmp dst
      BAIL_IF_ERR(err = NvCVImage_Transfer(tmp, dst, 1.f, stream, nullptr));    // GPU tmp dst --> CPU dst
    }

    else {                                                                      // GPU --> GPU
      params.set(src, dst, rgba_scale, rgba_offset);
      (*gfunc)<<<dimGrid, dimBlock, 0, stream>>>(params);
    }
    NvCVImage_DeallocAsync(&loc, stream);         // This is a no-op if loc wasn't used
  }

bail:
  return err;
}
