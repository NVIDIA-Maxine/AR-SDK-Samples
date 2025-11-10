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

#ifndef __NVCVIMAGE_NONUNIFORM_SCALE_OFFSET__
#define __NVCVIMAGE_NONUNIFORM_SCALE_OFFSET__

#include "nvCVImage.h"


/// Copy the src image to the dst image, applying a [potentially different] scale and offset to each component.
/// The images can be anywhere, on the CPU or GPU, and they do not need to match.
/// However, they do need to have different component types, i.e. one should be F32 and the other U8.
/// @param src[in]          The source image, {RGB,BGR,RGBA,BGRA,ARGB,ABGR}{U8,F32}.
/// @param rgba_scale[in]   The scale factor to be applied to each component, in {R,G,B,A} order.
/// @param rgba_offset[in]  The offset to be applied to each component, in {R,G,B,A} order.
/// @param dst[out]         The destination image, {RGB,BGR,RGBA,BGRA,ARGB,ABGR}{U8,F32}.
/// @param stream[in,out]   The CUDA stream.
/// @param tmp[in]          A CPU staging image to be used if the src and dst are in different memory spaces.
/// @return NVCV_SUCCESS          If the operation was completed successfully.
///         NVCV_ERR_PIXELFORMAT  If the images are not {RGB,BGR,RGBA,BGRA,ARGB,ABGR}{U8,F32},
///                               of different component types.
NvCV_Status NvCVImageNonUniformScaleOffset(const NvCVImage *src, const float rgba_scale[4], const float rgba_offset[4],
            NvCVImage *dst, struct CUstream_st *stream, NvCVImage *tmp);


#endif // __NVCVIMAGE_NONUNIFORM_SCALE_OFFSET__
