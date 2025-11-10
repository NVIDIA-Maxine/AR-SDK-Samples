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

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "nvAR.h"
#include "nvARFrameSelection.h"
#include "nvARLivePortrait.h"
#include "nvAR_defs.h"
#include "nvCVOpenCV.h"
#include "opencv2/opencv.hpp"

#include <cuda.h>

#include "npp.h"

#if CV_MAJOR_VERSION >= 4
#define CV_CAP_PROP_FRAME_WIDTH cv::CAP_PROP_FRAME_WIDTH
#define CV_CAP_PROP_FRAME_HEIGHT cv::CAP_PROP_FRAME_HEIGHT
#define CV_CAP_PROP_FPS cv::CAP_PROP_FPS
#define CV_CAP_PROP_FRAME_COUNT cv::CAP_PROP_FRAME_COUNT
#define CV_WINDOW_AUTOSIZE cv::WINDOW_AUTOSIZE
#endif

#ifndef M_PI
#define M_PI 3.1415926535897932385
#endif /* M_PI */
#ifndef M_2PI
#define M_2PI 6.2831853071795864769
#endif /* M_2PI */
#ifndef M_PI_2
#define M_PI_2 1.5707963267948966192
#endif /* M_PI_2 */
#define F_PI ((float)M_PI)
#define F_PI_2 ((float)M_PI_2)
#define F_2PI ((float)M_2PI)

#ifdef _MSC_VER
#define strcasecmp _stricmp
#endif /* _MSC_VER */

#define MODE_CROP_NONE 0
#define MODE_CROP_FACEBOX 1
#define MODE_CROP_BLEND 2
#define MODE_CROP_INSET_BLEND 3

#define LP_FACEBOX_CHECK_OK 0
#define LP_FACEBOX_CHECK_CLOSE_TO_BORDER 1
#define LP_FACEBOX_CHECK_OUTSIDE_BORDER 2
#define LP_FACEBOX_CHECK_BAD_AREA 3

#define MODEL_SEL_PERF 0
#define MODEL_SEL_QUAL 1

#define FS_GOOD_FRAME_MIN_INTERVAL_DEFAULT 0  // no interval needed b/w consecutive good frames.
#define FS_ACTIVEDURATION_DEFAULT 150         // 150: SDK checks for good frame for the first 150 frames.

#define FRAME_SELECTION_DISABLED 0
#define FRAME_SELECTION_TRIGGER_ONCE 1
#define FRAME_SELECTION_TRIGGER_MANY 2

#define FRAME_SELECTION_STRATEGY_STATIC 0
#define FRAME_SELECTION_STRATEGY_IMPROVING 1
#define FRAME_SELECTION_STRATEGY_DEFAULT FRAME_SELECTION_STRATEGY_IMPROVING


// gen_img_xx might be overwritten in InitOutput().
// This image size correspond to the final output and it may differ depending on which mode is used in the feature
unsigned int gen_img_width;
unsigned int gen_img_height;
/********************************************************************************
 * Command-line arguments
 ********************************************************************************/
// clang-format off
bool          FLAG_verbose          = false,
              FLAG_offlineMode      = false,              // reads driving video from file if set to true; webcam mode if set to false
              FLAG_captureOutputs   = false,              // write generated video to file if set to true. only in offline mode
              FLAG_ignoreAlpha      = false,              // igore the alpha channel of the source image (RGBA format only)
              FLAG_showDrive        = true,               // show the driving video
              FLAG_showBBox         = false;              // show the bounding box overlaying the driving video
int           FLAG_cameraID         = 0,
              FLAG_mode             = MODE_CROP_FACEBOX,
              FLAG_modelSel         = MODEL_SEL_QUAL,
              FLAG_FrameSelection   = FRAME_SELECTION_TRIGGER_MANY,  // select neutral frame from the driving video. 0 - disabled; 1 - trigger once; 2- trigger many times
              FLAG_logLevel         = NVCV_LOG_ERROR;
std::string   FLAG_outDir,
              FLAG_inSrc,
              FLAG_inBgImg,
              FLAG_inDrv,
              FLAG_outFile,
              FLAG_modelPath,
              FLAG_landmarks,
              FLAG_captureCodec     = "avc1",
              FLAG_camRes           = "640x480",
              FLAG_log              = "stderr";

// clang-format on

bool CheckResult(NvCV_Status nvErr, unsigned line) {
  if (NVCV_SUCCESS == nvErr) return true;
  std::cout << "ERROR: " << NvCV_GetErrorStringFromCode(nvErr) << ", line " << line << std::endl;
  return false;
}

std::string GetFaceBoxStatusAsString(const unsigned int face_box_status) {
  switch (face_box_status) {
    case LP_FACEBOX_CHECK_OK:
      return "OK";
    case LP_FACEBOX_CHECK_CLOSE_TO_BORDER:
      return "Close to border";
    case LP_FACEBOX_CHECK_OUTSIDE_BORDER:
      return "Outside border";
    case LP_FACEBOX_CHECK_BAD_AREA:
      return "Bad face area";
    default:
      return "Unknown";
  }
}

#define BAIL(err, code) \
  do {                  \
    err = code;         \
    goto bail;          \
  } while (0)

#define BAIL_IF_ERR(err) \
  do {                   \
    if (0 != (err)) {    \
      goto bail;         \
    }                    \
  } while (0)

#define BAIL_IF_NVERR(nvErr, err, code)  \
  do {                                   \
    if (!CheckResult(nvErr, __LINE__)) { \
      err = code;                        \
      goto bail;                         \
    }                                    \
  } while (0)

/********************************************************************************
 * Usage
 ********************************************************************************/

static void Usage() {
  printf(
      "LivePortraitApp [<args> ...]\n"
      "where <args> are\n"
      " --verbose[=(true|false)]          report interesting info\n"
      " --log=<file>                      log SDK errors to a file, \"stderr\" or \"\" (default stderr)\n"
      " --log_level=<N>                   the desired log level: {0, 1, 2, 3} = {FATAL, ERROR, WARNING, INFO}, "
      "respectively (default 1)\n"
      " --mode=<1|2|3>                    cropping mode. Choose from MODE_CROP_FACEBOX(1), MODE_CROP_BLEND(2), "
      "and MODE_CROP_INSET_BLEND(3). Default is 1.\n"
      " --model_path=<path>               specify the directory containing the TRT models\n"
      " --model_sel=<0|1>                 select the model. 0 for perf, 1 for quality. Default is 1\n"
      " --offline_mode[=(true|false)]     reads driving video from file if set to true; webcam mode if set to false. "
      "Default false\n"
      " --capture_outputs[=(true|false)]  write generated video to file if set to true. only in offline mode\n"
      " --cam_res=[WWWx]HHH               specify resolution as height or width x height. only in webcam mode. Default "
      "is 640x480\n"
      " --camera=<ID>                     specify the camera ID. Default 0\n"
      " --codec=<fourcc>                  FOURCC code for the desired codec (default H264)\n"
      " --in_src=<file>                   specify the input source file (portrait image)\n"
      " --in_drv=<file>                   specify the input driving file. only in offline mode\n"
      " --bg_img=<file>                   specify the image to use as background in the output\n"
      " --out=<file>                      specify the output file. only in offline mode and capture_outputs is true.\n"
      " --ignore_alpha[=(true|false)]     igore the alpha channel of the source image (RGBA format only) (default "
      "false)\n"
      " --show_drive[=(true|false)]       show the driving video (default true)\n"
      " --show_bbox[=(true|false)]        overlay the bounding box on the driving video (default false)\n"
      " --frame_selection=<0|1|2>         run frame selection on the driving video. 0 - disabled. 1 - run once. "
      "2(default) - run many times\n"
  );
}

static bool GetFlagArgVal(const char* flag, const char* arg, const char** val) {
  if (*arg != '-') {
    return false;
  }
  while (*++arg == '-') {
    continue;
  }
  const char* s = strchr(arg, '=');
  if (s == NULL) {
    if (strcmp(flag, arg) != 0) {
      return false;
    }
    *val = NULL;
    return true;
  }
  unsigned n = (unsigned)(s - arg);
  if ((strlen(flag) != n) || (strncmp(flag, arg, n) != 0)) {
    return false;
  }
  *val = s + 1;
  return true;
}

static bool GetFlagArgVal(const char* flag, const char* arg, std::string* val) {
  const char* val_str;
  if (!GetFlagArgVal(flag, arg, &val_str)) return false;
  val->assign(val_str ? val_str : "");
  return true;
}

static bool GetFlagArgVal(const char* flag, const char* arg, bool* val) {
  const char* val_str;
  bool success = GetFlagArgVal(flag, arg, &val_str);
  if (success) {
    *val = (val_str == NULL || strcasecmp(val_str, "true") == 0 || strcasecmp(val_str, "on") == 0 ||
            strcasecmp(val_str, "yes") == 0 || strcasecmp(val_str, "1") == 0);
  }
  return success;
}

bool GetFlagArgVal(const char* flag, const char* arg, long* val) {
  const char* valStr;
  bool success = GetFlagArgVal(flag, arg, &valStr);
  if (success) {
    *val = strtol(valStr, NULL, 0);  // accommodate 123 decimal, 0x123 hex and 0123 octal
  }
  return success;
}

static bool GetFlagArgVal(const char* flag, const char* arg, unsigned* val) {
  long long_val;
  bool success = GetFlagArgVal(flag, arg, &long_val);
  if (success) {
    *val = (unsigned)long_val;
  }
  return success;
}

static bool GetFlagArgVal(const char* flag, const char* arg, int* val) {
  long longVal;
  bool success = GetFlagArgVal(flag, arg, &longVal);
  if (success) *val = (int)longVal;
  return success;
}

static std::string getTimeStr() {
  std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  char s[30] = {0};
  std::strftime(s, 30, "%Y-%m-%d-%H-%M-%S", std::localtime(&now));
  return std::string(s);
}

static void saveImage(int frame_cnt, const cv::Mat& img) {
  auto t = std::time(nullptr);
  auto tm = *std::localtime(&t);
  std::string frame_name = std::to_string(frame_cnt) + "_" + getTimeStr() + "_frm" + ".jpg";
  cv::imwrite(frame_name, img);
}

/********************************************************************************
 * StringToFourcc
 ********************************************************************************/

static int StringToFourcc(const std::string& str) {
  union chint {
    int i;
    char c[4];
  };
  chint x = {0};
  for (int n = (str.size() < 4) ? (int)str.size() : 4; n--;) x.c[n] = str[n];
  return x.i;
}

/******************************* IMAGE UTILITIES ******************************/

/*******************************************************************************
 * Check whether the image has an alpha channel suitable as a segmentation mask.
 * @param[in]  im  the image to be tested.
 * @return     true if the imagew has a non-trivial alpha channel.
 ********************************************************************************/

static bool HasNontrivialAlpha(const NvCVImage* im) {
  int widthBytes = im->width * im->pixelBytes;
  int gap = im->pitch - widthBytes;
  const unsigned char *p, *pend;
  unsigned char same;
  int ao;

  if (im->numComponents != 4) return false;
  if (im->componentType != NVCV_U8) return false;
  NvCVImage_ComponentOffsets(im->pixelFormat, nullptr, nullptr, nullptr, &ao, nullptr);
  p = (const unsigned char*)im->pixels + ao;
  same = *p;
  for (pend = p + im->pitch * im->height; p != pend; p += gap)
    for (const unsigned char* prow = p + widthBytes; p != prow; p += 4)
      if (*p != same) return true;
  return false;
}

/*******************************************************************************
 * Make a vertical (top->bottom) gradient between 2 colors.
 * @param[in]   grad     two BGR colors that establish the extent of the gradient
 * @param[out]  im       the image where the vertical gradient image is to be generated.
 * @return      NVCV_SUCCESS if the operation was successful.
 ********************************************************************************/

static NvCV_Status MakeVerticalGradientBGR(const NvAR_Point3f grad[2], NvCVImage* im) {
  NvCV_Status err = NVCV_SUCCESS;
  unsigned char *p, *p_row, *p_end;
  unsigned char b, g, r;
  int gap, row, r_off, b_off, width_bytes;
  NvAR_Point3f color0, delta_color;

  if (im->componentType != NVCV_U8) {
    return NVCV_ERR_PIXELFORMAT;
  }
  NvCVImage_ComponentOffsets(im->pixelFormat, &r_off, nullptr, &b_off, nullptr, nullptr);
  p = reinterpret_cast<unsigned char*>(im->pixels) + ((r_off < b_off) ? r_off : b_off);
  p_end = p + im->pitch * im->height;
  gap = im->pitch - (im->width * im->pixelBytes);
  width_bytes = im->width * im->pixelBytes;

  color0 = {grad[0].x + 0.5f, grad[0].y + 0.5f, grad[0].z + 0.5f};
  delta_color = {(grad[1].x - grad[0].x) / (im->height - 1), (grad[1].y - grad[0].y) / (im->height - 1),
                 (grad[1].z - grad[0].z) / (im->height - 1)};
  for (row = 0; p != p_end; p += gap, row++) {
    b = (unsigned char)(color0.x + row * delta_color.x);
    g = (unsigned char)(color0.y + row * delta_color.y);
    r = (unsigned char)(color0.z + row * delta_color.z);
    for (p_row = p + width_bytes; p != p_row; p += im->pixelBytes) {
      p[0] = b;
      p[1] = g;
      p[2] = r;
    }
  }

  return err;
}

/*******************************************************************************
 * Resize an image while maintaining the aspect ratio, by cropping to avoid letterboxing.
 * Only supports chunky BGR u8 images on the GPU
 * @param[in]   src           the input image to be resized
 * @param[out]  dst           the output resized image.
 * @param[out]  pTmpImage     a temporary buffer that is sometimes needed when transferring images
 * @param[out]  context       an NPP stream context.
 * @return      NVCV_SUCCESS if the operation was successful.
 ********************************************************************************/

static NvCV_Status ResizeWithoutLetterboxing(NvCVImage* src, NvCVImage* dst, NvCVImage* pTmpImage,
                                             NppStreamContext context) {
  if (!(src->numComponents == 3 && src->componentType == NVCV_U8 && dst->componentType == NVCV_U8 &&
        src->gpuMem == NVCV_CUDA && dst->gpuMem == NVCV_CUDA && src->planar == NVCV_CHUNKY &&
        dst->planar == NVCV_CHUNKY))
    return NVCV_ERR_PIXELFORMAT;
  if (src->pixelFormat != dst->pixelFormat) return NVCV_ERR_MISMATCH;
  if (src->width == dst->width && src->height == dst->height) {
    if (src != dst) {
      return NvCVImage_Transfer(src, dst, 1.0f, context.hStream, pTmpImage);
    }
    return NVCV_SUCCESS;
  }

  NvCVRect2i srcRect = {0, 0, static_cast<int>(src->width), static_cast<int>(src->height)};
  float xScale = static_cast<float>(dst->width) / static_cast<float>(src->width),
        yScale = static_cast<float>(dst->height) / static_cast<float>(src->height);

  if (xScale > yScale)  // wider than height
  {
    // We need to copy (and resize) from a shorter subregion of the source image
    srcRect.height = int(dst->height / xScale + .5f);
  } else if (yScale > xScale)  // taller than width
  {
    // We need to copy (and resize) from a narrower subregion of the source image
    srcRect.width = int(dst->width / yScale + .5f);
  }

  NppiSize src_size = {static_cast<int>(src->width), static_cast<int>(src->height)},
           dst_size = {static_cast<int>(dst->width), static_cast<int>(dst->height)};
  NppiRect src_roi = {0, 0, static_cast<int>(srcRect.width), static_cast<int>(srcRect.height)},
           dst_roi = {0, 0, static_cast<int>(dst->width), static_cast<int>(dst->height)};
  NppiInterpolationMode interpolation_mode = NPPI_INTER_SUPER;
  if (xScale > 1.0f || yScale > 1.f) {
    interpolation_mode = NPPI_INTER_LANCZOS;
  }

  // clear the resized_buffer in GPU before using. Otherwise npp resize kernel will produce inconsistent values.
  cudaMemsetAsync(dst->pixels, 0, dst->bufferBytes, context.hStream);

  // resize from a subregion of the src buffer to accommodate the aspect ratio of the dst buffer
  NppStatus status =
      nppiResize_8u_C3R_Ctx((const Npp8u*)(src->pixels), (int)(src->pitch), src_size, src_roi, (Npp8u*)(dst->pixels),
                            (int)(dst->pitch), dst_size, dst_roi, interpolation_mode, context);
#ifdef DEBUG_IMAGE_UTILITIES
  if (status != NPP_SUCCESS) {
    printf("ResizeWithoutLetterbox failed: %d\n", (int)status);
  }
#endif  // DEBUG_IMAGE_UTILITIES
  return (status == NPP_SUCCESS) ? NVCV_SUCCESS : NVCV_ERR_NPP;
}

NvCV_Status createNPPStreamContext(CUstream stream, NppStreamContext& stream_context) {
  int dev_id;
  // CUDA device is already set by the time of effect creation and loading
  cudaGetDevice(&dev_id);
  stream_context.hStream = stream;
  stream_context.nCudaDeviceId = dev_id;
  int shared_mem_per_block;
  if (cudaDeviceGetAttribute(&stream_context.nCudaDevAttrComputeCapabilityMajor, cudaDevAttrComputeCapabilityMajor,
                             dev_id) != cudaSuccess ||
      cudaDeviceGetAttribute(&stream_context.nCudaDevAttrComputeCapabilityMinor, cudaDevAttrComputeCapabilityMinor,
                             dev_id) != cudaSuccess ||
      cudaDeviceGetAttribute(&stream_context.nMultiProcessorCount, cudaDevAttrMultiProcessorCount, dev_id) !=
          cudaSuccess ||
      cudaDeviceGetAttribute(&stream_context.nMaxThreadsPerMultiProcessor, cudaDevAttrMaxThreadsPerMultiProcessor,
                             dev_id) != cudaSuccess ||
      cudaDeviceGetAttribute(&stream_context.nMaxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, dev_id) !=
          cudaSuccess ||
      cudaDeviceGetAttribute(&shared_mem_per_block, cudaDevAttrMaxSharedMemoryPerBlock, dev_id) != cudaSuccess) {
    return NVCV_ERR_CUDA;
  }
  //  m_streamContext.nSharedMemPerBlock is of type size_t, while cudaDeviceGetAttribute only takes in int*
  stream_context.nSharedMemPerBlock = shared_mem_per_block;
  return NVCV_SUCCESS;
}

/********************************************************************************
 * ParseMyArgs
 ********************************************************************************/
static int ParseMyArgs(int argc, char** argv) {
  // query NVAR_MODEL_DIR environment variable first before checking the command line arguments
  const char* model_path = getenv("NVAR_MODEL_DIR");
  if (model_path) {
    FLAG_modelPath = model_path;
  }

  int errs = 0;
  for (--argc, ++argv; argc--; ++argv) {
    bool help;
    const char* arg = *argv;
    if (arg[0] != '-') {
      continue;
    } else if ((arg[1] == '-') &&                                               //
               (GetFlagArgVal("verbose", arg, &FLAG_verbose) ||                 //
                GetFlagArgVal("in_src", arg, &FLAG_inSrc) ||                    //
                GetFlagArgVal("bg_img", arg, &FLAG_inBgImg) ||                  //
                GetFlagArgVal("in_drv", arg, &FLAG_inDrv) ||                    //
                GetFlagArgVal("out", arg, &FLAG_outFile) ||                     //
                GetFlagArgVal("out_file", arg, &FLAG_outFile) ||                //
                GetFlagArgVal("offline_mode", arg, &FLAG_offlineMode) ||        //
                GetFlagArgVal("capture_outputs", arg, &FLAG_captureOutputs) ||  //
                GetFlagArgVal("cam_res", arg, &FLAG_camRes) ||                  //
                GetFlagArgVal("codec", arg, &FLAG_captureCodec) ||              //
                GetFlagArgVal("camera", arg, &FLAG_cameraID) ||                 //
                GetFlagArgVal("landmarks", arg, &FLAG_landmarks) ||             //
                GetFlagArgVal("log", arg, &FLAG_log) ||                         //
                GetFlagArgVal("log_level", arg, &FLAG_logLevel) ||              //
                GetFlagArgVal("model_path", arg, &FLAG_modelPath) ||            //
                GetFlagArgVal("mode", arg, &FLAG_mode) ||                       //
                GetFlagArgVal("model_sel", arg, &FLAG_modelSel) ||              //
                GetFlagArgVal("show_bbox", arg, &FLAG_showBBox) ||              //
                GetFlagArgVal("show_drive", arg, &FLAG_showDrive) ||            //
                GetFlagArgVal("ignore_alpha", arg, &FLAG_ignoreAlpha) ||        //
                GetFlagArgVal("frame_selection", arg, &FLAG_FrameSelection)
                    )) {
      continue;
    } else if (GetFlagArgVal("help", arg, &help)) {
      Usage();
      return -1;
    } else if (arg[1] != '-') {
      for (++arg; *arg; ++arg) {
        if (*arg == 'v') {
          FLAG_verbose = true;
        } else {
          // printf("Unknown flag: \"-%c\"\n", *arg);
        }
      }
      continue;
    } else {
      printf("Unknown flag: \"%s\"\n", arg);
      Usage();
      return -2;
    }
  }
  return errs;
}

class MyTimer {
 public:
  MyTimer() { m_dt = m_dt.zero(); }                                          /**< Clear the duration to 0. */
  void Start() { m_t0 = std::chrono::high_resolution_clock::now(); }         /**< Start  the timer. */
  void Pause() { m_dt = std::chrono::high_resolution_clock::now() - m_t0; }  /**< Pause  the timer. */
  void Resume() { m_t0 = std::chrono::high_resolution_clock::now() - m_dt; } /**< Resume the timer. */
  void Stop() { Pause(); }                                                   /**< Stop   the timer. */
  double ElapsedTimeFloat() const {
    return std::chrono::duration<double>(m_dt).count();
  } /**< Report the elapsed time as a float. */
 private:
  std::chrono::high_resolution_clock::time_point m_t0;
  std::chrono::high_resolution_clock::duration m_dt;
};

class DoApp {
 public:
  enum Err {
    errNone = 0,
    errGeneral,
    errRun,
    errInitialization,
    errRead,
    errEffect,
    errParameter,
    errUnimplemented,
    errMissing,
    errVideo,
    errImageSize,
    errNotFound,
    errNoFace,
    errSDK,
    errCuda,
    errCancel,
    errCamera,
    errVideoFile,
    errSourceFile,
    errMode,
    errReset,
    errFrameSelection,
  };

  DoApp();
  ~DoApp();

  Err CreateEffect(std::string model_path);
  Err InitCamera(const char* cam_res = nullptr);
  Err InitOfflineMode(const char* in_drv_file_name = nullptr);
  Err InitOutput(const std::string out_gen_file_name);
  Err Run();
  Err Stop();

  static const char* ErrorStringFromCode(Err code);

 private:
  void ProcessKey(int key);
  void GetFPS();
  void DrawFPS(cv::Mat& img);
  bool NeedReset();
  Err SignalReset(NvCVImage* neutral_drive_image);
  void FinishReset();
  MyTimer m_frameTimer;
  double m_frameTime;
  bool m_showFPS, m_showSource, m_showDrive, m_showBbox, m_needReset;
  NvAR_FeatureHandle m_livePortraitHandle{};
  NvAR_FeatureHandle m_frameSelectionHandle{};
  CUstream m_stream{}, m_fs_stream{};
  cv::VideoCapture m_cap{};
  cv::VideoWriter m_genVideo{};
  NvAR_BBoxes* m_bboxes;
  std::vector<NvAR_Rect> m_face_boxes_data;
  NvCVImage* m_srcImgGpu;
  int m_drv_width;
  int m_drv_height;
  bool m_srcAlpha;
  bool m_replaceBg;
};

DoApp::DoApp() {
  // Make sure things are initialized properly
  m_showFPS = false;
  m_showSource = false;
  m_needReset = false;
  m_showDrive = FLAG_showDrive;
  m_showBbox = FLAG_showBBox;

  m_bboxes = new NvAR_BBoxes;
  m_face_boxes_data.resize(25);
  m_bboxes->boxes = m_face_boxes_data.data();
  m_bboxes->max_boxes = 25;
  m_bboxes->num_boxes = 0;
  m_srcImgGpu = nullptr;
  m_drv_height = 0;
  m_drv_width = 0;
  m_srcAlpha = false;
  m_replaceBg = false;
}

DoApp::~DoApp() {
  if (m_bboxes) delete m_bboxes;
  if (m_stream) {
    NvAR_CudaStreamDestroy(m_stream);
  }
  NvAR_Destroy(m_livePortraitHandle);
  if (FLAG_FrameSelection) {
    if (m_fs_stream) {
      NvAR_CudaStreamDestroy(m_fs_stream);
    }
    NvAR_Destroy(m_frameSelectionHandle);
  }
}

DoApp::Err DoApp::CreateEffect(std::string model_path) {
  NvCV_Status err = NVCV_SUCCESS;

  // load trt plugins
  err = NvAR_Create(NvAR_Feature_LivePortrait, &m_livePortraitHandle);
  if (err) {
    std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
    return Err::errSDK;
  }

  err = NvAR_CudaStreamCreate(&m_stream);
  if (err) {
    std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
    return Err::errSDK;
  }

  err = NvAR_SetCudaStream(m_livePortraitHandle, NvAR_Parameter_Config(CUDAStream), m_stream);
  if (err) {
    std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
    return Err::errSDK;
  }

  err = NvAR_SetString(m_livePortraitHandle, NvAR_Parameter_Config(ModelDir), FLAG_modelPath.c_str());
  if (err) {
    std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
    return Err::errSDK;
  }

  err = NvAR_SetU32(m_livePortraitHandle, NvAR_Parameter_Config(Mode), FLAG_mode);
  if (err) {
    std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
    return Err::errSDK;
  }

  err = NvAR_SetU32(m_livePortraitHandle, NvAR_Parameter_Config(ModelSel), FLAG_modelSel);
  if (err) {
    std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
    return Err::errSDK;
  }

  err = NvAR_SetObject(m_livePortraitHandle, NvAR_Parameter_Output(BoundingBoxes), m_bboxes, sizeof(NvAR_BBoxes));
  if (err) {
    std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
    return Err::errSDK;
  }

  err = NvAR_SetU32(m_livePortraitHandle, NvAR_Parameter_Config(CheckFaceBox), FLAG_showBBox);
  if (err) {
    return Err::errSDK;
  }

  err = NvAR_Load(m_livePortraitHandle);
  if (err) {
    std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
    return Err::errSDK;
  }

  // frame selection
  if (FLAG_FrameSelection) {
    err = NvAR_Create(NvAR_Feature_FrameSelection, &m_frameSelectionHandle);
    if (err) {
      std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
      std::cout << "Cannot create frame selection. Start with first driving frame!" << std::endl;
      FLAG_FrameSelection = FRAME_SELECTION_DISABLED;
      return Err::errNone;
    }

    err = NvAR_CudaStreamCreate(&m_fs_stream);
    if (err) {
      std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
      return Err::errSDK;
    }

    err = NvAR_SetCudaStream(m_frameSelectionHandle, NvAR_Parameter_Config(CUDAStream), m_fs_stream);
    if (err) {
      std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
      return Err::errSDK;
    }

    err = NvAR_SetString(m_frameSelectionHandle, NvAR_Parameter_Config(ModelDir), FLAG_modelPath.c_str());
    if (err) {
      std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
      return Err::errSDK;
    }

    err = NvAR_SetU32(m_frameSelectionHandle, NvAR_Parameter_Config(GoodFrameMinInterval),
                      FS_GOOD_FRAME_MIN_INTERVAL_DEFAULT);
    if (err) {
      std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
      return Err::errSDK;
    }

    err = NvAR_SetU32(m_frameSelectionHandle, NvAR_Parameter_Config(ActiveDuration), FS_ACTIVEDURATION_DEFAULT);
    if (err) {
      std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
      return Err::errSDK;
    }

    err = NvAR_SetU32(m_frameSelectionHandle, NvAR_Parameter_Config(Strategy), FRAME_SELECTION_STRATEGY_DEFAULT);
    if (err) {
      std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
      return Err::errSDK;
    }

    err = NvAR_Load(m_frameSelectionHandle);
    if (err) {
      std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
      return Err::errSDK;
    }
  }

  return Err::errNone;
}

DoApp::Err DoApp::InitOfflineMode(const char* in_drv_file_name) {
  if (m_cap.open(in_drv_file_name)) {
    m_drv_width = (int)m_cap.get(CV_CAP_PROP_FRAME_WIDTH);
    m_drv_height = (int)m_cap.get(CV_CAP_PROP_FRAME_HEIGHT);
  } else {
    printf("ERROR: Unable to open the driving video file \"%s\" \n", in_drv_file_name);
    return Err::errVideo;
  }
  if (FLAG_captureOutputs) {
    std::string bdOutputVideoName, jdOutputVideoName;
    std::string outputFilePrefix;
    size_t lastindex = std::string(FLAG_inDrv).find_last_of(".");
    outputFilePrefix = std::string(FLAG_inDrv).substr(0, lastindex);
    if (FLAG_outFile.empty()) FLAG_outFile = outputFilePrefix + "_output.mp4";
  }
  return Err::errNone;
}

DoApp::Err DoApp::InitOutput(const std::string out_gen_file_name) {
  // read source image to know the resolution
  // output image in mode2 should have the same resolution
  cv::Mat img = cv::imread(FLAG_inSrc.c_str());
  if (!img.data) {
    return errSourceFile;
  }
  NvCV_Status err;
  switch (FLAG_mode) {
    case MODE_CROP_NONE:
    case MODE_CROP_FACEBOX:
      // Query the feature to know the output image size
      err = NvAR_GetU32(m_livePortraitHandle, NvAR_Parameter_Config(NetworkOutputImgWidth), &gen_img_width);
      if (err) {
        std::cout << "Error while getting width " << NvCV_GetErrorStringFromCode(err) << std::endl;
        return Err::errSDK;
      }
      err = NvAR_GetU32(m_livePortraitHandle, NvAR_Parameter_Config(NetworkOutputImgHeight), &gen_img_height);
      if (err) {
        std::cout << "Error while getting height " << NvCV_GetErrorStringFromCode(err) << std::endl;
        return Err::errSDK;
      }
      break;
    case MODE_CROP_BLEND:  // Intentional fall-through
    case MODE_CROP_INSET_BLEND:
      gen_img_width = img.cols;
      gen_img_height = img.rows;
      break;
    default:
      std::cout << "Mode " << FLAG_mode << " is not supported." << std::endl;
      return errMode;
  }

  if (FLAG_offlineMode && FLAG_captureOutputs) {
    const int codec = StringToFourcc(FLAG_captureCodec);
    double fps = m_cap.get(CV_CAP_PROP_FPS);

    const cv::Size frame_size(gen_img_width, gen_img_height);
    if (FLAG_verbose) {
      printf("fps of generated video is %f\n", fps);
    }

    if (!m_genVideo.open(out_gen_file_name, codec, fps, frame_size)) {
      printf("ERROR: Unable to open the output video file \"%s\" \n", out_gen_file_name.c_str());
      return Err::errGeneral;
    }
  }
  return Err::errNone;
}

DoApp::Err DoApp::InitCamera(const char* cam_res) {
  int drv_w, drv_h;
  if (m_cap.open(FLAG_cameraID)) {
    if (cam_res) {
      int n;
      n = sscanf(cam_res, "%d%*[xX]%d", &drv_w, &drv_h);
      switch (n) {
        case 2:
          break;  // We have read both width and height
        case 1:
          drv_h = drv_w;
          drv_w = (int)(drv_h * (4. / 3.) + .5);
          break;
        default:
          drv_h = 0;
          drv_w = 0;
          break;
      }
      if (drv_w) m_cap.set(CV_CAP_PROP_FRAME_WIDTH, drv_w);
      if (drv_h) m_cap.set(CV_CAP_PROP_FRAME_HEIGHT, drv_h);

      drv_w = (int)m_cap.get(CV_CAP_PROP_FRAME_WIDTH);
      drv_h = (int)m_cap.get(CV_CAP_PROP_FRAME_HEIGHT);

      m_drv_width = drv_w;
      m_drv_height = drv_h;

      // openCV API(CAP_PROP_FRAME_WIDTH) to get camera resolution is not always reliable with some cameras
      cv::Mat tmp_frame;
      m_cap.read(tmp_frame);
      if (tmp_frame.empty()) return errCamera;
      if (drv_w != tmp_frame.cols || drv_h != tmp_frame.rows) {
        std::cout << "!!! warning: openCV API(CAP_PROP_FRAME_WIDTH/CV_CAP_PROP_FRAME_HEIGHT) to get camera resolution "
                     "is not trustable. Using the resolution from the actual frame"
                  << std::endl;
        m_drv_width = tmp_frame.cols;
        m_drv_height = tmp_frame.rows;
      }
    }
  } else
    return errCamera;
  return errNone;
}

bool DoApp::NeedReset() { return m_needReset; }

DoApp::Err DoApp::SignalReset(NvCVImage* neutral_drive_image) {
  if (!neutral_drive_image) {
    return errReset;
  }
  if (NvAR_SetObject(m_livePortraitHandle, NvAR_Parameter_Input(NeutralDriveImage), neutral_drive_image,
                     sizeof(NvCVImage))) {
    return errReset;
  }
  m_needReset = true;
  return errNone;
}

void DoApp::FinishReset() { m_needReset = false; }

DoApp::Err DoApp::Run(void) {
  DoApp::Err do_err = errNone;
  NvCV_Status err = NVCV_SUCCESS;
  double fps;
  NvCVImage c_src, g_src, c_drv, g_drvBGR, c_dst, g_dst, tmp;
  NvCVImage c_bgImg, g_bgImgBGR, g_compBGRA;
  cv::Mat o_drv, o_dst, comp;
  char win_name[] = "LivePortrait";

  // source image
  cv::Mat img = cv::imread(FLAG_inSrc.c_str(), cv::ImreadModes::IMREAD_UNCHANGED);
  if (!img.data) return errSourceFile;
  (void)NVWrapperForCVMat(&img, &c_src);

  // This is useful if the populated alpha channel is not background segmentation mask. User has an option to ignore it.
  if (c_src.numComponents == 4 && FLAG_ignoreAlpha) {
    std::cout << "The alpha channel of the source image will be igored." << std::endl;
    m_srcAlpha = false;
  }
  // Check if it is a four-channel RGBA image, and if the alpha channel has been implemented
  else {
    m_srcAlpha = HasNontrivialAlpha(&c_src);
  }
  // if (FLAG_mode == MODE_CROP_NONE && (c_src.width != NETWORK_OUTPUT_SIZE_W || c_src.height != NETWORK_OUTPUT_SIZE_H))
  // {
  //   std::cout << "Please use fixed 512x512 source image in mode " << FLAG_mode << std::endl;
  //   return errMode;
  // }

  if (err = NvCVImage_Alloc(&g_src, c_src.width, c_src.height, (m_srcAlpha) ? NVCV_BGRA : NVCV_BGR, NVCV_U8,
                            NVCV_CHUNKY, NVCV_GPU, 1)) {
    std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
    return Err::errSDK;
  }

  if (err = NvCVImage_Transfer(&c_src, &g_src, 1, m_stream, &tmp)) {
    std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
    return Err::errSDK;
  }

  m_srcImgGpu = &g_src;

  if (err = NvAR_SetObject(m_livePortraitHandle, NvAR_Parameter_Input(SourceImage), &g_src, sizeof(NvCVImage))) {
    std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
    return Err::errSDK;
  }

  // Background replacement image
  if (m_srcAlpha && !FLAG_inBgImg.empty()) {
    m_replaceBg = true;
  } else if (!m_srcAlpha && !FLAG_inBgImg.empty()) {
    std::cout << "Background image replacement is not supported when using RGB source image. Any provided background "
                 "image will be ignored.\n"
              << std::endl;
  } else if (m_srcAlpha && FLAG_inBgImg.empty()) {
    m_replaceBg = true;
    std::cout << "Input image is RGBA, but no background image is provided. Using grey gradient image as backround.\n"
              << std::endl;
  }

  if (m_replaceBg) {
    // Allocate a GPU buffer for the background image, which is the same size as the output buffer from LivePortrait
    // feature
    if (err = NvCVImage_Realloc(&g_bgImgBGR, gen_img_width, gen_img_height, NVCV_BGR, NVCV_U8, NVCV_CHUNKY, NVCV_GPU,
                                1)) {
      std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
      return Err::errSDK;
    }
    // background image is provided by user
    if (!FLAG_inBgImg.empty()) {
      cv::Mat bg_img;

      bg_img = cv::imread(FLAG_inBgImg.c_str());
      if (!bg_img.data) {
        return errSourceFile;
      }
      (void)NVWrapperForCVMat(&bg_img, &c_bgImg);

      // Use a temporary image if the background image needs to be resized, without letterboxing, to fit the output
      // image and maintain aspect ratio
      if (!(c_bgImg.width == gen_img_width && c_bgImg.height == gen_img_height)) {
        NvCVImage bg_origsize_gpu;
        if (err = NvCVImage_Realloc(&bg_origsize_gpu, c_bgImg.width, c_bgImg.height, NVCV_BGR, NVCV_U8, NVCV_CHUNKY,
                                    NVCV_GPU, 1)) {
          std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
          return Err::errSDK;
        }
        // Copy the background image from the CPU to the GPU
        if (err = NvCVImage_Transfer(&c_bgImg, &bg_origsize_gpu, 1, m_stream, &tmp)) {
          std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
          return Err::errSDK;
        }

        // Create an NPP stream context that NPP Image resize requires
        NppStreamContext stream_context;
        if (err = createNPPStreamContext(m_stream, stream_context)) {
          std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
          return Err::errSDK;
        }

        // Resize the background image while maintaining aspect ratio
        if (err = ResizeWithoutLetterboxing(&bg_origsize_gpu, &g_bgImgBGR, &tmp, stream_context)) {
          std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
          return Err::errSDK;
        }
      }
    } else {
      // No background image is provided, but the source has an alpha channel. So we demonstrate replacing the
      // background with a gray gradient
      NvCVImage bg_default_gradient_cpu(gen_img_width, gen_img_height, NVCV_BGR, NVCV_U8, NVCV_CHUNKY, NVCV_CPU, 0);

      // A gradient between 2 shades of gray
      NvAR_Point3f gradient_colors[2] = {{80, 80, 80}, {175, 175, 175}};
      if (err = MakeVerticalGradientBGR(gradient_colors, &bg_default_gradient_cpu)) {
        std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
        return Err::errSDK;
      }
      if (err = NvCVImage_Transfer(&bg_default_gradient_cpu, &g_bgImgBGR, 1.f, m_stream, &tmp)) {
        std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
        return Err::errSDK;
      }
    }
  }

  // drive image
  fps = m_cap.get(cv::CAP_PROP_FPS);
  if (err = NvCVImage_Alloc(&g_drvBGR, m_drv_width, m_drv_height, NVCV_BGR, NVCV_U8, NVCV_CHUNKY, NVCV_GPU, 1)) {
    std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
    return Err::errSDK;
  }
  if (err = NvAR_SetObject(m_livePortraitHandle, NvAR_Parameter_Input(DriveImage), &g_drvBGR, sizeof(NvCVImage))) {
    std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
    return Err::errSDK;
  }

  // generated image
  if (err = NvCVImage_Alloc(&c_dst, gen_img_width, gen_img_height, NVCV_BGR, NVCV_U8, NVCV_CHUNKY, NVCV_CPU, 1)) {
    std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
    return Err::errSDK;
  }


  if (err = NvCVImage_Alloc(&g_dst, gen_img_width, gen_img_height, (m_srcAlpha) ? NVCV_BGRA : NVCV_BGR, NVCV_U8,
                            NVCV_CHUNKY, NVCV_GPU, 1)) {
    std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
    return Err::errSDK;
  }
  // If generated image has a alpha channel, create a buffer to use for compositing a background image
  if (m_srcAlpha) {
    if (err =
            NvCVImage_Realloc(&g_compBGRA, g_dst.width, g_dst.height, NVCV_BGRA, NVCV_U8, NVCV_CHUNKY, NVCV_CUDA, 0)) {
      std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
      return Err::errSDK;
    }
  }
  if (err = NvAR_SetObject(m_livePortraitHandle, NvAR_Parameter_Output(GeneratedImage), &g_dst, sizeof(NvCVImage))) {
    std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
    return Err::errSDK;
  }

  // Combined composition output image
  unsigned comp_size_height = std::max((unsigned)m_drv_height, gen_img_height);
  unsigned comp_size_width = m_drv_width + gen_img_width;
  comp = cv::Mat::zeros(static_cast<int>(comp_size_height), static_cast<int>(comp_size_width),
                        CV_8UC3);  // alloc & clear TODO: too big?
  if (!FLAG_offlineMode) cv::namedWindow(win_name, CV_WINDOW_AUTOSIZE);

  if (m_genVideo.isOpened() && FLAG_offlineMode && FLAG_captureOutputs && FLAG_showDrive) {
    m_genVideo.release();
    const cv::Size frame_size(comp.cols, comp.rows);  // resize to the composition dimension
    if (!m_genVideo.open(FLAG_outFile, StringToFourcc(FLAG_captureCodec), fps, frame_size)) {
      printf("ERROR: Unable to open the output video file \"%s\" \n", FLAG_outFile.c_str());
      return Err::errGeneral;
    }
  }

  // frame selection input
  if (FLAG_FrameSelection) {
    if (err = NvAR_SetObject(m_frameSelectionHandle, NvAR_Parameter_Input(Image), &g_drvBGR, sizeof(NvCVImage))) {
      std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
      return Err::errSDK;
    }
  }

  // notice window before neutral driving image detected
  cv::Mat o_notice;

  bool first_neutral_drv_found = false;
  bool fs_active_duration_expired = false;
  unsigned frame_selector_status = 0;

  // loop
  for (unsigned frame_count = 1; m_cap.read(o_drv); ++frame_count) {
    unsigned int facebox_status;
    if (o_drv.empty()) {
      std::cout << "Error: Frame is empty\n";
      if (FLAG_offlineMode) {
        return Err::errVideoFile;
      } else {
        return Err::errCamera;
      }
    }

    NVWrapperForCVMat(&o_drv, &c_drv);  // TODO: Does the OpenCV image buffer change every frame?
    if (err = NvCVImage_Transfer(&c_drv, &g_drvBGR, 1, m_stream, &tmp)) {
      std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
      return Err::errSDK;
    }

    // run frame selection
    if (FLAG_FrameSelection) {
      if (FLAG_FrameSelection == FRAME_SELECTION_TRIGGER_ONCE && first_neutral_drv_found) {
        goto _liveportrait;
      }
      if (FLAG_FrameSelection == FRAME_SELECTION_TRIGGER_MANY && fs_active_duration_expired) {
        goto _liveportrait;
      }

      if (err = NvAR_Run(m_frameSelectionHandle)) {
        std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
        return Err::errSDK;
      }
      if (err =
              NvAR_GetU32(m_frameSelectionHandle, NvAR_Parameter_Output(FrameSelectorStatus), &frame_selector_status)) {
        std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
        return Err::errSDK;
      }
      if (frame_selector_status == NVAR_FRAME_SELECTOR_SUCCESS) {
        if (!first_neutral_drv_found) {
          first_neutral_drv_found = true;
        }
        // reset live portrait when a new neutral driving image is found
        if (SignalReset(&g_drvBGR)) {
          return Err::errReset;
        }
        // save good neutral driving frame
        if (FLAG_offlineMode && FLAG_verbose) {
          saveImage(frame_count, o_drv);
        }
      } else if (frame_selector_status == NVAR_FRAME_SELECTOR_ACTIVE_DURATION_EXPIRED) {
        fs_active_duration_expired = true;
        if (!first_neutral_drv_found) {
          std::cout << "!!! warning: no good frame has been selected before active duration expired" << std::endl;
          return Err::errFrameSelection;
        }
      } else {
        if (!first_neutral_drv_found) {
          if (!FLAG_offlineMode) {
            o_notice = o_drv.clone();
            cv::putText(o_notice,
                        "Please maintain neutral head pose, straight gaze and neutral facial expression to trigger",
                        cv::Point(0, o_notice.rows - 10), cv::FONT_HERSHEY_DUPLEX, 0.4, CV_RGB(118, 185, 0), 1);
            cv::imshow(win_name, o_notice);
          }
          goto _next_frame;
        }
      }
    }  // if (FLAG_FrameSelection)

  _liveportrait:
    err = NvAR_Run(m_livePortraitHandle);
    switch (err) {
      case NVCV_SUCCESS:
      case NVCV_ERR_CONVERGENCE:
      case NVCV_ERR_NOTHINGRENDERED:
        // If nothing is rendered (e.g. no face is detected in drive video), or if the movement was too difficult
        // to track we should still continue, and not exit the app.
        break;
      default:
        std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
        return Err::errSDK;
    }

    // Composite background image onto LivePortrait output, if the src image, and hence, output image, have alpha
    // channel
    if (m_replaceBg) {
      // Composite the LP output as foreground, background image as background, using the LP output matte, into a
      // composite BGRA image
      if (err = NvCVImage_CompositeRect(&g_dst, 0, &g_bgImgBGR, 0, &g_dst, 0, &g_compBGRA, 0, m_stream)) {
        std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
        return Err::errSDK;
      }
      // Copy composite BGRA output from GPU to CPU
      if (err = NvCVImage_Transfer(&g_compBGRA, &c_dst, 1.f, m_stream, &tmp)) {
        std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
        return Err::errSDK;
      }
    } else {
      // Copy RGB output from GPU to CPU
      if (err = NvCVImage_Transfer(&g_dst, &c_dst, 1.f, m_stream, &tmp)) {
        std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
        return Err::errSDK;
      }
    }

    // Make composite
    (void)CVWrapperForNvCVImage(&c_dst, &o_dst);

    facebox_status = LP_FACEBOX_CHECK_OK;
    if (FLAG_showBBox) {
      if (err = NvAR_GetU32(m_livePortraitHandle, NvAR_Parameter_Output(FaceBoxStatus), &facebox_status)) {
        std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
        return Err::errSDK;
      }
    }
    if (FLAG_verbose) {
      if (facebox_status != LP_FACEBOX_CHECK_OK) {
        printf("Warning! facebox_status : %s\n", GetFaceBoxStatusAsString(facebox_status).c_str());
      }
    }

    if (m_showBbox) {
      auto facebox_color = CV_RGB(118, 185, 0);
      std::string facebox_message = "";
      if (facebox_status != LP_FACEBOX_CHECK_OK) {
        if (facebox_status == LP_FACEBOX_CHECK_BAD_AREA) {
          facebox_color = CV_RGB(256, 256, 0);
          facebox_message = "Move closer!";
        } else if (facebox_status == LP_FACEBOX_CHECK_CLOSE_TO_BORDER) {
          facebox_color = CV_RGB(256, 256, 0);
          facebox_message = "Move to center!";
        } else {
          facebox_color = CV_RGB(256, 0, 0);
          facebox_message = "Move to center!";
        }
        cv::putText(o_drv, facebox_message, cv::Point(lroundf(m_bboxes->boxes[0].x), lroundf(m_bboxes->boxes[0].y - 5)),
                    cv::FONT_HERSHEY_DUPLEX, 0.7, facebox_color, 1);
      }
      cv::Rect rect(static_cast<unsigned>(m_bboxes->boxes[0].x), static_cast<unsigned>(m_bboxes->boxes[0].y),
                    static_cast<unsigned>(m_bboxes->boxes[0].width), static_cast<unsigned>(m_bboxes->boxes[0].height));
      // LP reset happened. Highlight BBOX with thicker lines
      if (NeedReset()) {
        cv::rectangle(o_drv, rect, facebox_color, 8);
        cv::putText(o_drv, "reset", cv::Point(lroundf(m_bboxes->boxes[0].x), lroundf(m_bboxes->boxes[0].y - 5)),
                    cv::FONT_HERSHEY_DUPLEX, 0.7, facebox_color, 2);
      } else {
        cv::rectangle(o_drv, rect, facebox_color, 2);
      }
    }
    cv::putText(o_drv, "driving video", cv::Point(60, m_drv_height - 20), cv::FONT_HERSHEY_DUPLEX, 1.0,
                CV_RGB(118, 185, 0), 1);

    if (FLAG_verbose) {
      cv::putText(o_dst, "generated video", cv::Point(gen_img_width / 2 - 60, gen_img_height - 20),
                  cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(118, 185, 0), 1);
    }
    if (m_showDrive) {
      o_drv.copyTo(comp(cv::Rect(0, 0, m_drv_width, m_drv_height)));
      o_dst.copyTo(comp(cv::Rect(m_drv_width, 0, gen_img_width, gen_img_height)));
    }

    if (!FLAG_offlineMode) {
      if (m_showFPS) {
        if (m_showDrive) {
          DrawFPS(comp);
        } else {
          DrawFPS(o_dst);
        }
      }
      if (m_showDrive) {
        cv::imshow(win_name, comp);
      } else {
        cv::imshow(win_name, o_dst);
      }
    }

    if (FLAG_captureOutputs) {
      FLAG_showDrive ? m_genVideo.write(comp) : m_genVideo.write(o_dst);
    }

    // clear reset in each iteration
    // In each Run, SDK detects if a NvCVImage object is associated with the SourceImage API. If yes, SDK considers a
    // new source image has been passed in. If you don't intend to update the source image, make sure to set the API to
    // nullptr so that in the next Run, SDK can reuse the previous source image.
    if (NeedReset()) {
      FinishReset();
    }
  _next_frame:
    if (!FLAG_offlineMode) {
      int n = cv::waitKey(1);
      if (n >= 0) {
        static const int ESC_KEY = 27;
        if (n == ESC_KEY) break;
        ProcessKey(n);
      }
    }
  }  // for()

  return do_err;
}

DoApp::Err DoApp::Stop(void) {
  m_cap.release();
  if (FLAG_offlineMode && FLAG_captureOutputs) {
    m_genVideo.release();
  }
  return Err::errNone;
}

void DoApp::ProcessKey(int key) {
  switch (key) {
    case 'B':
    case 'b':
      m_showBbox = !m_showBbox;
      break;
    case 'D':
    case 'd':
      m_showDrive = !m_showDrive;
      break;
    case 'F':
    case 'f':
      m_showFPS = !m_showFPS;
      break;
    default:
      break;
  }
}

void DoApp::GetFPS() {
  const float timeConstant = 16.f;
  m_frameTimer.Stop();
  float t = (float)m_frameTimer.ElapsedTimeFloat();
  if (t < 100.f) {
    if (m_frameTime)
      m_frameTime += (t - m_frameTime) * (1.f / timeConstant);  // 1 pole IIR filter
    else
      m_frameTime = t;
  } else {              // Ludicrous time interval; reset
    m_frameTime = 0.f;  // WAKE UP
  }
  m_frameTimer.Start();
}

void DoApp::DrawFPS(cv::Mat& img) {
  GetFPS();
  if (m_frameTime && m_showFPS) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%.1f", 1. / m_frameTime);
    cv::putText(img, buf, cv::Point(img.cols - 80, img.rows - 10), cv::FONT_HERSHEY_SIMPLEX, 1,
                cv::Scalar(255, 255, 255), 1);
  }
}

const char* DoApp::ErrorStringFromCode(DoApp::Err code) {
  struct LUTEntry {
    Err code;
    const char* str;
  };
  static const LUTEntry lut[] = {
      {errNone, "no error"},
      {errGeneral, "an error has occured"},
      {errRun, "an error has occured while the feature is running"},
      {errInitialization, "Initializing Face Engine failed"},
      {errRead, "an error has occured while reading a file"},
      {errEffect, "an error has occured while creating a feature"},
      {errParameter, "an error has occured while setting a parameter for a feature"},
      {errUnimplemented, "the feature is unimplemented"},
      {errMissing, "missing input parameter"},
      {errVideo, "no video source has been found"},
      {errImageSize, "the image size cannot be accommodated"},
      {errNotFound, "the item cannot be found"},
      {errNoFace, "no face has been found"},
      {errSDK, "an SDK error has occurred"},
      {errCuda, "a CUDA error has occurred"},
      {errCancel, "the user cancelled"},
      {errCamera, "unable to connect to the camera"},
      {errVideoFile, "unable to open driving video file"},
      {errSourceFile, "unable to open source image file"},
      {errMode, "unsupported mode or wrong source image size in that mode"},
      {errReset, "unable to reset live portrait"},
      {errFrameSelection, "unable to find a single good frame from the driving video"},
  };
  for (const LUTEntry* p = lut; p < &lut[sizeof(lut) / sizeof(lut[0])]; ++p)
    if (p->code == code) return p->str;
  static char msg[18];
  snprintf(msg, sizeof(msg), "error #%d", code);
  return msg;
}

char* g_nvARSDKPath = NULL;

/********************************************************************************
 * main
 ********************************************************************************/

int main(int argc, char** argv) {
  // Parse the arguments
  if (ParseMyArgs(argc, argv)) {
    return 100;
  }
  DoApp app;
  DoApp::Err do_err = DoApp::Err::errNone;

  NvCV_Status err = NvAR_ConfigureLogger(FLAG_logLevel, FLAG_log.c_str(), nullptr, nullptr);
  if (NVCV_SUCCESS != err)
    printf("%s: while configuring logger to \"%s\"\n", NvCV_GetErrorStringFromCode(err), FLAG_log.c_str());

  if (FLAG_modelPath.empty()) {
    printf(
        "WARNING: Model path not specified. Please set --model_path=/path/to/trt/and/face/models, "
        "SDK will attempt to load the models from NVAR_MODEL_DIR environment variable, "
        "please restart your application after the SDK Installation. \n");
  }

  do_err = app.CreateEffect(FLAG_modelPath);
  BAIL_IF_ERR(do_err);

  if (FLAG_inSrc.empty()) {
    do_err = DoApp::errMissing;
    printf("ERROR: %s, please specify your source portrait file using --in_src \n", app.ErrorStringFromCode(do_err));
    goto bail;
  }

  if (FLAG_offlineMode) {
    if (FLAG_inDrv.empty()) {
      do_err = DoApp::errMissing;
      printf("ERROR: %s, please specify driving video file using --in_drv in offline mode\n",
             app.ErrorStringFromCode(do_err));
      goto bail;
    }
    do_err = app.InitOfflineMode(FLAG_inDrv.c_str());
  } else {
    do_err = app.InitCamera(FLAG_camRes.c_str());
  }

  do_err = app.InitOutput(FLAG_outFile);
  BAIL_IF_ERR(do_err);

  do_err = app.Run();
  BAIL_IF_ERR(do_err);

bail:
  if (do_err) printf("ERROR: %s\n", app.ErrorStringFromCode(do_err));
  app.Stop();
  return (int)do_err;
}
