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

#include <cuda.h>
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

#include "npp.h"
#include "nvAR.h"
#include "nvARSpeechLivePortrait.h"
#include "nvAR_defs.h"
#include "nvCVOpenCV.h"
#include "opencv2/opencv.hpp"
#include "waveReadWrite.h"

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

#define NETWORK_OUTPUT_SIZE_W 512
#define NETWORK_OUTPUT_SIZE_H 512

#define NUMBER_FEATURES_PER_FRAME 40
#define NUMBER_FRAMES_PER_BUFFER 41

#define MODEL_SEL_PERF 0
#define MODEL_SEL_QUAL 1

#define MODE_CROP_NONE 0
#define MODE_CROP_FACEBOX 1
#define MODE_CROP_BLEND 2
#define MODE_CROP_INSET_BLEND 3

#define OUTPUT_VIDEO_FRAME_RATE (1.0 / 33.0e-3)  // 1.0 / (33 ms / frame)

// gen_img_xx might be overwritten in InitOutput().
// This image size correspond to the final output and it may differ depending on which mode is used in the feature
unsigned int gen_img_width = NETWORK_OUTPUT_SIZE_W;
unsigned int gen_img_height = NETWORK_OUTPUT_SIZE_H;
/********************************************************************************
 * Command-line arguments
 ********************************************************************************/
// clang-format off
bool          FLAG_debug = false,
              FLAG_verbose = false,
              FLAG_offlineMode = true,                 // reads driving audio from file if set to true; webcam mode if set to false; currently only offline mode supported
              FLAG_captureOutputs = true,              // write generated video to file if set to true. only in offline mode
              FLAG_enableLookAway = false;
int           FLAG_modelSel = MODEL_SEL_QUAL;
int           FLAG_blinkFrequency = 15;                // Set to default value for blinks per minute
int           FLAG_blinkDuration = 6;                  // Set to default value for blink duration in frames
int           FLAG_mode = 1;                           // set to default value for Mode
unsigned      FLAG_lookAwayOffsetMax = 20;             // set to default value for Gaze Lookaway offset maximum
unsigned      FLAG_lookAwayIntervalRange = 3;          // set to default value for Gaze Lookaway interval range
unsigned      FLAG_lookAwayIntervalMin = 8;            // set to default value for Gaze Lookaway Minimum interval
unsigned      FLAG_headPoseMode = 2;                   // set to default value for Head Pose mode
unsigned      FLAG_logLevel = NVCV_LOG_ERROR;
double        FLAG_mouthExpressionMultiplier = -1.f;   // We have a negative value here on purpose. The expected range for user input 
                                                       // is [1.0f, 1.6f]. If no user input, sample app will set the optimal value: 1.4f. This API setter is also optional 
                                                       // in the sample app as SDK has optimal setting internally.
double        FLAG_mouthExpressionBase = -1.f;         // We have a negative value here on purpose. The expected range for user input 
                                                       // is [0.0f, 1.0f]. If no user input, sample app will set the optimal value: 0.3f. This API setter is also optional 
                                                       // in the sample app as SDK has optimal setting internally.
double        FLAG_headPoseMultiplier = -1.f;	         // We have a negative value here on purpose. The expected range for user input 
                                                       // is [0.0f, 1.0f]. If no user input, sample app will set the optimal value: 1.0f. This API setter is also optional 
                                                       // in the sample app as SDK has optimal setting internally.
std::string   FLAG_outDir,
              FLAG_inSrc,
              FLAG_inDrv,
              FLAG_outFile,
              FLAG_modelPath,
              FLAG_captureCodec = "avc1",
              FLAG_log = "stderr";
// clang-format on

bool CheckResult(NvCV_Status nvErr, unsigned line) {
  if (NVCV_SUCCESS == nvErr) return true;
  std::cout << "ERROR: " << NvCV_GetErrorStringFromCode(nvErr) << ", line " << line << std::endl;
  return false;
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
  // todo: clear the misuse of square and curly braces
  printf(
      "SpeechLivePortraitApp [<args> ...]\n"
      "where <args> are\n"
      " --verbose[=(true|false)]              report interesting info\n"
      " --debug[=(true|false)]                report debugging info\n"
      " --log=<file>                          log SDK errors to a file, \"stderr\" or \"\" (default stderr)\n"
      " --log_level=<N>                       the desired log level: {0, 1, 2, 3} = {FATAL, ERROR, WARNING, INFO}, "
      "respectively (default 1)\n"
      " --model_path=<path>                   specify the directory containing the TRT models\n"
      " --capture_outputs[=(true|false)]      write generated video to file if set to true. only in offline mode\n"
      " --codec=<fourcc>                      FOURCC code for the desired codec (default H264)\n"
      " --in_src=<file>                       specify the input source file (portrait image)\n"
      " --in_drv=<file>                       specify the input driving file. only in offline mode\n"
      " --out=<file>                          specify the output file. only in offline mode and capture_outputs is "
      "true.\n"
      " --model_sel=[=n]                      select the model. 0 for perf, 1 for quality. Default is 1\n"
      " --blink_duration=[=n]                 duration of Eye Blinks in Frames. Default is 6\n"
      " --blink_frequency=[=n]                frequency of blinks per minute. Default is 15\n"
      " --mode[=n]                            cropping mode. Choose from MODE_CROP_FACEBOX(1), MODE_CROP_BLEND(2) and "
      "MODE_CROP_INSET_BLEND(3). Default is 1. \n"
      " --head_pose_mode[=n]                  select the mode for head pose. 1 for source image head pose, 2 for "
      "predefined head pose. 3 for user-provided head pose Default: 2\n"
      " --enable_look_away[=(true|false)]     enables random look away to avoid staring, Default: False\n"
      " --look_away_offset_max=[=n]           maximum integer value in degree of gaze offset when lookaway is enabled. "
      "Default: 20\n"
      " --look_away_interval_min=[=n]         minimum interval in seconds (integer value) for triggering the lookaway "
      "event. Default: 8 \n"
      " --look_away_interval_range=[=n]       range of the interval in seconds (integer value) for triggering the "
      "lookaway event. Default: 3 \n"
      "                                       Note that the lookaway event will be occurred every "
      "[look_away_interval_min, look_away_interval_min + look_away_interval_range] \n"
      "                                       Default: the look away event will trigger every rand([8, 11]) second. \n"
      " --mouth_expression_multiplier=[=n]    Specifies the degree of exaggeration for mouth movements. Range: [1.0f, "
      "1.6f]  Default: 1.4f. Higher values result in more exaggerated mouth motions. \n"
      " --mouth_expression_base=[=n]          Defines the base openness of the mouth when idle (i.e., silence audio "
      "input). Range: [0.0f, 1.0f]  Default: 0.3f. Higher values lead to a more open mouth appearance during the idle "
      "state. \n"
      " --head_pose_multiplier=[=n]           multiplier to dampen the head animation and the range is [0.0f, 1.0f] "
      "Only applicable to HeadPoseMode=2. Default: 1.f \n");
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

bool GetFlagArgVal(const char* flag, const char* arg, double* val) {
  const char* valStr;
  bool success = GetFlagArgVal(flag, arg, &valStr);
  if (success) *val = valStr ? strtod(valStr, nullptr) : 1.0;
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
    } else if ((arg[1] == '-') &&                                                                      //
               (GetFlagArgVal("verbose", arg, &FLAG_verbose) ||                                        //
                GetFlagArgVal("debug", arg, &FLAG_debug) ||                                            //
                GetFlagArgVal("log", arg, &FLAG_log) ||                                                //
                GetFlagArgVal("log_level", arg, &FLAG_logLevel) ||                                     //
                GetFlagArgVal("in_src", arg, &FLAG_inSrc) ||                                           //
                GetFlagArgVal("in_drv", arg, &FLAG_inDrv) ||                                           //
                GetFlagArgVal("out", arg, &FLAG_outFile) ||                                            //
                GetFlagArgVal("codec", arg, &FLAG_captureCodec) ||                                     //
                GetFlagArgVal("model_sel", arg, &FLAG_modelSel) ||                                     //
                GetFlagArgVal("enable_look_away", arg, &FLAG_enableLookAway) ||                        //
                GetFlagArgVal("look_away_offset_max", arg, &FLAG_lookAwayOffsetMax) ||                 //
                GetFlagArgVal("look_away_interval_range", arg, &FLAG_lookAwayIntervalRange) ||         //
                GetFlagArgVal("look_away_interval_min", arg, &FLAG_lookAwayIntervalMin) ||             //
                GetFlagArgVal("blink_frequency", arg, &FLAG_blinkFrequency) ||                         //
                GetFlagArgVal("blink_duration", arg, &FLAG_blinkDuration) ||                           //
                GetFlagArgVal("head_pose_mode", arg, &FLAG_headPoseMode) ||                            //
                GetFlagArgVal("mouth_expression_multiplier", arg, &FLAG_mouthExpressionMultiplier) ||  //
                GetFlagArgVal("mouth_expression_base", arg, &FLAG_mouthExpressionBase) ||              //
                GetFlagArgVal("head_pose_multiplier", arg, &FLAG_headPoseMultiplier) ||                //
                GetFlagArgVal("out_file", arg, &FLAG_outFile) ||                                       //
                GetFlagArgVal("mode", arg, &FLAG_mode) ||                                              //
                GetFlagArgVal("capture_outputs", arg, &FLAG_captureOutputs) ||                         //
                GetFlagArgVal("model_path", arg, &FLAG_modelPath))) {
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

class App {
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
    errAudio,
    errHeadPose,
    errImageSize,
    errNotFound,
    errNoFace,
    errSDK,
    errCuda,
    errCancel,
    errAudioFile,
    errSourceFile,
    errMode,
  };

  App();
  ~App();

  Err CreateEffect(std::string model_path);
  Err InitOfflineMode();
  Err InitOutput(const std::string out_gen_file_name);
  Err ProcessOutputVideo();
  Err Run();
  Err Stop();
  Err UpdateHeadPose(bool update_anim_index);

  static const char* ErrorStringFromCode(Err code);

 private:
  void ProcessKey(int key);
  void GetFPS();
  MyTimer m_frameTimer;
  double m_frameTime;
  cv::VideoCapture m_cap{};
  void DrawFPS(cv::Mat& img);
  void CreateHeadPoseAnimation();

  bool m_showFPS;
  NvAR_FeatureHandle m_speechLivePortraitHandle{};
  CUstream m_stream{};
  NvCVImage* m_srcImgGpu;                                   // source image
  NvCVImage m_tmp;                                          // tmp image
  NvCVImage m_c_dst, m_g_dst;                               // output image
  cv::VideoWriter m_genVideo{};                             // output video file
  int m_animation_index;                                    // the animation index for animation
  bool m_rotation;                                          // this bool is to switch between rotation and translation
  std::vector<NvAR_Quaternion> m_head_rotation_animation;   // showcase of head rotation
  std::vector<NvAR_Vector3f> m_head_translation_animation;  // showcase of head translation
  NvAR_Quaternion* m_head_rotation;
  NvAR_Vector3f* m_head_translation;
};

App::App() {
  // Make sure things are initialized properly
  m_animation_index = 0;
  m_rotation = true;
  m_frameTime = -1;
  m_showFPS = false;
  m_srcImgGpu = nullptr;
  m_head_rotation_animation.clear();
  m_head_translation_animation.clear();
  m_head_rotation = nullptr;
  m_head_translation = nullptr;
}

App::~App() {
  if (m_stream) {
    NvAR_CudaStreamDestroy(m_stream);
  }
  NvAR_Destroy(m_speechLivePortraitHandle);
}

App::Err App::CreateEffect(std::string model_path) {
  NvCV_Status err = NVCV_SUCCESS;

  // load trt plugins
  err = NvAR_Create(NvAR_Feature_SpeechLivePortrait, &m_speechLivePortraitHandle);
  if (err) {
    std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
    return Err::errSDK;
  }

  err = NvAR_CudaStreamCreate(&m_stream);
  if (err) {
    std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
    return Err::errSDK;
  }

  err = NvAR_SetCudaStream(m_speechLivePortraitHandle, NvAR_Parameter_Config(CUDAStream), m_stream);
  if (err) {
    std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
    return Err::errSDK;
  }

  err = NvAR_SetU32(m_speechLivePortraitHandle, NvAR_Parameter_Config(ModelSel), FLAG_modelSel);
  if (err) {
    std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
    return Err::errSDK;
  }

  err = NvAR_SetU32(m_speechLivePortraitHandle, NvAR_Parameter_Config(Mode), FLAG_mode);
  if (err) {
    std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
    return Err::errSDK;
  }

  err = NvAR_SetString(m_speechLivePortraitHandle, NvAR_Parameter_Config(ModelDir), FLAG_modelPath.c_str());
  if (err) {
    std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
    return Err::errSDK;
  }

  err = NvAR_SetU32(m_speechLivePortraitHandle, NvAR_Parameter_Config(BlinkDuration), FLAG_blinkDuration);
  if (err) {
    std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
    return Err::errSDK;
  }

  err = NvAR_SetU32(m_speechLivePortraitHandle, NvAR_Parameter_Config(BlinkFrequency), FLAG_blinkFrequency);
  if (err) {
    std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
    return Err::errSDK;
  }

  // If MouthExpressionMultiplier is provided, update the value. Note the range is [1.0f, 1.6f] and this value will be
  // verified in the API
  if (FLAG_mouthExpressionMultiplier != -1.f) {
    err = NvAR_SetF32(m_speechLivePortraitHandle, NvAR_Parameter_Config(MouthExpressionMultiplier),
                      (float)FLAG_mouthExpressionMultiplier);
    if (err) {
      std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
      return Err::errSDK;
    }
  }

  // If MouthExpressionBase is provided, update the value. Note the range is [0.0f, 1.0f] and this value will be
  // verified in the API
  if (FLAG_mouthExpressionBase != -1.f) {
    err = NvAR_SetF32(m_speechLivePortraitHandle, NvAR_Parameter_Config(MouthExpressionBase),
                      (float)FLAG_mouthExpressionBase);
    if (err) {
      std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
      return Err::errSDK;
    }
  }

  err = NvAR_SetU32(m_speechLivePortraitHandle, NvAR_Parameter_Config(EnableLookAway), FLAG_enableLookAway);
  if (err) {
    std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
    return Err::errSDK;
  }

  err = NvAR_SetU32(m_speechLivePortraitHandle, NvAR_Parameter_Config(LookAwayOffsetMax), FLAG_lookAwayOffsetMax);
  if (err) {
    std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
    return Err::errSDK;
  }

  // FLAG_lookAwayIntervalRange's unit is in seconds. This needs to be converted to frames for LookAwayIntervalRange
  // API.
  unsigned int lookAwayIntervalRange = int(FLAG_lookAwayIntervalRange * OUTPUT_VIDEO_FRAME_RATE);
  err = NvAR_SetU32(m_speechLivePortraitHandle, NvAR_Parameter_Config(LookAwayIntervalRange), lookAwayIntervalRange);
  if (err) {
    std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
    return Err::errSDK;
  }

  // FLAG_lookAwayIntervalMin's unit is in seconds. This needs to be converted to frames for LookAwayIntervalMin API.
  unsigned int lookAwayIntervalMin = int(FLAG_lookAwayIntervalMin * OUTPUT_VIDEO_FRAME_RATE);
  err = NvAR_SetU32(m_speechLivePortraitHandle, NvAR_Parameter_Config(LookAwayIntervalMin), lookAwayIntervalMin);
  if (err) {
    std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
    return Err::errSDK;
  }

  err = NvAR_SetU32(m_speechLivePortraitHandle, NvAR_Parameter_Config(HeadPoseMode), FLAG_headPoseMode);
  // If user-provided head pose is selected, enable the animation array
  if (FLAG_headPoseMode == 3) {
    CreateHeadPoseAnimation();
  }
  if (err) {
    std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
    return Err::errSDK;
  }

  // The HeadPoseMultiplier value can only be set when HeadPoseMode=2.
  // If HeadPoseMultiplier is provided, update the value. Note the range is [0.0. 1.0] and this value will be verified
  // in the API
  if (FLAG_headPoseMode == 2 && FLAG_headPoseMultiplier != -1.f) {
    err = NvAR_SetF32(m_speechLivePortraitHandle, NvAR_Parameter_Config(HeadPoseMultiplier),
                      (float)FLAG_headPoseMultiplier);
    if (err) {
      std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
      return Err::errSDK;
    }
  }
  err = NvAR_Load(m_speechLivePortraitHandle);
  if (err) {
    std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
    return Err::errSDK;
  }

  return Err::errNone;
}

App::Err App::InitOfflineMode() {
  if (FLAG_captureOutputs) {
    std::string bdOutputVideoName, jdOutputVideoName;
    std::string outputFilePrefix;
    size_t lastindex = std::string(FLAG_inDrv).find_last_of(".");
    outputFilePrefix = std::string(FLAG_inDrv).substr(0, lastindex);
    if (FLAG_outFile.empty()) FLAG_outFile = outputFilePrefix + "_output.mp4";
  }
  return Err::errNone;
}

App::Err App::InitOutput(const std::string out_gen_file_name) {
  // read source image to know the resolution
  // output image in mode2 should have the same resolution
  cv::Mat img = cv::imread(FLAG_inSrc.c_str());
  if (!img.data) {
    return errSourceFile;
  }
  switch (FLAG_mode) {
    case MODE_CROP_NONE:
    case MODE_CROP_FACEBOX:
      gen_img_width = NETWORK_OUTPUT_SIZE_W;
      gen_img_height = NETWORK_OUTPUT_SIZE_H;
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
    double fps = OUTPUT_VIDEO_FRAME_RATE;  // 1.0 / (33 ms / frame)

    const cv::Size frame_size(gen_img_width, gen_img_height);
    if (FLAG_debug) {
      printf("fps of generated video is %f\n", fps);
    }

    if (!m_genVideo.open(out_gen_file_name, codec, fps, frame_size)) {
      printf("ERROR: Unable to open the output video file \"%s\" \n", out_gen_file_name.c_str());
      return Err::errGeneral;
    }
  }
  return Err::errNone;
}

App::Err App::Run(void) {
  App::Err app_err = errNone;
  NvCV_Status err = NVCV_SUCCESS;
  NvCVImage c_src, g_src, g_drvBGR, tmp;
  NvCVImage c_bgImg, g_bgImgBGR, g_compBGRA;

  // source image
  cv::Mat img = cv::imread(FLAG_inSrc.c_str(), cv::ImreadModes::IMREAD_UNCHANGED);
  if (!img.data) {
    return errSourceFile;
  }
  (void)NVWrapperForCVMat(&img, &c_src);

  if (err = NvCVImage_Alloc(&g_src, c_src.width, c_src.height, NVCV_BGR, NVCV_U8, NVCV_CHUNKY, NVCV_GPU, 1)) {
    std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
    return Err::errSDK;
  }

  if (err = NvCVImage_Transfer(&c_src, &g_src, 1, m_stream, &tmp)) {
    std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
    return Err::errSDK;
  }

  m_srcImgGpu = &g_src;

  if (err = NvAR_SetObject(m_speechLivePortraitHandle, NvAR_Parameter_Input(SourceImage), &g_src, sizeof(NvCVImage))) {
    std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
    return Err::errSDK;
  }

  // get input sample rate
  unsigned int input_sample_rate_ = 0;
  if (err = NvAR_GetU32(m_speechLivePortraitHandle, NvAR_Parameter_Config(SampleRate), &input_sample_rate_)) {
    std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
    return Err::errSDK;
  }

  // get input num channels
  unsigned int num_channels_ = 0;
  if (err = NvAR_GetU32(m_speechLivePortraitHandle, NvAR_Parameter_Config(NumChannels), &num_channels_)) {
    std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
    return Err::errSDK;
  }

  // set audio frame
  uint32_t samples_per_frame = 0;
  if (err = NvAR_GetU32(m_speechLivePortraitHandle, NvAR_Parameter_Config(SamplesPerFrame), &samples_per_frame)) {
    std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
    return Err::errSDK;
  }

  // get initial audio frames before the 1st video frame been generated
  uint32_t init_latency_frame_cnt = 0;
  if (err = NvAR_GetU32(m_speechLivePortraitHandle, NvAR_Parameter_Config(NumInitialFrames), &init_latency_frame_cnt)) {
    std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
    return Err::errSDK;
  }

  unsigned int input_num_samples = 0;
  std::vector<float>* input_wav_samples;

  std::string audio_path = FLAG_inDrv.c_str();

  // read the input audio file
  if (!ReadWavFile(audio_path, input_sample_rate_, num_channels_, &input_wav_samples, &input_num_samples, nullptr,
                   samples_per_frame,  // align input samples to samples_per_frame i.e. 528
                   FLAG_debug || FLAG_verbose)) {
    std::cerr << "Unable to read wav file: " << "sample_speech.wav" << std::endl;
  }

  // ReadWavFile returns input_wav_samples aligned to samples_per_frame by padding 0s in last frame
  size_t input_audio_frame_cnt = (*input_wav_samples).size() / samples_per_frame;

  // generated image
  if (err = NvCVImage_Alloc(&m_c_dst, gen_img_width, gen_img_height, NVCV_BGR, NVCV_U8, NVCV_CHUNKY, NVCV_CPU, 1)) {
    std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
    return Err::errSDK;
  }
  if (err = NvCVImage_Alloc(&m_g_dst, gen_img_width, gen_img_height, NVCV_BGR, NVCV_U8, NVCV_CHUNKY, NVCV_GPU, 1)) {
    std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
    return Err::errSDK;
  }

  if (err = NvAR_SetObject(m_speechLivePortraitHandle, NvAR_Parameter_Output(GeneratedImage), &m_g_dst,
                           sizeof(NvCVImage))) {
    std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
    return Err::errSDK;
  }

  // Combined composition output image

  /**
   * The feature is implemented as a pipeline with 3 stages: PRIME, PUMP, and FLUSH.
   * In the PRIME phase, audio frames are fed in without any expectation of getting any video frames out.
   * In the PUMP phase, a video frame is retrieved for every audio frame supplied.
   * In the FLUSH phase, there are no more audio frames left, but there are still more video frames to retrieve from the
   * pipeline.
   */

  size_t flush_frame_cnt = init_latency_frame_cnt;

  // Prime: feed init_latency_frame_cnt audio frames up till first video output frame is returned
  for (unsigned int i = 0; i < init_latency_frame_cnt && i < input_audio_frame_cnt; i++) {
    std::vector<float>::iterator offset = (*input_wav_samples).begin() + i * samples_per_frame;
    std::vector<float> audio_frame(offset, offset + samples_per_frame);

    if (err = NvAR_SetF32Array(m_speechLivePortraitHandle, NvAR_Parameter_Input(AudioFrameBuffer), audio_frame.data(),
                               samples_per_frame)) {
      std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
      return Err::errSDK;
    }

    // the headpose will only get updated in headposemode=3
    // false: do not increment the index during Prime
    m_animation_index = 0;
    if (UpdateHeadPose(false) != Err::errNone) return app_err;

    if (err = NvAR_Run(m_speechLivePortraitHandle)) {
      std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
      return Err::errSDK;
    }
    // output video frame is invalid; hence no action
  }

  // Pump: submit the rest of the audio frames, retrieving a video frame for each one.
  for (int i = init_latency_frame_cnt; i < input_audio_frame_cnt; i++) {
    std::vector<float>::iterator offset = (*input_wav_samples).begin() + i * samples_per_frame;
    std::vector<float> audio_frame(offset, offset + samples_per_frame);
    if (err = NvAR_SetF32Array(m_speechLivePortraitHandle, NvAR_Parameter_Input(AudioFrameBuffer), audio_frame.data(),
                               samples_per_frame)) {
      std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
      return Err::errSDK;
    }

    // the headpose will only get updated in headposemode=3
    // true: increment the index during Pump
    if (UpdateHeadPose(true) != Err::errNone) return app_err;

    if (err = NvAR_Run(m_speechLivePortraitHandle)) {
      std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
      return Err::errSDK;
    }
    app_err = ProcessOutputVideo();
  }

  // Flush: retrieve the last flush_frame_cnt video frames from the pipeline
  for (int i = 0; i < flush_frame_cnt; i++) {
    std::vector<float> audio_frame(samples_per_frame, 0.0);
    if (err = NvAR_SetF32Array(m_speechLivePortraitHandle, NvAR_Parameter_Input(AudioFrameBuffer), audio_frame.data(),
                               samples_per_frame)) {
      std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
      return Err::errSDK;
    }

    // the headpose will only get updated in headposemode=3
    // true: increment the index during Flush
    if (UpdateHeadPose(true) != Err::errNone) return app_err;

    if (err = NvAR_Run(m_speechLivePortraitHandle)) {
      std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
      return Err::errSDK;
    }
    app_err = ProcessOutputVideo();
  }
  return app_err;
}

App::Err App::UpdateHeadPose(bool update_anim_index) {
  App::Err app_err = errNone;
  NvCV_Status err = NVCV_SUCCESS;

  if (FLAG_headPoseMode != 3) return Err::errNone;

  if (m_rotation) {
    m_head_rotation = &m_head_rotation_animation[m_animation_index];
    if (err = NvAR_SetObject(m_speechLivePortraitHandle, NvAR_Parameter_Input(HeadPoseRotation), m_head_rotation,
                             sizeof(NvAR_Quaternion))) {
      std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
      return Err::errHeadPose;
    }
  } else {
    m_head_translation = &m_head_translation_animation[m_animation_index];
    if (err = NvAR_SetObject(m_speechLivePortraitHandle, NvAR_Parameter_Input(HeadPoseTranslation), m_head_translation,
                             sizeof(NvAR_Vector3f))) {
      std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
      return Err::errHeadPose;
    }
  }

  if (update_anim_index) {
    int max_frame;
    if (m_rotation) {
      max_frame = (int)m_head_rotation_animation.size() - 1;
    } else {
      max_frame = (int)m_head_translation_animation.size() - 1;
    }

    // increment the index or reset the index if it reaches the end
    if (m_animation_index == max_frame) {
      m_animation_index = 0;
      // switch from rotation to translation
      // this is just an example to switch between animations
      m_rotation = !m_rotation;
    } else {
      m_animation_index += 1;
    }
  }

  return app_err;
}

// Todo: separate the frame compositing from the video writing.
App::Err App::ProcessOutputVideo()

{
  NvCV_Status err = NVCV_SUCCESS;
  cv::Mat o_dst;

  if (err = NvCVImage_Transfer(&m_g_dst, &m_c_dst, 1.f, m_stream, &m_tmp)) {
    std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
    return Err::errSDK;
  }

  (void)CVWrapperForNvCVImage(&m_c_dst, &o_dst);

  if (FLAG_debug) {
    cv::putText(o_dst, "generated video", cv::Point(gen_img_width / 2 - 60, gen_img_height - 20),
                cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(118, 185, 0), 1);
  }

  if (!FLAG_offlineMode) {
    if (m_showFPS) {
      DrawFPS(o_dst);
    }
  }

  if (m_genVideo.isOpened() && FLAG_offlineMode && FLAG_captureOutputs) {
    m_genVideo.write(o_dst);
  }
  return Err::errNone;
}

App::Err App::Stop(void) {
  m_cap.release();
  if (m_genVideo.isOpened()) {
    m_genVideo.release();
  }
  return Err::errNone;
}

void App::ProcessKey(int key) {
  switch (key) {
    case 'F':
    case 'f':
      m_showFPS = !m_showFPS;
      break;
    default:
      break;
  }
}

void App::CreateHeadPoseAnimation() {
  // this function is to show case how to use head pose mode=3.
  // the animation will be reset in the UpdateHeadPose() when it reach to the end
  m_animation_index = 0;
  // This rotation aniamtion is an example of how to use head pose mode3
  //   0 -  59 frame   -> Pitch [- 6 - + 6] degree
  //  60 - 119 frame   -> Yaw   [- 8 - + 8] degree
  // 120 - 179 frame   -> Roll  [- 5 - + 5] degree
  m_head_rotation_animation = {
      // Rotation [Qx, Qy, Qz, Qw]
      {0.00000f, 0.00000f, 0.00000f, 1.00000f},   // frame: 0
      {0.00349f, 0.00000f, 0.00000f, 0.99999f},   // frame: 1
      {0.00698f, 0.00000f, 0.00000f, 0.99998f},   // frame: 2
      {0.01047f, 0.00000f, 0.00000f, 0.99995f},   // frame: 3
      {0.01396f, 0.00000f, 0.00000f, 0.99990f},   // frame: 4
      {0.01745f, 0.00000f, 0.00000f, 0.99985f},   // frame: 5
      {0.02094f, 0.00000f, 0.00000f, 0.99978f},   // frame: 6
      {0.02443f, 0.00000f, 0.00000f, 0.99970f},   // frame: 7
      {0.02792f, 0.00000f, 0.00000f, 0.99961f},   // frame: 8
      {0.03141f, 0.00000f, 0.00000f, 0.99951f},   // frame: 9
      {0.03490f, 0.00000f, 0.00000f, 0.99939f},   // frame: 10
      {0.03839f, 0.00000f, 0.00000f, 0.99926f},   // frame: 11
      {0.04188f, 0.00000f, 0.00000f, 0.99912f},   // frame: 12
      {0.04536f, 0.00000f, 0.00000f, 0.99897f},   // frame: 13
      {0.04885f, 0.00000f, 0.00000f, 0.99881f},   // frame: 14
      {0.04885f, 0.00000f, 0.00000f, 0.99881f},   // frame: 15
      {0.04536f, 0.00000f, 0.00000f, 0.99897f},   // frame: 16
      {0.04188f, 0.00000f, 0.00000f, 0.99912f},   // frame: 17
      {0.03839f, 0.00000f, 0.00000f, 0.99926f},   // frame: 18
      {0.03490f, 0.00000f, 0.00000f, 0.99939f},   // frame: 19
      {0.03141f, 0.00000f, 0.00000f, 0.99951f},   // frame: 20
      {0.02792f, 0.00000f, 0.00000f, 0.99961f},   // frame: 21
      {0.02443f, 0.00000f, 0.00000f, 0.99970f},   // frame: 22
      {0.02094f, 0.00000f, 0.00000f, 0.99978f},   // frame: 23
      {0.01745f, 0.00000f, 0.00000f, 0.99985f},   // frame: 24
      {0.01396f, 0.00000f, 0.00000f, 0.99990f},   // frame: 25
      {0.01047f, 0.00000f, 0.00000f, 0.99995f},   // frame: 26
      {0.00698f, 0.00000f, 0.00000f, 0.99998f},   // frame: 27
      {0.00349f, 0.00000f, 0.00000f, 0.99999f},   // frame: 28
      {0.00000f, 0.00000f, 0.00000f, 1.00000f},   // frame: 29
      {0.00000f, 0.00000f, 0.00000f, 1.00000f},   // frame: 30
      {-0.00349f, 0.00000f, 0.00000f, 0.99999f},  // frame: 31
      {-0.00698f, 0.00000f, 0.00000f, 0.99998f},  // frame: 32
      {-0.01047f, 0.00000f, 0.00000f, 0.99995f},  // frame: 33
      {-0.01396f, 0.00000f, 0.00000f, 0.99990f},  // frame: 34
      {-0.01745f, 0.00000f, 0.00000f, 0.99985f},  // frame: 35
      {-0.02094f, 0.00000f, 0.00000f, 0.99978f},  // frame: 36
      {-0.02443f, 0.00000f, 0.00000f, 0.99970f},  // frame: 37
      {-0.02792f, 0.00000f, 0.00000f, 0.99961f},  // frame: 38
      {-0.03141f, 0.00000f, 0.00000f, 0.99951f},  // frame: 39
      {-0.03490f, 0.00000f, 0.00000f, 0.99939f},  // frame: 40
      {-0.03839f, 0.00000f, 0.00000f, 0.99926f},  // frame: 41
      {-0.04188f, 0.00000f, 0.00000f, 0.99912f},  // frame: 42
      {-0.04536f, 0.00000f, 0.00000f, 0.99897f},  // frame: 43
      {-0.04885f, 0.00000f, 0.00000f, 0.99881f},  // frame: 44
      {-0.04885f, 0.00000f, 0.00000f, 0.99881f},  // frame: 45
      {-0.04536f, 0.00000f, 0.00000f, 0.99897f},  // frame: 46
      {-0.04188f, 0.00000f, 0.00000f, 0.99912f},  // frame: 47
      {-0.03839f, 0.00000f, 0.00000f, 0.99926f},  // frame: 48
      {-0.03490f, 0.00000f, 0.00000f, 0.99939f},  // frame: 49
      {-0.03141f, 0.00000f, 0.00000f, 0.99951f},  // frame: 50
      {-0.02792f, 0.00000f, 0.00000f, 0.99961f},  // frame: 51
      {-0.02443f, 0.00000f, 0.00000f, 0.99970f},  // frame: 52
      {-0.02094f, 0.00000f, 0.00000f, 0.99978f},  // frame: 53
      {-0.01745f, 0.00000f, 0.00000f, 0.99985f},  // frame: 54
      {-0.01396f, 0.00000f, 0.00000f, 0.99990f},  // frame: 55
      {-0.01047f, 0.00000f, 0.00000f, 0.99995f},  // frame: 56
      {-0.00698f, 0.00000f, 0.00000f, 0.99998f},  // frame: 57
      {-0.00349f, 0.00000f, 0.00000f, 0.99999f},  // frame: 58
      {0.00000f, 0.00000f, 0.00000f, 1.00000f},   // frame: 59
      {0.00000f, 0.00000f, 0.00000f, 1.00000f},   // frame: 60
      {0.00000f, 0.00465f, 0.00000f, 0.99999f},   // frame: 61
      {0.00000f, 0.00931f, 0.00000f, 0.99996f},   // frame: 62
      {0.00000f, 0.01396f, 0.00000f, 0.99990f},   // frame: 63
      {0.00000f, 0.01862f, 0.00000f, 0.99983f},   // frame: 64
      {0.00000f, 0.02327f, 0.00000f, 0.99973f},   // frame: 65
      {0.00000f, 0.02792f, 0.00000f, 0.99961f},   // frame: 66
      {0.00000f, 0.03257f, 0.00000f, 0.99947f},   // frame: 67
      {0.00000f, 0.03723f, 0.00000f, 0.99931f},   // frame: 68
      {0.00000f, 0.04188f, 0.00000f, 0.99912f},   // frame: 69
      {0.00000f, 0.04653f, 0.00000f, 0.99892f},   // frame: 70
      {0.00000f, 0.05117f, 0.00000f, 0.99869f},   // frame: 71
      {0.00000f, 0.05582f, 0.00000f, 0.99844f},   // frame: 72
      {0.00000f, 0.06047f, 0.00000f, 0.99817f},   // frame: 73
      {0.00000f, 0.06511f, 0.00000f, 0.99788f},   // frame: 74
      {0.00000f, 0.06511f, 0.00000f, 0.99788f},   // frame: 75
      {0.00000f, 0.06047f, 0.00000f, 0.99817f},   // frame: 76
      {0.00000f, 0.05582f, 0.00000f, 0.99844f},   // frame: 77
      {0.00000f, 0.05117f, 0.00000f, 0.99869f},   // frame: 78
      {0.00000f, 0.04653f, 0.00000f, 0.99892f},   // frame: 79
      {0.00000f, 0.04188f, 0.00000f, 0.99912f},   // frame: 80
      {0.00000f, 0.03723f, 0.00000f, 0.99931f},   // frame: 81
      {0.00000f, 0.03257f, 0.00000f, 0.99947f},   // frame: 82
      {0.00000f, 0.02792f, 0.00000f, 0.99961f},   // frame: 83
      {0.00000f, 0.02327f, 0.00000f, 0.99973f},   // frame: 84
      {0.00000f, 0.01862f, 0.00000f, 0.99983f},   // frame: 85
      {0.00000f, 0.01396f, 0.00000f, 0.99990f},   // frame: 86
      {0.00000f, 0.00931f, 0.00000f, 0.99996f},   // frame: 87
      {0.00000f, 0.00465f, 0.00000f, 0.99999f},   // frame: 88
      {0.00000f, 0.00000f, 0.00000f, 1.00000f},   // frame: 89
      {0.00000f, 0.00000f, 0.00000f, 1.00000f},   // frame: 90
      {0.00000f, -0.00465f, 0.00000f, 0.99999f},  // frame: 91
      {0.00000f, -0.00931f, 0.00000f, 0.99996f},  // frame: 92
      {0.00000f, -0.01396f, 0.00000f, 0.99990f},  // frame: 93
      {0.00000f, -0.01862f, 0.00000f, 0.99983f},  // frame: 94
      {0.00000f, -0.02327f, 0.00000f, 0.99973f},  // frame: 95
      {0.00000f, -0.02792f, 0.00000f, 0.99961f},  // frame: 96
      {0.00000f, -0.03257f, 0.00000f, 0.99947f},  // frame: 97
      {0.00000f, -0.03723f, 0.00000f, 0.99931f},  // frame: 98
      {0.00000f, -0.04188f, 0.00000f, 0.99912f},  // frame: 99
      {0.00000f, -0.04653f, 0.00000f, 0.99892f},  // frame: 100
      {0.00000f, -0.05117f, 0.00000f, 0.99869f},  // frame: 101
      {0.00000f, -0.05582f, 0.00000f, 0.99844f},  // frame: 102
      {0.00000f, -0.06047f, 0.00000f, 0.99817f},  // frame: 103
      {0.00000f, -0.06511f, 0.00000f, 0.99788f},  // frame: 104
      {0.00000f, -0.06511f, 0.00000f, 0.99788f},  // frame: 105
      {0.00000f, -0.06047f, 0.00000f, 0.99817f},  // frame: 106
      {0.00000f, -0.05582f, 0.00000f, 0.99844f},  // frame: 107
      {0.00000f, -0.05117f, 0.00000f, 0.99869f},  // frame: 108
      {0.00000f, -0.04653f, 0.00000f, 0.99892f},  // frame: 109
      {0.00000f, -0.04188f, 0.00000f, 0.99912f},  // frame: 110
      {0.00000f, -0.03723f, 0.00000f, 0.99931f},  // frame: 111
      {0.00000f, -0.03257f, 0.00000f, 0.99947f},  // frame: 112
      {0.00000f, -0.02792f, 0.00000f, 0.99961f},  // frame: 113
      {0.00000f, -0.02327f, 0.00000f, 0.99973f},  // frame: 114
      {0.00000f, -0.01862f, 0.00000f, 0.99983f},  // frame: 115
      {0.00000f, -0.01396f, 0.00000f, 0.99990f},  // frame: 116
      {0.00000f, -0.00931f, 0.00000f, 0.99996f},  // frame: 117
      {0.00000f, -0.00465f, 0.00000f, 0.99999f},  // frame: 118
      {0.00000f, 0.00000f, 0.00000f, 1.00000f},   // frame: 119
      {0.00000f, 0.00000f, 0.00000f, 1.00000f},   // frame: 120
      {0.00000f, 0.00000f, 0.00291f, 1.00000f},   // frame: 121
      {0.00000f, 0.00000f, 0.00582f, 0.99998f},   // frame: 122
      {0.00000f, 0.00000f, 0.00873f, 0.99996f},   // frame: 123
      {0.00000f, 0.00000f, 0.01164f, 0.99993f},   // frame: 124
      {0.00000f, 0.00000f, 0.01454f, 0.99989f},   // frame: 125
      {0.00000f, 0.00000f, 0.01745f, 0.99985f},   // frame: 126
      {0.00000f, 0.00000f, 0.02036f, 0.99979f},   // frame: 127
      {0.00000f, 0.00000f, 0.02327f, 0.99973f},   // frame: 128
      {0.00000f, 0.00000f, 0.02618f, 0.99966f},   // frame: 129
      {0.00000f, 0.00000f, 0.02908f, 0.99958f},   // frame: 130
      {0.00000f, 0.00000f, 0.03199f, 0.99949f},   // frame: 131
      {0.00000f, 0.00000f, 0.03490f, 0.99939f},   // frame: 132
      {0.00000f, 0.00000f, 0.03781f, 0.99929f},   // frame: 133
      {0.00000f, 0.00000f, 0.04071f, 0.99917f},   // frame: 134
      {0.00000f, 0.00000f, 0.04071f, 0.99917f},   // frame: 135
      {0.00000f, 0.00000f, 0.03781f, 0.99929f},   // frame: 136
      {0.00000f, 0.00000f, 0.03490f, 0.99939f},   // frame: 137
      {0.00000f, 0.00000f, 0.03199f, 0.99949f},   // frame: 138
      {0.00000f, 0.00000f, 0.02908f, 0.99958f},   // frame: 139
      {0.00000f, 0.00000f, 0.02618f, 0.99966f},   // frame: 140
      {0.00000f, 0.00000f, 0.02327f, 0.99973f},   // frame: 141
      {0.00000f, 0.00000f, 0.02036f, 0.99979f},   // frame: 142
      {0.00000f, 0.00000f, 0.01745f, 0.99985f},   // frame: 143
      {0.00000f, 0.00000f, 0.01454f, 0.99989f},   // frame: 144
      {0.00000f, 0.00000f, 0.01164f, 0.99993f},   // frame: 145
      {0.00000f, 0.00000f, 0.00873f, 0.99996f},   // frame: 146
      {0.00000f, 0.00000f, 0.00582f, 0.99998f},   // frame: 147
      {0.00000f, 0.00000f, 0.00291f, 1.00000f},   // frame: 148
      {0.00000f, 0.00000f, 0.00000f, 1.00000f},   // frame: 149
      {0.00000f, 0.00000f, 0.00000f, 1.00000f},   // frame: 150
      {0.00000f, 0.00000f, -0.00291f, 1.00000f},  // frame: 151
      {0.00000f, 0.00000f, -0.00582f, 0.99998f},  // frame: 152
      {0.00000f, 0.00000f, -0.00873f, 0.99996f},  // frame: 153
      {0.00000f, 0.00000f, -0.01164f, 0.99993f},  // frame: 154
      {0.00000f, 0.00000f, -0.01454f, 0.99989f},  // frame: 155
      {0.00000f, 0.00000f, -0.01745f, 0.99985f},  // frame: 156
      {0.00000f, 0.00000f, -0.02036f, 0.99979f},  // frame: 157
      {0.00000f, 0.00000f, -0.02327f, 0.99973f},  // frame: 158
      {0.00000f, 0.00000f, -0.02618f, 0.99966f},  // frame: 159
      {0.00000f, 0.00000f, -0.02908f, 0.99958f},  // frame: 160
      {0.00000f, 0.00000f, -0.03199f, 0.99949f},  // frame: 161
      {0.00000f, 0.00000f, -0.03490f, 0.99939f},  // frame: 162
      {0.00000f, 0.00000f, -0.03781f, 0.99929f},  // frame: 163
      {0.00000f, 0.00000f, -0.04071f, 0.99917f},  // frame: 164
      {0.00000f, 0.00000f, -0.04071f, 0.99917f},  // frame: 165
      {0.00000f, 0.00000f, -0.03781f, 0.99929f},  // frame: 166
      {0.00000f, 0.00000f, -0.03490f, 0.99939f},  // frame: 167
      {0.00000f, 0.00000f, -0.03199f, 0.99949f},  // frame: 168
      {0.00000f, 0.00000f, -0.02908f, 0.99958f},  // frame: 169
      {0.00000f, 0.00000f, -0.02618f, 0.99966f},  // frame: 170
      {0.00000f, 0.00000f, -0.02327f, 0.99973f},  // frame: 171
      {0.00000f, 0.00000f, -0.02036f, 0.99979f},  // frame: 172
      {0.00000f, 0.00000f, -0.01745f, 0.99985f},  // frame: 173
      {0.00000f, 0.00000f, -0.01454f, 0.99989f},  // frame: 174
      {0.00000f, 0.00000f, -0.01164f, 0.99993f},  // frame: 175
      {0.00000f, 0.00000f, -0.00873f, 0.99996f},  // frame: 176
      {0.00000f, 0.00000f, -0.00582f, 0.99998f},  // frame: 177
      {0.00000f, 0.00000f, -0.00291f, 1.00000f},  // frame: 178
      {0.00000f, 0.00000f, 0.00000f, 1.00000f},   // frame: 179
  };
  // This translation aniamtion is an example of how to use head pose mode3
  //   0 -  59 frame   -> Tx [-0.05 - +0.05]
  //  60 - 119 frame   -> Ty [-0.05 - +0.05]
  // 120 - 179 frame   -> Sz [ 0.97 -  1.03]
  m_head_translation_animation = {
      // Translation [Tx, Ty, Sz]
      {0.000f, 0.000f, 1.000f},   // frame: 0
      {0.003f, 0.000f, 1.000f},   // frame: 1
      {0.007f, 0.000f, 1.000f},   // frame: 2
      {0.010f, 0.000f, 1.000f},   // frame: 3
      {0.013f, 0.000f, 1.000f},   // frame: 4
      {0.017f, 0.000f, 1.000f},   // frame: 5
      {0.020f, 0.000f, 1.000f},   // frame: 6
      {0.023f, 0.000f, 1.000f},   // frame: 7
      {0.027f, 0.000f, 1.000f},   // frame: 8
      {0.030f, 0.000f, 1.000f},   // frame: 9
      {0.033f, 0.000f, 1.000f},   // frame: 10
      {0.037f, 0.000f, 1.000f},   // frame: 11
      {0.040f, 0.000f, 1.000f},   // frame: 12
      {0.043f, 0.000f, 1.000f},   // frame: 13
      {0.047f, 0.000f, 1.000f},   // frame: 14
      {0.047f, 0.000f, 1.000f},   // frame: 15
      {0.043f, 0.000f, 1.000f},   // frame: 16
      {0.040f, 0.000f, 1.000f},   // frame: 17
      {0.037f, 0.000f, 1.000f},   // frame: 18
      {0.033f, 0.000f, 1.000f},   // frame: 19
      {0.030f, 0.000f, 1.000f},   // frame: 20
      {0.027f, 0.000f, 1.000f},   // frame: 21
      {0.023f, 0.000f, 1.000f},   // frame: 22
      {0.020f, 0.000f, 1.000f},   // frame: 23
      {0.017f, 0.000f, 1.000f},   // frame: 24
      {0.013f, 0.000f, 1.000f},   // frame: 25
      {0.010f, 0.000f, 1.000f},   // frame: 26
      {0.007f, 0.000f, 1.000f},   // frame: 27
      {0.003f, 0.000f, 1.000f},   // frame: 28
      {0.000f, 0.000f, 1.000f},   // frame: 29
      {0.000f, 0.000f, 1.000f},   // frame: 30
      {-0.003f, 0.000f, 1.000f},  // frame: 31
      {-0.007f, 0.000f, 1.000f},  // frame: 32
      {-0.010f, 0.000f, 1.000f},  // frame: 33
      {-0.013f, 0.000f, 1.000f},  // frame: 34
      {-0.017f, 0.000f, 1.000f},  // frame: 35
      {-0.020f, 0.000f, 1.000f},  // frame: 36
      {-0.023f, 0.000f, 1.000f},  // frame: 37
      {-0.027f, 0.000f, 1.000f},  // frame: 38
      {-0.030f, 0.000f, 1.000f},  // frame: 39
      {-0.033f, 0.000f, 1.000f},  // frame: 40
      {-0.037f, 0.000f, 1.000f},  // frame: 41
      {-0.040f, 0.000f, 1.000f},  // frame: 42
      {-0.043f, 0.000f, 1.000f},  // frame: 43
      {-0.047f, 0.000f, 1.000f},  // frame: 44
      {-0.047f, 0.000f, 1.000f},  // frame: 45
      {-0.043f, 0.000f, 1.000f},  // frame: 46
      {-0.040f, 0.000f, 1.000f},  // frame: 47
      {-0.037f, 0.000f, 1.000f},  // frame: 48
      {-0.033f, 0.000f, 1.000f},  // frame: 49
      {-0.030f, 0.000f, 1.000f},  // frame: 50
      {-0.027f, 0.000f, 1.000f},  // frame: 51
      {-0.023f, 0.000f, 1.000f},  // frame: 52
      {-0.020f, 0.000f, 1.000f},  // frame: 53
      {-0.017f, 0.000f, 1.000f},  // frame: 54
      {-0.013f, 0.000f, 1.000f},  // frame: 55
      {-0.010f, 0.000f, 1.000f},  // frame: 56
      {-0.007f, 0.000f, 1.000f},  // frame: 57
      {-0.003f, 0.000f, 1.000f},  // frame: 58
      {0.000f, 0.000f, 1.000f},   // frame: 59
      {0.000f, 0.000f, 1.000f},   // frame: 60
      {0.000f, 0.003f, 1.000f},   // frame: 61
      {0.000f, 0.007f, 1.000f},   // frame: 62
      {0.000f, 0.010f, 1.000f},   // frame: 63
      {0.000f, 0.013f, 1.000f},   // frame: 64
      {0.000f, 0.017f, 1.000f},   // frame: 65
      {0.000f, 0.020f, 1.000f},   // frame: 66
      {0.000f, 0.023f, 1.000f},   // frame: 67
      {0.000f, 0.027f, 1.000f},   // frame: 68
      {0.000f, 0.030f, 1.000f},   // frame: 69
      {0.000f, 0.033f, 1.000f},   // frame: 70
      {0.000f, 0.037f, 1.000f},   // frame: 71
      {0.000f, 0.040f, 1.000f},   // frame: 72
      {0.000f, 0.043f, 1.000f},   // frame: 73
      {0.000f, 0.047f, 1.000f},   // frame: 74
      {0.000f, 0.047f, 1.000f},   // frame: 75
      {0.000f, 0.043f, 1.000f},   // frame: 76
      {0.000f, 0.040f, 1.000f},   // frame: 77
      {0.000f, 0.037f, 1.000f},   // frame: 78
      {0.000f, 0.033f, 1.000f},   // frame: 79
      {0.000f, 0.030f, 1.000f},   // frame: 80
      {0.000f, 0.027f, 1.000f},   // frame: 81
      {0.000f, 0.023f, 1.000f},   // frame: 82
      {0.000f, 0.020f, 1.000f},   // frame: 83
      {0.000f, 0.017f, 1.000f},   // frame: 84
      {0.000f, 0.013f, 1.000f},   // frame: 85
      {0.000f, 0.010f, 1.000f},   // frame: 86
      {0.000f, 0.007f, 1.000f},   // frame: 87
      {0.000f, 0.003f, 1.000f},   // frame: 88
      {0.000f, 0.000f, 1.000f},   // frame: 89
      {0.000f, 0.000f, 1.000f},   // frame: 90
      {0.000f, -0.003f, 1.000f},  // frame: 91
      {0.000f, -0.007f, 1.000f},  // frame: 92
      {0.000f, -0.010f, 1.000f},  // frame: 93
      {0.000f, -0.013f, 1.000f},  // frame: 94
      {0.000f, -0.017f, 1.000f},  // frame: 95
      {0.000f, -0.020f, 1.000f},  // frame: 96
      {0.000f, -0.023f, 1.000f},  // frame: 97
      {0.000f, -0.027f, 1.000f},  // frame: 98
      {0.000f, -0.030f, 1.000f},  // frame: 99
      {0.000f, -0.033f, 1.000f},  // frame: 100
      {0.000f, -0.037f, 1.000f},  // frame: 101
      {0.000f, -0.040f, 1.000f},  // frame: 102
      {0.000f, -0.043f, 1.000f},  // frame: 103
      {0.000f, -0.047f, 1.000f},  // frame: 104
      {0.000f, -0.047f, 1.000f},  // frame: 105
      {0.000f, -0.043f, 1.000f},  // frame: 106
      {0.000f, -0.040f, 1.000f},  // frame: 107
      {0.000f, -0.037f, 1.000f},  // frame: 108
      {0.000f, -0.033f, 1.000f},  // frame: 109
      {0.000f, -0.030f, 1.000f},  // frame: 110
      {0.000f, -0.027f, 1.000f},  // frame: 111
      {0.000f, -0.023f, 1.000f},  // frame: 112
      {0.000f, -0.020f, 1.000f},  // frame: 113
      {0.000f, -0.017f, 1.000f},  // frame: 114
      {0.000f, -0.013f, 1.000f},  // frame: 115
      {0.000f, -0.010f, 1.000f},  // frame: 116
      {0.000f, -0.007f, 1.000f},  // frame: 117
      {0.000f, -0.003f, 1.000f},  // frame: 118
      {0.000f, 0.000f, 1.000f},   // frame: 119
      {0.000f, 0.000f, 1.000f},   // frame: 120
      {0.000f, 0.000f, 0.998f},   // frame: 121
      {0.000f, 0.000f, 0.996f},   // frame: 122
      {0.000f, 0.000f, 0.994f},   // frame: 123
      {0.000f, 0.000f, 0.992f},   // frame: 124
      {0.000f, 0.000f, 0.990f},   // frame: 125
      {0.000f, 0.000f, 0.988f},   // frame: 126
      {0.000f, 0.000f, 0.986f},   // frame: 127
      {0.000f, 0.000f, 0.984f},   // frame: 128
      {0.000f, 0.000f, 0.982f},   // frame: 129
      {0.000f, 0.000f, 0.980f},   // frame: 130
      {0.000f, 0.000f, 0.978f},   // frame: 131
      {0.000f, 0.000f, 0.976f},   // frame: 132
      {0.000f, 0.000f, 0.974f},   // frame: 133
      {0.000f, 0.000f, 0.972f},   // frame: 134
      {0.000f, 0.000f, 0.972f},   // frame: 135
      {0.000f, 0.000f, 0.974f},   // frame: 136
      {0.000f, 0.000f, 0.976f},   // frame: 137
      {0.000f, 0.000f, 0.978f},   // frame: 138
      {0.000f, 0.000f, 0.980f},   // frame: 139
      {0.000f, 0.000f, 0.982f},   // frame: 140
      {0.000f, 0.000f, 0.984f},   // frame: 141
      {0.000f, 0.000f, 0.986f},   // frame: 142
      {0.000f, 0.000f, 0.988f},   // frame: 143
      {0.000f, 0.000f, 0.990f},   // frame: 144
      {0.000f, 0.000f, 0.992f},   // frame: 145
      {0.000f, 0.000f, 0.994f},   // frame: 146
      {0.000f, 0.000f, 0.996f},   // frame: 147
      {0.000f, 0.000f, 0.998f},   // frame: 148
      {0.000f, 0.000f, 1.000f},   // frame: 149
      {0.000f, 0.000f, 1.000f},   // frame: 150
      {0.000f, 0.000f, 1.002f},   // frame: 151
      {0.000f, 0.000f, 1.004f},   // frame: 152
      {0.000f, 0.000f, 1.006f},   // frame: 153
      {0.000f, 0.000f, 1.008f},   // frame: 154
      {0.000f, 0.000f, 1.010f},   // frame: 155
      {0.000f, 0.000f, 1.012f},   // frame: 156
      {0.000f, 0.000f, 1.014f},   // frame: 157
      {0.000f, 0.000f, 1.016f},   // frame: 158
      {0.000f, 0.000f, 1.018f},   // frame: 159
      {0.000f, 0.000f, 1.020f},   // frame: 160
      {0.000f, 0.000f, 1.022f},   // frame: 161
      {0.000f, 0.000f, 1.024f},   // frame: 162
      {0.000f, 0.000f, 1.026f},   // frame: 163
      {0.000f, 0.000f, 1.028f},   // frame: 164
      {0.000f, 0.000f, 1.028f},   // frame: 165
      {0.000f, 0.000f, 1.026f},   // frame: 166
      {0.000f, 0.000f, 1.024f},   // frame: 167
      {0.000f, 0.000f, 1.022f},   // frame: 168
      {0.000f, 0.000f, 1.020f},   // frame: 169
      {0.000f, 0.000f, 1.018f},   // frame: 170
      {0.000f, 0.000f, 1.016f},   // frame: 171
      {0.000f, 0.000f, 1.014f},   // frame: 172
      {0.000f, 0.000f, 1.012f},   // frame: 173
      {0.000f, 0.000f, 1.010f},   // frame: 174
      {0.000f, 0.000f, 1.008f},   // frame: 175
      {0.000f, 0.000f, 1.006f},   // frame: 176
      {0.000f, 0.000f, 1.004f},   // frame: 177
      {0.000f, 0.000f, 1.002f},   // frame: 178
      {0.000f, 0.000f, 1.000f},   // frame: 179
  };
}

void App::GetFPS() {
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

void App::DrawFPS(cv::Mat& img) {
  GetFPS();
  if (m_frameTime && m_showFPS) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%.1f", 1. / m_frameTime);
    cv::putText(img, buf, cv::Point(img.cols - 80, img.rows - 10), cv::FONT_HERSHEY_SIMPLEX, 1,
                cv::Scalar(255, 255, 255), 1);
  }
}

const char* App::ErrorStringFromCode(App::Err code) {
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
      {errAudio, "no audio source has been found"},
      {errHeadPose, "an error has occured while set the head pose"},
      {errImageSize, "the image size cannot be accommodated"},
      {errNotFound, "the item cannot be found"},
      {errNoFace, "no face has been found"},
      {errSDK, "an SDK error has occurred"},
      {errCuda, "a CUDA error has occurred"},
      {errCancel, "the user cancelled"},
      {errAudioFile, "unable to open driving audio file"},
      {errSourceFile, "unable to open source image file"},
      {errMode, "unsupported mode or wrong source image size in that mode"},
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
  App app;
  App::Err app_err = App::Err::errNone;

  NvCV_Status err = NvAR_ConfigureLogger(FLAG_logLevel, FLAG_log.c_str(), nullptr, nullptr);
  if (NVCV_SUCCESS != err)
    printf("%s: while configuring logger to \"%s\"\n", NvCV_GetErrorStringFromCode(err), FLAG_log.c_str());

  if (FLAG_modelPath.empty()) {
    printf(
        "WARNING: Model path not specified. Please set --model_path=/path/to/trt/and/face/models, "
        "SDK will attempt to load the models from NVAR_MODEL_DIR environment variable, "
        "please restart your application after the SDK Installation. \n");
  }
  app_err = app.CreateEffect(FLAG_modelPath);
  BAIL_IF_ERR(app_err);

  if (FLAG_inSrc.empty()) {
    app_err = App::errMissing;
    printf("ERROR: %s, please specify your source portrait file using --in_src \n", app.ErrorStringFromCode(app_err));
    goto bail;
  }

  if (FLAG_offlineMode) {
    if (FLAG_inDrv.empty()) {
      app_err = App::errMissing;
      printf("ERROR: %s, please specify driving audio file using --in_drv in offline mode\n",
             app.ErrorStringFromCode(app_err));
      goto bail;
    }
    app_err = app.InitOfflineMode();
  } else {
    app_err = App::errMode;
    printf("ERROR: %s, Live capture mode not supported currently \n", app.ErrorStringFromCode(app_err));
    goto bail;
  }

  app_err = app.InitOutput(FLAG_outFile);
  BAIL_IF_ERR(app_err);

  app_err = app.Run();
  BAIL_IF_ERR(app_err);

  printf("Input audio file successfully processed.");
  if (FLAG_offlineMode && FLAG_captureOutputs && FLAG_verbose) {
    printf("Output video saved at %s", FLAG_outFile.c_str());
  }

bail:
  if (app_err) printf("ERROR: %s\n", app.ErrorStringFromCode(app_err));
  app.Stop();
  return (int)app_err;
}
