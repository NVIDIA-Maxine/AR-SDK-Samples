/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cassert>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>

#include "nvAR.h"
#include "nvARLipSync.h"
#include "nvAR_defs.h"
#include "nvCVOpenCV.h"
#include "opencv2/core/utils/filesystem.hpp"
#include "opencv2/opencv.hpp"
#include "waveReadWrite.h"

#if CV_MAJOR_VERSION >= 4
#define CV_CAP_PROP_FRAME_WIDTH cv::CAP_PROP_FRAME_WIDTH
#define CV_CAP_PROP_FRAME_HEIGHT cv::CAP_PROP_FRAME_HEIGHT
#define CV_CAP_PROP_FPS cv::CAP_PROP_FPS
#define CV_CAP_PROP_FRAME_COUNT cv::CAP_PROP_FRAME_COUNT
#define CV_WINDOW_AUTOSIZE cv::WINDOW_AUTOSIZE
#endif

#ifdef _MSC_VER
#define strcasecmp _stricmp
#endif /* _MSC_VER */

/********************************************************************************
 * Command-line arguments
 ********************************************************************************/
static const char* kExtendOff = "off";
static const char* kExtendVideoForward = "forward_loop";
static const char* kExtendVideoReverse = "reverse_loop";
static const char* kExtendAudioSilence = "silence";

// clang-format off
bool          FLAG_debug = false,
              FLAG_verbose = false,
              FLAG_offlineMode = true,                 // reads audio and video from files if set to true; webcam mode if set to false; currently only offline mode supported
              FLAG_captureOutputs = true,              // write generated video to file if set to true. only in offline mode
              FLAG_enableLookAway = false,
              FLAG_roiSkipFaceDetect = false;
unsigned      FLAG_headMovementSpeed = 0;              // set to default value for Head Movement Speed (SLOW)
unsigned      FLAG_logLevel = NVCV_LOG_ERROR;
double        FLAG_bypassFactor = 0.0f;
std::string   FLAG_inVid,
              FLAG_inAudio,
              FLAG_outFile,
              FLAG_modelPath,
              FLAG_extendVideo = kExtendOff,
              FLAG_extendAudio = kExtendOff,
              FLAG_captureCodec = "avc1",
              FLAG_inBgImg,
              FLAG_log = "stderr",
              FLAG_roiRect;
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
#define RETURN_APPERR_IF_NVERR(nv_err, app_err)         \
  do {                                                  \
    if (!CheckResult(nv_err, __LINE__)) return app_err; \
  } while (0)
/********************************************************************************
 * Usage
 ********************************************************************************/

static void Usage() {
  // todo: clear the misuse of square and curly braces
  printf(
      "LipSyncApp [<args> ...]\n"
      "where <args> are\n"
      " --verbose[=(true|false)]              report interesting info\n"
      " --debug[=(true|false)]                report debugging info\n"
      " --log=<file>                          log SDK errors to a file, \"stderr\" or \"\" (default stderr)\n"
      " --log_level=<N>                       the desired log level: {0, 1, 2, 3} = {FATAL, ERROR, WARNING, INFO}, "
      "respectively (default 1)\n"
      " --model_path=<path>                   specify the directory containing the TRT models\n"
      " --capture_outputs[=(true|false)]      write generated video to file if set to true. only in offline mode\n"
      " --offline_mode[=(true|false)]         reads video from file if set to true; webcam mode if set to false. "
      "Default true. Webcam mode is not currently supported\n"
      " --codec=<fourcc>                      FOURCC code for the desired codec (default H264)\n"
      " --in_video=<file>                     specify the input video file\n"
      " --in_audio=<file>                     specify the input audio file.\n"
      " --roi_rect=<x,y,w,h>                  specify the region of interest rectangle as x,y,width,height (no space "
      "allowed after comma) \n"
      " --roi_skip_fd[=(true|false)]          specify true to skip face detection and use the ROI rectangle as the "
      "face bounding box "
      "(default is false (perform face detection on ROI))\n"
      " --bypass_factor=<[0.0,..1.0]>         specify the bypass factor, value in between 0.0 and 1.0 for partial "
      "bypass."
      "0.0 = effect fully enabled, 1.0 = effect fully bypassed (default 0.0)\n "
      " --out=<file>                          specify the output file. only in offline mode and capture_outputs is "
      "true.\n"
      " --extend_short_video=<str>            desired behavior when the input video is shorter than the input audio "
      "(default off):\n"
      "                                         off - truncate the output when the input video ends\n"
      "                                         forward_loop - extend the video by restarting it from the "
      "beginning\n"
      "                                         reverse_loop - extend the video by reversing it and playing frames "
      "backwards from the end. Warning: This may increase execution time compared to forward_loop.\n"
      " --extend_short_audio=<str>            desired behavior when the input audio is shorter than the input video "
      "(default off):\n"
      "                                         off - truncate the output when the input audio ends\n"
      "                                         silence - extend the audio by adding silence\n"
      " --head_movement_speed=<N>               specify the expected speed of head motion in the input video: 0=SLOW, "
      "1=FAST. Default: 0 (SLOW).\n");
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
    } else if ((arg[1] == '-') &&                                               //
               (GetFlagArgVal("verbose", arg, &FLAG_verbose) ||                 //
                GetFlagArgVal("debug", arg, &FLAG_debug) ||                     //
                GetFlagArgVal("log", arg, &FLAG_log) ||                         //
                GetFlagArgVal("log_level", arg, &FLAG_logLevel) ||              //
                GetFlagArgVal("in_video", arg, &FLAG_inVid) ||                  //
                GetFlagArgVal("in_audio", arg, &FLAG_inAudio) ||                //
                GetFlagArgVal("out", arg, &FLAG_outFile) ||                     //
                GetFlagArgVal("extend_short_video", arg, &FLAG_extendVideo) ||  //
                GetFlagArgVal("extend_short_audio", arg, &FLAG_extendAudio) ||  //
                GetFlagArgVal("head_movement_speed", arg, &FLAG_headMovementSpeed) ||
                GetFlagArgVal("codec", arg, &FLAG_captureCodec) ||              //
                GetFlagArgVal("bypass_factor", arg, &FLAG_bypassFactor) ||      //
                GetFlagArgVal("roi_skip_fd", arg, &FLAG_roiSkipFaceDetect) ||   //
                GetFlagArgVal("roi_rect", arg, &FLAG_roiRect) ||                //
                GetFlagArgVal("out_file", arg, &FLAG_outFile) ||                //
                GetFlagArgVal("capture_outputs", arg, &FLAG_captureOutputs) ||  //
                GetFlagArgVal("offline_mode", arg, &FLAG_offlineMode) ||        //
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

constexpr float kMaxFPS = 60.0f;
constexpr float kFPSTolerance = 1.0f;  // Allow small floating-point errors

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
    errMode,
    errMissing,
    errAudio,
    errImageSize,
    errNotFound,
    errNoFace,
    errSDK,
    errCuda,
    errCancel,
    errAudioFile,
    errSourceFile,
    errSmallVideo
  };

  App();
  ~App();
  Err CreateEffect();
  Err InitOfflineMode();
  Err InitOutput();
  Err ProcessOutputVideo();
  Err Run();
  Err Stop();

  static const char* ErrorStringFromCode(Err code);

 private:
  MyTimer m_frameTimer;
  double m_frameTime;

  bool m_showFPS;
  float m_fps;
  NvAR_FeatureHandle m_lipSyncHandle{};
  CUstream m_stream{};
  cv::VideoCapture m_cap{};
  NvCVImage* m_srcImgGpu;        // source image
  NvCVImage m_tmp;               // tmp image
  NvCVImage m_cDst, m_gDst;      // output image
  cv::VideoWriter m_genVideo{};  // output video file
  unsigned m_srcWidth, m_srcHeight;
  unsigned m_frameCount;

  // Output values for debugging.
  float m_lipSyncActivation = 0.0f;

  std::string m_outParentPath, m_outFilename;  // Decomposition of FLAG_outFile.
};

App::App() {
  // Make sure things are initialized properly
  m_frameTime = -1;
  m_showFPS = false;
  m_srcImgGpu = nullptr;
  m_frameCount = 0;
}

App::~App() {
  if (m_stream) {
    NvAR_CudaStreamDestroy(m_stream);
  }
  NvAR_Destroy(m_lipSyncHandle);
}

App::Err App::CreateEffect() {
  NvCV_Status err = NVCV_SUCCESS;

  // load trt plugins
  err = NvAR_Create(NvAR_Feature_LipSync, &m_lipSyncHandle);
  if (err) {
    std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
    return Err::errSDK;
  }

  err = NvAR_CudaStreamCreate(&m_stream);
  if (err) {
    std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
    return Err::errSDK;
  }

  err = NvAR_SetCudaStream(m_lipSyncHandle, NvAR_Parameter_Config(CUDAStream), m_stream);
  if (err) {
    std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
    return Err::errSDK;
  }

  err = NvAR_SetString(m_lipSyncHandle, NvAR_Parameter_Config(ModelDir), FLAG_modelPath.c_str());
  if (err) {
    std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
    return Err::errSDK;
  }

  err = NvAR_SetF32(m_lipSyncHandle, NvAR_Parameter_Config(VideoFPS), m_fps);
  if (err) {
    std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
    return Err::errSDK;
  }

  err = NvAR_Load(m_lipSyncHandle);
  if (err) {
    std::cout << NvCV_GetErrorStringFromCode(err) << std::endl;
    return Err::errSDK;
  }

  return Err::errNone;
}

App::Err App::InitOfflineMode() {
  if (!m_cap.open(FLAG_inVid)) {
    printf("ERROR: Unable to open the source video file \"%s\" \n", FLAG_inVid.c_str());
    return Err::errSourceFile;
  }
  m_srcWidth = (unsigned)m_cap.get(CV_CAP_PROP_FRAME_WIDTH);
  m_srcHeight = (unsigned)m_cap.get(CV_CAP_PROP_FRAME_HEIGHT);
  m_fps = m_cap.get(CV_CAP_PROP_FPS);

  if (m_fps > kMaxFPS + kFPSTolerance) {
    printf("ERROR: Unsupported FPS %f in source video file \"%s\" \n", m_fps, FLAG_inVid.c_str());
    return Err::errSourceFile;
  }

  if (m_srcWidth > 4096 || m_srcHeight > 2160) {
    printf("ERROR: Unsupported high resolution (%u x %u) source video file \"%s\" \n", m_srcWidth, m_srcHeight,
           FLAG_inVid.c_str());
    return Err::errSourceFile;
  }

  if (m_srcHeight < 360) {
    printf("WARNING: Low resolution (%u x %u) source video file \"%s\" \n", m_srcWidth, m_srcHeight,
           FLAG_inVid.c_str());
  }

  return Err::errNone;
}

App::Err App::InitOutput() {
  cv::Mat img;
  cv::Size frame_size(m_srcWidth, m_srcHeight);

  if (FLAG_captureOutputs) {
    if (FLAG_outFile.empty()) {
      const size_t ext_index = FLAG_inAudio.find_last_of(".");
      const std::string prefix = FLAG_inAudio.substr(0, ext_index);
      FLAG_outFile = prefix + "_output.mp4";
    }

    // Decompose FLAG_outFile into parent path and filename.
    const size_t sep_index = FLAG_outFile.find_last_of("/\\");
    if (sep_index == std::string::npos) {
      m_outParentPath = ".";
      m_outFilename = FLAG_outFile;
    } else {
      m_outParentPath = FLAG_outFile.substr(0, sep_index);
      m_outFilename = FLAG_outFile.substr(sep_index + 1);
      // Create the output parent path, if necessary.
      if (!cv::utils::fs::createDirectories(m_outParentPath)) {
        printf("ERROR: Unable to create the output directory \"%s\" \n", m_outParentPath.c_str());
        return Err::errGeneral;
      }
    }

    if (FLAG_debug) {
      printf("fps of generated video is %f\n", m_fps);
    }
    if (!m_genVideo.open(FLAG_outFile, StringToFourcc(FLAG_captureCodec), (double)m_fps, frame_size)) {
      printf("ERROR: Unable to open the output video file \"%s\" \n", FLAG_outFile.c_str());
      return Err::errGeneral;
    }
  }
  return Err::errNone;
}

App::Err App::Run(void) {
  App::Err app_err = errNone;
  NvCV_Status err = NVCV_SUCCESS;
  NvCVImage c_src, g_src, tmp;

  NvAR_Rect roi = {0.0f, 0.0f, 0.0f, 0.0f};
  if (!FLAG_roiRect.empty()) {
    int n = sscanf(FLAG_roiRect.c_str(), "%f,%f,%f,%f", &roi.x, &roi.y, &roi.width, &roi.height);
    if (n == 4) {
      if (FLAG_debug) printf("Using ROI: %f, %f, %f, %f\n", roi.x, roi.y, roi.width, roi.height);
    } else {
      std::cerr << "Error: Invalid ROI format (expected x,y,w,h with no spaces between commas), but recieved: "
                << FLAG_roiRect << std::endl;
      return errParameter;
    }
  }
  if (!(FLAG_bypassFactor >= 0.0f && FLAG_bypassFactor <= 1.0f)) {
    std::cerr << "Error: Invalid bypass factor (expected value in between 0.0 to 1.0), but recieved: "
              << FLAG_bypassFactor << std::endl;
    return errParameter;
  }

  if (FLAG_debug) {
    printf("Size of the video frame: %dx%d\n", m_srcWidth, m_srcHeight);
  }

  // Allocate Source Image GPU
  err = NvCVImage_Alloc(&g_src, m_srcWidth, m_srcHeight, NVCV_BGR, NVCV_U8, NVCV_CHUNKY, NVCV_GPU, 1);

  // get input sample rate
  unsigned int input_sample_rate = 0;
  RETURN_APPERR_IF_NVERR(err = NvAR_GetU32(m_lipSyncHandle, NvAR_Parameter_Config(SampleRate), &input_sample_rate),
                         errSDK);

  // get input num channels
  unsigned int num_channels = 0;
  RETURN_APPERR_IF_NVERR(err = NvAR_GetU32(m_lipSyncHandle, NvAR_Parameter_Config(NumChannels), &num_channels), errSDK);

  // Number of initial input video frames to process before retrieving an output frame.
  uint32_t init_latency_frame_cnt = 0;
  RETURN_APPERR_IF_NVERR(
      err = NvAR_GetU32(m_lipSyncHandle, NvAR_Parameter_Config(NumInitialFrames), &init_latency_frame_cnt), errSDK);

  // Read audio file
  std::vector<float>* input_wav_samples;
  unsigned int input_num_samples = 0;
  // read the input audio file
  if (!ReadWavFile(FLAG_inAudio.c_str(), input_sample_rate, num_channels, &input_wav_samples, &input_num_samples,
                   nullptr, /* align_samples */ -1, FLAG_debug || FLAG_verbose)) {
    std::cerr << "Unable to read wav file: " << FLAG_inAudio << std::endl;
    return errAudioFile;
  }

  // Setup output images
  err = NvCVImage_Alloc(&m_cDst, m_srcWidth, m_srcHeight, NVCV_BGR, NVCV_U8, NVCV_CHUNKY, NVCV_CPU, 1);
  err = NvCVImage_Alloc(&m_gDst, m_srcWidth, m_srcHeight, NVCV_BGR, NVCV_U8, NVCV_CHUNKY, NVCV_GPU, 1);
  RETURN_APPERR_IF_NVERR(
      err = NvAR_SetObject(m_lipSyncHandle, NvAR_Parameter_Output(Image), &m_gDst, sizeof(NvCVImage)), errSDK);

  RETURN_APPERR_IF_NVERR(
      err = NvAR_SetF32Array(m_lipSyncHandle, NvAR_Parameter_Output(Activation), &m_lipSyncActivation, 1), errSDK);

  // Flag that will be set when we run the feature, to indicate whether an output image is ready.
  unsigned int output_ready = 0;
  RETURN_APPERR_IF_NVERR(err = NvAR_SetU32Array(m_lipSyncHandle, NvAR_Parameter_Output(Ready), &output_ready, 1),
                         errSDK);

  const unsigned int samples_per_second = input_sample_rate;  // 16000 Hz
  unsigned int last_audio_end_sample = 0;
  float estimated_video_frame_duration = 1.0f / m_fps;
  cv::Mat img;

  bool audio_finished = false;
  bool video_finished = false;
  int frame_step = 1;

  // The end_frame_index will be set when the inputs end.
  unsigned end_frame_index = std::numeric_limits<unsigned>::max();
  for (unsigned input_frame_index = 0; input_frame_index < end_frame_index; ++input_frame_index) {
    const int pos_frames = static_cast<int>(m_cap.get(cv::CAP_PROP_POS_FRAMES));
    // Check if we have more video to read.
    bool got_video_frame = m_cap.read(img);
    if (frame_step < 0) {
      if (pos_frames == 0) {
        // We are at frame 0, trying to step backwards.
        frame_step = 1;
      } else {
        if (!m_cap.set(cv::CAP_PROP_POS_FRAMES, pos_frames - 1)) {
          std::cerr << "Error: Unable to seek video" << std::endl;
          return errRead;
        }
      }
    }

    // Get timestamp from video frame in seconds
    double frame_timestamp = input_frame_index * estimated_video_frame_duration;

    // Calculate audio window based on video timestamp
    const unsigned int audio_start_sample = last_audio_end_sample;
    const unsigned int audio_end_sample =
        static_cast<unsigned int>((frame_timestamp + estimated_video_frame_duration) * samples_per_second);
    const unsigned audio_frame_length = audio_end_sample - audio_start_sample;
    // Store end sample for next frame.
    last_audio_end_sample = audio_end_sample;

    if (FLAG_debug) std::cerr << "Processing frame index: " << input_frame_index << std::endl;

    if (!video_finished && !got_video_frame) {
      // The first time we fail to read a frame, it's the end of the video.
      video_finished = true;
    }

    if (video_finished && !got_video_frame) {
      // If the video has finished and we didn't get a frame, we must be at the end of the video going forwards.
      // What to do next...
      if (FLAG_extendVideo == kExtendVideoForward) {
        if (FLAG_debug) std::cerr << "Looping video forwards from beginning" << std::endl;
        if (!m_cap.set(cv::CAP_PROP_POS_FRAMES, 0)) {
          std::cerr << "Error: Unable to seek video" << std::endl;
          return errRead;
        }
        got_video_frame = m_cap.read(img);
        if (!got_video_frame) {
          std::cerr << "Error: Failed to read video frame after looping" << std::endl;
          return errRead;
        }
      } else if (FLAG_extendVideo == kExtendVideoReverse) {
        if (FLAG_debug) std::cerr << "Looping video backwards from end" << std::endl;
        frame_step = -1;
        if (!m_cap.set(cv::CAP_PROP_POS_FRAMES, pos_frames - 1)) {
          std::cerr << "Error: Unable to seek video" << std::endl;
          return errRead;
        }
        got_video_frame = m_cap.read(img);
        if (!got_video_frame) {
          std::cerr << "Error: Failed to read video frame after looping" << std::endl;
          return errRead;
        }
        if (!m_cap.set(cv::CAP_PROP_POS_FRAMES, pos_frames - 1)) {
          std::cerr << "Error: Unable to seek video" << std::endl;
          return errRead;
        }
      }
    }

    // Create audio frame initialized to zero.
    // Generates silence if audio file has finished
    std::vector<float> audio_frame(audio_frame_length, 0.0f);
    // Copy valid audio into the frame.
    const size_t valid_audio_start_sample = std::min<size_t>(audio_start_sample, input_wav_samples->size());
    const size_t valid_audio_end_sample = std::min<size_t>(audio_end_sample, input_wav_samples->size());
    std::copy(input_wav_samples->begin() + valid_audio_start_sample,
              input_wav_samples->begin() + valid_audio_end_sample, audio_frame.begin());

    // Check if we have more audio to read.
    bool got_audio_frame = audio_start_sample < input_wav_samples->size();

    if (!audio_finished && !got_audio_frame) {
      // The first time we fail to read a frame, it's the end of the audio.
      audio_finished = true;
    }

    if (audio_finished && !got_audio_frame) {
      if (FLAG_extendAudio == kExtendAudioSilence) {
        got_audio_frame = true;
      }
    }

    // Check if we should stop processing.
    const bool should_stop = (video_finished && audio_finished) || !got_video_frame || !got_audio_frame;
    // Only update `end_frame_index` once, the first time `should_stop` is true.
    if (end_frame_index == std::numeric_limits<unsigned>::max() && should_stop) {
      if (video_finished && !audio_finished) {
        std::cerr << "Warning: video finished before audio. Audio may be truncated" << std::endl;
      }
      // We will process `init_latency_frame_cnt` more frames to flush the pipeline.
      end_frame_index = input_frame_index + init_latency_frame_cnt;
    }

    if (FLAG_bypassFactor != 0 || (roi.width > 0 && roi.height > 0)) {
      NvAR_SpeakerData speaker_data{};
      speaker_data.audio_frame_data = audio_frame.data();
      speaker_data.audio_frame_size = audio_frame.size();
      speaker_data.bypass = FLAG_bypassFactor;
      speaker_data.region_type = static_cast<int>(FLAG_roiSkipFaceDetect);
      speaker_data.region = roi;
      RETURN_APPERR_IF_NVERR(
          err = NvAR_SetObject(m_lipSyncHandle, NvAR_Parameter_Input(SpeakerData), &speaker_data, sizeof(speaker_data)),
          errSDK);
    } else {
      // Set Audio Frame
      RETURN_APPERR_IF_NVERR(err = NvAR_SetF32Array(m_lipSyncHandle, NvAR_Parameter_Input(AudioFrameBuffer),
                                                    audio_frame.data(), audio_frame.size()),
                             errSDK);
    }

    RETURN_APPERR_IF_NVERR(
        err = NvAR_SetU32(m_lipSyncHandle, NvAR_Parameter_Input(HeadMovementSpeed), FLAG_headMovementSpeed), errSDK);

    if (got_video_frame) {
      (void)NVWrapperForCVMat(&img, &c_src);
      RETURN_APPERR_IF_NVERR(err = NvCVImage_Transfer(&c_src, &g_src, 1, m_stream, &tmp), errSDK);
      RETURN_APPERR_IF_NVERR(
          err = NvAR_SetObject(m_lipSyncHandle, NvAR_Parameter_Input(Image), &g_src, sizeof(NvCVImage)), errSDK);
    }

    // Run the feature.
    err = NvAR_Run(m_lipSyncHandle);
    if (err == NVCV_ERR_OBJECTNOTFOUND) {
      std::cerr << "Warning: face not found in input image" << std::endl;
    } else {
      RETURN_APPERR_IF_NVERR(err, errSDK);
    }

    // If we are after the initial input frames, there should be an output frame available.
    if (output_ready) {
      app_err = ProcessOutputVideo();
      if (app_err) return app_err;
    }
  }

  return Err::errNone;
}

App::Err App::ProcessOutputVideo() {
  NvCV_Status err = NVCV_SUCCESS;
  cv::Mat o_dst;

  err = NvCVImage_Transfer(&m_gDst, &m_cDst, 1.f, m_stream, &m_tmp);
  (void)CVWrapperForNvCVImage(&m_cDst, &o_dst);

  if (FLAG_debug) {
    const int font_face = cv::FONT_HERSHEY_DUPLEX;
    const int pixel_height = m_srcHeight / 50;
    const int thickness = 1;
    const double font_scale = cv::getFontScaleFromHeight(font_face, pixel_height);
    const int pad = m_srcHeight / 200;

    // Set background color and text based on the active score.
    const cv::Scalar bg_color = cv::Scalar(0.0f, 255.0f * m_lipSyncActivation, 255.0f * (1.0f - m_lipSyncActivation));
    const std::string text = cv::format("LipSync Active: %3.1f", m_lipSyncActivation);

    // Compute size and location of the background rectangle and the text.
    int baseline = 0;
    const cv::Size text_size = cv::getTextSize(text, font_face, font_scale, thickness, &baseline);
    const cv::Point text_origin((m_srcWidth - text_size.width) / 2, pixel_height + text_size.height);
    const cv::Rect bg_rect(text_origin.x - pad, text_origin.y - text_size.height - pad,  //
                           text_size.width + 2 * pad, text_size.height + baseline + thickness + 2 * pad);
    cv::rectangle(o_dst, bg_rect, bg_color, -1);
    cv::putText(o_dst, text, text_origin, font_face, font_scale, cv::Scalar(0));
  }

  if (m_genVideo.isOpened()) {
    m_genVideo.write(o_dst);
  }
  return Err::errNone;
}

App::Err App::Stop(void) {
  if (m_cap.isOpened()) {
    m_cap.release();
  }
  if (m_genVideo.isOpened()) {
    m_genVideo.release();
  }
  return Err::errNone;
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
      {errImageSize, "the image size cannot be accommodated"},
      {errNotFound, "the item cannot be found"},
      {errNoFace, "no face has been found"},
      {errSDK, "an SDK error has occurred"},
      {errCuda, "a CUDA error has occurred"},
      {errCancel, "the user cancelled"},
      {errAudioFile, "unable to open source audio file"},
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

  // Webcam mode is currently not supported, check and error out if it is enabled.
  if (FLAG_offlineMode == false) {
    printf("ERROR: Webcam mode is not supported currently");
    goto bail;
  }

  if (FLAG_modelPath.empty()) {
    printf(
        "WARNING: Model path not specified. Please set --model_path=/path/to/trt/and/lipsync/models, "
        "SDK will attempt to load the models from NVAR_MODEL_DIR environment variable, "
        "please restart your application after the SDK Installation. \n");
  }

  if (FLAG_inVid.empty()) {
    app_err = App::errMissing;
    printf("ERROR: %s, please specify your source video file using --in_video \n", app.ErrorStringFromCode(app_err));
    goto bail;
  }

  if (FLAG_offlineMode) {
    if (FLAG_inAudio.empty()) {
      app_err = App::errMissing;
      printf("ERROR: %s, please specify source audio file using --in_audio in offline mode\n",
             app.ErrorStringFromCode(app_err));
      goto bail;
    }
    app_err = app.InitOfflineMode();
    BAIL_IF_ERR(app_err);
  } else {
    app_err = App::errMode;
    printf("ERROR: %s, Live capture mode not supported currently \n", app.ErrorStringFromCode(app_err));
    goto bail;
  }

  app_err = app.CreateEffect();
  BAIL_IF_ERR(app_err);

  app_err = app.InitOutput();
  BAIL_IF_ERR(app_err);

  app_err = app.Run();

  BAIL_IF_ERR(app_err);

  if (FLAG_offlineMode && FLAG_captureOutputs && FLAG_verbose) {
    printf("Output video saved at %s", FLAG_outFile.c_str());
  }

bail:
  if (app_err) printf("ERROR: %s\n", app.ErrorStringFromCode(app_err));
  app.Stop();
  return (int)app_err;
}
