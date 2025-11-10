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

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include <memory>
#include <string>

#include "batchUtilities.h"
#include "nvAR.h"
#include "nvCVOpenCV.h"
#include "opencv2/opencv.hpp"
#include "waveReadWrite.h"

#define BAIL_IF_ERR(err) \
  do {                   \
    if (0 != (err)) {    \
      goto bail;         \
    }                    \
  } while (0)
#define BAIL_IF_FALSE(x, err, code) \
  do {                              \
    if (!(x)) {                     \
      err = code;                   \
      goto bail;                    \
    }                               \
  } while (0)

char* g_nvARSDKPath = NULL;

namespace SpeechLPConstants {
constexpr int kModeCropNone = 0;
constexpr int kModeCropFaceBox = 1;
constexpr int kModeCropBlend = 2;
constexpr int kModeCropInsetBlend = 3;
constexpr int kModelSelPerf = 0;
constexpr int kModelSelQual = 1;
constexpr int kInputSampleRate = 16000;
constexpr int kAudioNumChannels = 1;
constexpr int kSamplesPerFrame = 528;
constexpr int kfps = 30.3;
constexpr int kInitLatencyFrameCnt = 6;
}  // namespace SpeechLPConstants

bool FLAG_verbose = false;
bool FLAG_isLandmarks126 = false;
bool FLAG_useTritonGRPC = false;
std::string FLAG_tritonURL = "localhost:8001";
std::string FLAG_outputNameTag = "output";
std::string FLAG_log = "stderr";
std::vector<std::string> FLAG_srcImages;
std::vector<std::string> FLAG_inDrvAudioFiles;
unsigned FLAG_logLevel = NVCV_LOG_ERROR;
unsigned FLAG_slpMode = SpeechLPConstants::kModeCropFaceBox;
unsigned FLAG_slpModelSel = SpeechLPConstants::kModelSelQual;
bool FLAG_ignoreAlpha = false;  // ignore the alpha channel of the source image (RGBA format only)
bool FLAG_showBboxes = false;

static bool GetFlagArgVal(const char* flag, const char* arg, const char** val) {
  if (*arg != '-') return false;
  while (*++arg == '-') continue;
  const char* s = strchr(arg, '=');
  if (s == NULL) {
    if (strcmp(flag, arg) != 0) return false;
    *val = NULL;
    return true;
  }
  size_t n = s - arg;
  if ((strlen(flag) != n) || (strncmp(flag, arg, n) != 0)) return false;
  *val = s + 1;
  return true;
}

static bool GetFlagArgVal(const char* flag, const char* arg, std::string* val) {
  const char* valStr;
  if (!GetFlagArgVal(flag, arg, &valStr)) return false;
  val->assign(valStr ? valStr : "");
  return true;
}

static bool GetFlagArgVal(const char* flag, const char* arg, bool* val) {
  const char* valStr;
  bool success = GetFlagArgVal(flag, arg, &valStr);
  if (success) {
    *val = (valStr == NULL || strcasecmp(valStr, "true") == 0 || strcasecmp(valStr, "on") == 0 ||
            strcasecmp(valStr, "yes") == 0 || strcasecmp(valStr, "1") == 0);
  }
  return success;
}

static bool GetFlagArgVal(const char* flag, const char* arg, float* val) {
  const char* valStr;
  bool success = GetFlagArgVal(flag, arg, &valStr);
  if (success) *val = strtof(valStr, NULL);
  return success;
}

static bool GetFlagArgVal(const char* flag, const char* arg, long* val) {
  const char* valStr;
  bool success = GetFlagArgVal(flag, arg, &valStr);
  if (success) *val = strtol(valStr, NULL, 10);
  return success;
}

static bool GetFlagArgVal(const char* flag, const char* arg, unsigned* val) {
  long longVal;
  bool success = GetFlagArgVal(flag, arg, &longVal);
  if (success) {
    *val = (unsigned)longVal;
  }
  return success;
}

static bool GetFlagArgValAndSplit(const char* flag, const char* arg, std::vector<std::string>& vals) {
  const char* valStr;
  if (!GetFlagArgVal(flag, arg, &valStr)) return false;

  if (valStr) {
    std::string value(valStr);
    std::istringstream iss(value);
    std::string part;
    while (std::getline(iss, part, ',')) {
      if (!part.empty()) {  // Making sure not to add empty strings
        vals.push_back(part);
      }
    }
  }
  return true;
}

static void Usage() {
  printf(
      "SpeechLivePortraitTritonClient [flags ...] inFile1 [inFileN ...]\n"
      "  where flags are:\n"
      "  --verbose[=(true|false)]           Print interesting information (default false).\n"
      "  --url=<URL>                        URL to the Triton server\n"
      "  --grpc[=(true|false)]              use gRPC for data transfer to the Triton server instead of CUDA shared "
      "memory.\n"
      "  --output_name_tag=<string>         a string appended to each inFile to create the corresponding output file "
      "name\n"
      "  --log=<file>                       log SDK errors to a file, \"stderr\" or \"\" (default stderr)\n"
      "  --log_level=<N>                    the desired log level: {0, 1, 2} = {FATAL, ERROR, WARNING}, respectively "
      "(default 1)\n"
      "  --mode                             Live Portrait Mode %d: Crop (Default), %d: Registration Blend %d: Inset "
      "Blend\n"
      "  --src_images=<src1[, ...]>         comma separated list of identically sized source images\n"
      "  --model_sel                        Live Portrait Model. 0: Performance, 1: Quality(Default)\n"
      "  --show_bounding_boxes              Show face bounding boxes in the output video. only available in mode 2 and "
      "3 (default false)\n"
      "  --ignore_alpha                     Ignore the alpha channel of a RBGA input source image (default false)\n"
      "  --help                             Print out this message\n",
      SpeechLPConstants::kModeCropFaceBox, SpeechLPConstants::kModeCropBlend, SpeechLPConstants::kModeCropInsetBlend);
}

static int ParseMyArgs(int argc, char** argv) {
  int errs = 0;
  for (--argc, ++argv; argc--; ++argv) {
    bool help;
    const char* arg = *argv;
    if (arg[0] == '-') {
      if (arg[1] == '-') {                                                  // double-dash
        if (GetFlagArgVal("verbose", arg, &FLAG_verbose) ||                 //
            GetFlagArgVal("url", arg, &FLAG_tritonURL) ||                   //
            GetFlagArgVal("grpc", arg, &FLAG_useTritonGRPC) ||              //
            GetFlagArgVal("output_name_tag", arg, &FLAG_outputNameTag) ||   //
            GetFlagArgVal("log", arg, &FLAG_log) ||                         //
            GetFlagArgVal("log_level", arg, &FLAG_logLevel) ||              //
            GetFlagArgVal("mode", arg, &FLAG_slpMode) ||                    //
            GetFlagArgVal("model_sel", arg, &FLAG_slpModelSel) ||           //
            GetFlagArgVal("show_bounding_boxes", arg, &FLAG_showBboxes) ||  //
            GetFlagArgVal("ignore_alpha", arg, &FLAG_ignoreAlpha)) {
          continue;
        } else if (GetFlagArgVal("help", arg, &help)) {  // --help
          Usage();
          errs = 1;
        } else if (GetFlagArgValAndSplit("src_images", arg, FLAG_srcImages)) {
          continue;
        }
      } else {  // single dash
        for (++arg; *arg; ++arg) {
          if (*arg == 'v') {
            FLAG_verbose = true;
          } else {
            printf("Unknown flag: \"-%c\"\n", *arg);
            Usage();
            errs = 1;
            break;
          }
        }
        continue;
      }
    } else {  // no dash
      FLAG_inDrvAudioFiles.push_back(arg);
    }
  }
  return errs;
}

static bool HasSuffix(const char* str, const char* suf) {
  size_t strSize = strlen(str), sufSize = strlen(suf);
  if (strSize < sufSize) return false;
  return (0 == strcasecmp(suf, str + strSize - sufSize));
}

static bool HasOneOfTheseSuffixes(const char* str, ...) {
  bool matches = false;
  const char* suf;
  va_list ap;
  va_start(ap, str);
  while (nullptr != (suf = va_arg(ap, const char*))) {
    if (HasSuffix(str, suf)) {
      matches = true;
      break;
    }
  }
  va_end(ap);
  return matches;
}

class SpeechLivePortraitApp {
 public:
  std::string m_effectName;
  NvAR_TritonServer m_triton;
  NvAR_FeatureHandle m_effect;
  NvCVImage m_firstSrc, m_tmpImg;
  CUstream m_cudaStream;
  unsigned m_numOfStreams;
  unsigned m_outputImgVizWidth, m_outputImgVizHeight;
  std::vector<NvAR_StateHandle> m_arrayOfAllStateObjects;
  std::vector<NvAR_StateHandle> m_batchOfStateObjects;
  NvCVImage m_dst, m_firstDst, m_srcImg, m_nthSrcImg, m_firstSrcImg;
  NvCVImage m_nvTempResult, m_nthImg;
  std::vector<NvAR_BBoxes> m_outputBboxes;
  std::vector<std::vector<NvAR_Rect>> m_outputBboxData;
  static constexpr int kMaxBoxes = 25;
  bool m_srcAlpha = false;

  SpeechLivePortraitApp()
      : m_triton(nullptr), m_effect(nullptr), m_cudaStream(0), m_numOfStreams(0), m_effectName("SpeechLivePortrait") {}

  ~SpeechLivePortraitApp() {
    if (m_effect) NvAR_Destroy(m_effect);
    if (m_cudaStream) NvAR_CudaStreamDestroy(m_cudaStream);
    if (m_triton) NvAR_DisconnectTritonServer(m_triton);
  }

  NvCV_Status InitStream(unsigned n) { return NvAR_AllocateState(m_effect, &m_arrayOfAllStateObjects[n]); }
  NvCV_Status ReleaseVideoStream(unsigned n) { return NvAR_DeallocateState(m_effect, m_arrayOfAllStateObjects[n]); }

  NvCV_Status Init(unsigned num_streams) {
    NvCV_Status err = NVCV_SUCCESS;
    m_numOfStreams = num_streams;
    err = NvAR_ConnectTritonServer(FLAG_tritonURL.c_str(), &m_triton);
    if (err != NVCV_SUCCESS) printf("Error connecting to the server at %s.\n", FLAG_tritonURL.c_str());
    BAIL_IF_ERR(err);
    err = NvAR_CreateTriton(m_effectName.c_str(), &m_effect);
    if (err != NVCV_SUCCESS)
      printf("Error creating the %s feature on the server at %s.\n", m_effectName.c_str(), FLAG_tritonURL.c_str());
    BAIL_IF_ERR(err);
    err = NvAR_SetTritonServer(m_effect, m_triton);
    if (err != NVCV_SUCCESS)
      printf("Error creating the %s feature on the server at %s.\n", m_effectName.c_str(), FLAG_tritonURL.c_str());
    BAIL_IF_ERR(err);
    m_arrayOfAllStateObjects.resize(m_numOfStreams, nullptr);
    m_batchOfStateObjects.resize(m_numOfStreams, nullptr);
    if (FLAG_verbose) {
      printf("Using triton server\n");
    }
  bail:
    return err;
  }

  NvCV_Status AllocateBuffers() {
    NvCV_Status err = NVCV_SUCCESS;

    if (FLAG_srcImages.size() != m_numOfStreams) {
      printf("Error: Number of source images does not match the number of video streams.\n");
      return NVCV_ERR_READ;
    }

    // Reading source image to get the width and height
    cv::Mat tmp_img = cv::imread(FLAG_srcImages[0], cv::ImreadModes::IMREAD_UNCHANGED);
    if (tmp_img.channels() == 4) {
      if (FLAG_ignoreAlpha && FLAG_verbose) {
        printf("The alpha channel of the source image will be ignored.\n");
      }
      m_srcAlpha = !FLAG_ignoreAlpha;
    }
    if (!tmp_img.data) {
      printf("Error: Could not read %s.\n", FLAG_srcImages[0].c_str());
      return NVCV_ERR_READ;
    }
    unsigned src_img_width = tmp_img.cols;
    unsigned src_img_height = tmp_img.rows;

    BAIL_IF_ERR(err = AllocateBatchBuffer(&m_srcImg, m_numOfStreams, src_img_width, src_img_height,
                                          m_srcAlpha ? NVCV_BGRA : NVCV_BGR, NVCV_U8, NVCV_CHUNKY,
                                          FLAG_useTritonGRPC ? NVCV_CPU : NVCV_CUDA, 1));  // same as src image

    // Allocate bounding boxes
    m_outputBboxes.resize(m_numOfStreams);
    m_outputBboxData.resize(m_numOfStreams);

    // bbox
    for (int i = 0; i < m_numOfStreams; i++) {
      m_outputBboxData[i].resize(kMaxBoxes);
      m_outputBboxData[i].assign(kMaxBoxes, {0.f, 0.f, 0.f, 0.f});
      m_outputBboxes[i].boxes = m_outputBboxData[i].data();
      m_outputBboxes[i].max_boxes = kMaxBoxes;
      m_outputBboxes[i].num_boxes = 0;
    }
  bail:
    return err;
  }

  NvCV_Status SetParametersBeforeLoad() {
    NvCV_Status err = NVCV_SUCCESS;
    BAIL_IF_ERR(err = NvAR_SetU32(m_effect, NvAR_Parameter_Config(Mode),
                                  FLAG_slpMode));  // Need to set mode first before setting other image parameters
    BAIL_IF_ERR(err = NvAR_SetU32(m_effect, NvAR_Parameter_Config(ModelSel), FLAG_slpModelSel));
  bail:
    return err;
  }

  // source image, generated image
  NvCV_Status SetParametersAfterLoad() {
    NvCV_Status err = NVCV_SUCCESS;
    cv::Mat src_img_cv_buffer;

    for (int stream_idx = 0; stream_idx < m_numOfStreams; stream_idx++) {
      src_img_cv_buffer = cv::imread(FLAG_srcImages[stream_idx].c_str());
      if (src_img_cv_buffer.empty()) {
        printf("Error: Could not read %s.\n", FLAG_srcImages[stream_idx].c_str());
        return NVCV_ERR_READ;
      }
      NVWrapperForCVMat(&src_img_cv_buffer, &m_nthSrcImg);
      TransferToNthImage(stream_idx, &m_nthSrcImg, &m_srcImg, 1, m_cudaStream, &m_tmpImg);
    }

    BAIL_IF_ERR(err = NvAR_SetObject(m_effect, NvAR_Parameter_Input(SourceImage),
                                     NthImage(0, m_srcImg.height / m_numOfStreams, &m_srcImg, &m_firstSrcImg),
                                     sizeof(NvCVImage)));  // Set the first of the batched images in ...

    BAIL_IF_ERR(err = NvAR_SetObject(m_effect, NvAR_Parameter_Output(BoundingBoxes), m_outputBboxes.data(),
                                     sizeof(NvAR_BBoxes)));

    // Get the output image size if live portrait mode is 1
    if (FLAG_slpMode == SpeechLPConstants::kModeCropFaceBox) {
      m_outputImgVizWidth = 512;
      m_outputImgVizHeight = 512;
    } else {
      m_outputImgVizWidth = m_srcImg.width;
      m_outputImgVizHeight = m_srcImg.height / m_numOfStreams;
    }
    // Allocate the output image
    BAIL_IF_ERR(err = AllocateBatchBuffer(&m_dst, m_numOfStreams, m_outputImgVizWidth, m_outputImgVizHeight,
                                          m_srcAlpha ? NVCV_BGRA : NVCV_BGR, NVCV_U8, NVCV_CHUNKY,
                                          FLAG_useTritonGRPC ? NVCV_CPU : NVCV_CUDA, 1));

    BAIL_IF_ERR(err = NvAR_SetObject(m_effect, NvAR_Parameter_Output(GeneratedImage),
                                     NthImage(0, m_dst.height / m_numOfStreams, &m_dst, &m_firstDst),
                                     sizeof(NvCVImage)));  // Set the first of the batched images in ...
  bail:
    return err;
  }

  NvCV_Status Load() { return NvAR_Load(m_effect); }

  NvCV_Status Run(std::vector<float>& audio_batched, const unsigned* batch_indices, unsigned batchsize) {
    NvCV_Status err = NVCV_SUCCESS;

    if (err = NvAR_SetF32Array(m_effect, NvAR_Parameter_Input(AudioFrameBuffer), audio_batched.data(),
                               batchsize * SpeechLPConstants::kSamplesPerFrame)) {
      printf("%s\n", NvCV_GetErrorStringFromCode(err));
      return err;
    }

    for (int i = 0; i < batchsize; i++) {
      m_batchOfStateObjects[i] = m_arrayOfAllStateObjects[batch_indices[i]];
    }
    BAIL_IF_ERR(err = NvAR_SetU32(m_effect, NvAR_Parameter_Config(BatchSize), batchsize));
    BAIL_IF_ERR(err = NvAR_SetObject(m_effect, NvAR_Parameter_InOut(State), m_batchOfStateObjects.data(),
                                     batchsize));  // This can change every Run
    BAIL_IF_ERR(err = NvAR_Run(m_effect));
    BAIL_IF_ERR(err = NvAR_SynchronizeTriton(m_effect));
  bail:
    return err;
  }

  NvCV_Status GenerateNthOutputVizImage(unsigned n, cv::Mat& result) {
    // get NvAR_Parameter_Config(VideoGenerationReady) to check if the video is ready
    NvCV_Status err = NVCV_SUCCESS;
    unsigned* video_generation_ready;
    const void** video_generation_ready_ptr =
        const_cast<const void**>(reinterpret_cast<void**>(&video_generation_ready));
    BAIL_IF_ERR(
        err = NvAR_GetObject(m_effect, NvAR_Parameter_Output(VideoGenerationReady), video_generation_ready_ptr, 0));
    if (!video_generation_ready[n]) return NVCV_SUCCESS;
    result = cv::Mat(m_dst.height / m_numOfStreams, m_dst.width, CV_8UC3);
    NVWrapperForCVMat(&result, &m_nvTempResult);
    BAIL_IF_ERR(err = NvCVImage_Transfer(NthImage(n, m_outputImgVizHeight, &m_dst, &m_nthImg), &m_nvTempResult, 1,
                                         m_cudaStream, &m_tmpImg));

    if (FLAG_showBboxes && FLAG_verbose) {
      printf("Num boxes detected in stream %d : %d\n", n, m_outputBboxes[n].num_boxes);

      for (int i = 0; i < unsigned(m_outputBboxes[n].num_boxes); i++) {
        printf("Bounding box number %d : %f %f %f %f\n", i, m_outputBboxes[n].boxes[i].x, m_outputBboxes[n].boxes[i].y,
               m_outputBboxes[n].boxes[i].width, m_outputBboxes[n].boxes[i].height);
      }
      // nawu todo: bbox overlay if in mode2 and mode3 and show_bbox
    }

  bail:
    return err;
  }
};

NvCV_Status BatchProcessVideos() {
  NvCV_Status err = NVCV_SUCCESS;
  unsigned num_streams = (unsigned)FLAG_inDrvAudioFiles.size();
  std::unique_ptr<SpeechLivePortraitApp> app(new SpeechLivePortraitApp());
  cv::Mat cv_img;
  NvCVImage nv_img;
  unsigned src_video_width = 0, src_video_height = 0;
  std::vector<std::vector<float>*> list_of_audio(num_streams);
  std::vector<unsigned> audio_nr_chunks(num_streams);
  std::vector<cv::Mat> src_img_buffer(num_streams);
  std::vector<cv::VideoWriter> list_of_writers(num_streams);
  std::vector<unsigned> batch_indices(num_streams);
  float* audio_input;
  double fps;
  unsigned input_num_samples = 0;

  BAIL_IF_FALSE(app != nullptr, err, NVCV_ERR_UNIMPLEMENTED);
  BAIL_IF_FALSE(num_streams > 0, err, NVCV_ERR_MISSINGINPUT);

  for (int i = 0; i < num_streams; i++) {
    if (!ReadWavFile(FLAG_inDrvAudioFiles[i], SpeechLPConstants::kInputSampleRate, SpeechLPConstants::kAudioNumChannels,
                     &list_of_audio[i], &input_num_samples, nullptr, SpeechLPConstants::kSamplesPerFrame,
                     FLAG_verbose)) {
      printf("Unable to read wav file: %s\n", FLAG_inDrvAudioFiles[i].c_str());
    }
    audio_nr_chunks[i] = (*list_of_audio[i]).size() / SpeechLPConstants::kSamplesPerFrame;
  }

  BAIL_IF_ERR(err = app->Init(num_streams));          // Init effect
  BAIL_IF_ERR(err = app->AllocateBuffers());          // Allocate buffers
  BAIL_IF_ERR(err = app->SetParametersBeforeLoad());  // Set IO and config
  BAIL_IF_ERR(err = app->Load());                     // Load the feature
  BAIL_IF_ERR(err = app->SetParametersAfterLoad());   // Set IO and config

  for (unsigned i = 0; i < num_streams; i++) {
    if (audio_nr_chunks[i] == 0)
      continue;
    else                                // if a frame is read
      BAIL_IF_ERR(app->InitStream(i));  // initialize stream. Allocate state for each stream.
  }

  // Open video writers
  for (unsigned i = 0; i < num_streams; i++) {
    size_t period_loc = std::string(FLAG_inDrvAudioFiles[i]).find_last_of(".");
    std::string dst_video = std::string(FLAG_inDrvAudioFiles[i]).substr(0, period_loc);
    dst_video = dst_video + "_" + FLAG_outputNameTag + ".mp4";
    fps = SpeechLPConstants::kfps;

    list_of_writers[i].open(dst_video, cv::VideoWriter::fourcc('a', 'v', 'c', '1'), fps,
                            cv::Size(app->m_outputImgVizWidth, app->m_outputImgVizHeight));
    if (!list_of_writers[i].isOpened()) {
      printf("Error: Could not open video writer for video %s.\n", dst_video.c_str());
      return NVCV_ERR_WRITE;
    }
  }

  for (int audio_chunk_idx = 0;; audio_chunk_idx++) {
    // Read inputs
    unsigned batchsize = 0;  // batchsize = number of active videos
    std::vector<float> audio_frame_batched;

    for (unsigned i = 0; i < num_streams; i++) {
      std::vector<float>::iterator offset;
      std::vector<float> audio_frame;
      if (audio_chunk_idx >= audio_nr_chunks[i] + SpeechLPConstants::kInitLatencyFrameCnt) {
        continue;
      } else if (audio_chunk_idx == audio_nr_chunks[i] + SpeechLPConstants::kInitLatencyFrameCnt - 1) {
        audio_frame.assign(SpeechLPConstants::kSamplesPerFrame, 0.f);
        BAIL_IF_ERR(app->ReleaseVideoStream(i));           // free state
      } else if (audio_chunk_idx >= audio_nr_chunks[i]) {  // Flush: retrieve the last few frames from the pipeline
        audio_frame.assign(SpeechLPConstants::kSamplesPerFrame, 0.f);
      } else {
        offset = (*list_of_audio[i]).begin() + audio_chunk_idx * SpeechLPConstants::kSamplesPerFrame;
        audio_frame.assign(offset, offset + SpeechLPConstants::kSamplesPerFrame);
      }

      audio_frame_batched.insert(audio_frame_batched.end(), audio_frame.begin(), audio_frame.end());
      batch_indices[batchsize] = i;  // storing video indices for creating output videos
      batchsize++;                   // counting the number of active videos
    }

    if (batchsize == 0) break;

    // Run batched inference
    BAIL_IF_ERR(err = app->Run(audio_frame_batched, batch_indices.data(), batchsize));

    // Write Output
    for (unsigned i = 0; i < batchsize; i++) {
      unsigned video_idx = batch_indices[i];
      cv::Mat display_frame;
      BAIL_IF_ERR(err = app->GenerateNthOutputVizImage(i, display_frame));
      if (!display_frame.empty()) {
        list_of_writers[video_idx] << display_frame;
      }
    }
  }
bail:
  for (auto& writer : list_of_writers) writer.release();
  return err;
}

int main(int argc, char** argv) {
  int num_errs;
  NvCV_Status nv_errs;

  num_errs = ParseMyArgs(argc, argv);
  if (num_errs) return num_errs;

  nv_errs = NvAR_ConfigureLogger(FLAG_logLevel, FLAG_log.c_str(), nullptr, nullptr);
  if (NVCV_SUCCESS != nv_errs)
    printf("%s: while configuring logger to \"%s\"\n", NvCV_GetErrorStringFromCode(nv_errs), FLAG_log.c_str());
  nv_errs = BatchProcessVideos();
  if (NVCV_SUCCESS != nv_errs) {
    printf("Error: %s\n", NvCV_GetErrorStringFromCode(nv_errs));
    num_errs = (int)nv_errs;
  }
  return num_errs;
}
