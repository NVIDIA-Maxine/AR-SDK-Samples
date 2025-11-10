/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <iostream>
#include <memory>
#include <string>

#include "batchUtilities.h"
#include "nvAR.h"
#include "nvARLipSync.h"
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

namespace LipsyncConstants {
constexpr unsigned kInputSampleRate = 16000;
constexpr unsigned kAudioNumChannels = 1;
constexpr double kFPS = 30.0;
constexpr unsigned kNumAudioLookAheadFrames = 3;
}  // namespace LipsyncConstants

bool FLAG_verbose = false;
bool FLAG_useTritonGRPC = false;
std::string FLAG_tritonURL = "localhost:8001";
std::string FLAG_outputNameTag = "output";
std::string FLAG_outputCodec = "avc1";
std::string FLAG_outputFormat = "mp4";
std::string FLAG_log = "stderr";
std::vector<std::string> FLAG_srcVideoFiles;
std::vector<std::string> FLAG_srcAudioFiles;
unsigned FLAG_logLevel = NVCV_LOG_ERROR;
unsigned FLAG_headMovementSpeed = 0;  // set to default value for Head Movement Speed (SLOW)

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
      "LipSyncTritonClient [flags ...] inFile1 [inFileN ...]\n"
      "  where flags are:\n"
      "  --verbose[=(true|false)]           Print verbose information (default false).\n"
      "  --url=<URL>                        URL to the Triton server\n"
      "  --grpc[=(true|false)]              use gRPC for data transfer to the Triton server instead of CUDA shared "
      "memory.\n"
      "  --output_name_tag=<string>         a string appended to each input video file to create the corresponding "
      "output file name\n"
      "  --output_codec=<fourcc>            FOURCC code for the desired codec (default H264)\n"
      "  --output_format=<format>           Format of the output video (default mp4)\n"
      "  --log=<file>                       log SDK errors to a file, \"stderr\" or \"\" (default stderr)\n"
      "  --log_level=<N>                    the desired log level: {0, 1, 2} = {FATAL, ERROR, WARNING}, respectively "
      "(default 1)\n"
      "  --src_videos=<src1[, ...]>         Comma separated list of identically sized source video files\n"
      "  --src_audios=<src1[, ...]>         Comma separated list of source audio files\n"
      "  --head_movement_speed=<N>          Specify the expected speed of head motion in the input video: 0=SLOW, "
      "1=FAST. Default: 0 (SLOW)\n"
      "  --help                             Print out this message\n");
}

static int StringToFourcc(const std::string& str) {
  union chint {
    int i;
    char c[4];
  };
  chint x = {0};
  for (int n = (str.size() < 4) ? (int)str.size() : 4; n--;) x.c[n] = str[n];
  return x.i;
}
static int ParseMyArgs(int argc, char** argv) {
  int errs = 0;
  for (--argc, ++argv; argc--; ++argv) {
    bool help;
    const char* arg = *argv;
    if (arg[0] == '-') {
      if (arg[1] == '-') {                                                         // double-dash
        if (GetFlagArgVal("verbose", arg, &FLAG_verbose) ||                        //
            GetFlagArgVal("url", arg, &FLAG_tritonURL) ||                          //
            GetFlagArgVal("grpc", arg, &FLAG_useTritonGRPC) ||                     //
            GetFlagArgVal("output_name_tag", arg, &FLAG_outputNameTag) ||          //
            GetFlagArgVal("log", arg, &FLAG_log) ||                                //
            GetFlagArgVal("output_codec", arg, &FLAG_outputCodec) ||               //
            GetFlagArgVal("output_format", arg, &FLAG_outputFormat) ||             //
            GetFlagArgVal("head_movement_speed", arg, &FLAG_headMovementSpeed) ||  //
            GetFlagArgVal("log_level", arg, &FLAG_logLevel)) {
          continue;
        } else if (GetFlagArgVal("help", arg, &help)) {  // --help
          Usage();
          errs = 1;
        } else if (GetFlagArgValAndSplit("src_videos", arg, FLAG_srcVideoFiles)) {
          continue;
        } else if (GetFlagArgValAndSplit("src_audios", arg, FLAG_srcAudioFiles)) {
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
      FLAG_srcAudioFiles.push_back(arg);
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

class BaseApp {
 public:
  std::string m_effectName;
  NvAR_TritonServer m_triton;
  NvAR_FeatureHandle m_effect;
  NvCVImage m_srcVid, m_tmpImg;
  CUstream m_cudaStream;
  unsigned m_numOfStreams;
  unsigned m_outputImgVizWidth, m_outputImgVizHeight;
  std::vector<NvAR_StateHandle> m_arrayOfAllStateObjects;
  std::vector<NvAR_StateHandle> m_batchOfStateObjects;

  static BaseApp* Create(const char* effect_name);
  virtual ~BaseApp() {
    if (m_effect) {
      NvAR_Destroy(m_effect);
      m_effect = nullptr;
    }
    if (m_cudaStream) NvAR_CudaStreamDestroy(m_cudaStream);
    if (m_triton) NvAR_DisconnectTritonServer(m_triton);
  }
  virtual NvCV_Status Init(unsigned num_streams) {
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
  virtual NvCV_Status AllocateBuffers(unsigned src_vid_width, unsigned src_vid_height, unsigned num_streams) {
    return NVCV_SUCCESS;
  }
  virtual NvCV_Status SetParameters() { return NVCV_SUCCESS; }
  virtual NvCV_Status GetNumInitialFrames(unsigned int& num_initial_frames) { return NVCV_SUCCESS; }
  virtual NvCV_Status GenerateNthOutputVizImage(unsigned n, cv::Mat& result) = 0;
  virtual NvCV_Status Load() { return NvAR_Load(m_effect); }
  virtual NvCV_Status Run(std::vector<float>& audio_batched, std::vector<unsigned>& audio_num_samples,
                          const unsigned* batch_indices, unsigned batchsize) {
    NvCV_Status err = NVCV_SUCCESS;
    BAIL_IF_ERR(err = NvAR_SetF32Array(m_effect, NvAR_Parameter_Input(AudioFrameBuffer), audio_batched.data(), -1));

    BAIL_IF_ERR(err = NvAR_SetObject(m_effect, NvAR_Parameter_Input(AudioFrameLength), (void*)audio_num_samples.data(),
                                     batchsize));
    BAIL_IF_ERR(err = NvAR_SetU32(m_effect, NvAR_Parameter_Input(HeadMovementSpeed), FLAG_headMovementSpeed));

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
  virtual NvCV_Status InitStream(unsigned n) { return NvAR_AllocateState(m_effect, &m_arrayOfAllStateObjects[n]); }
  virtual NvCV_Status ReleaseStream(unsigned n) { return NvAR_DeallocateState(m_effect, m_arrayOfAllStateObjects[n]); }

 protected:
  BaseApp() : m_triton(nullptr), m_effect(nullptr), m_cudaStream(0), m_numOfStreams(0) {}
};

class LipsyncApp : public BaseApp {
 public:
  NvCVImage m_outVid, m_nthDstImg, m_nthSrcImg, m_firstSrcImg;
  NvCVImage m_nvTempResult, m_nthImg;

  NvCV_Status AllocateBuffers(unsigned src_vid_width, unsigned src_vid_height, unsigned num_streams) {
    NvCV_Status err = NVCV_SUCCESS;
    BAIL_IF_ERR(err = AllocateBatchBuffer(&m_srcVid, num_streams, src_vid_width, src_vid_height, NVCV_BGR, NVCV_U8,
                                          NVCV_CHUNKY, FLAG_useTritonGRPC ? NVCV_CPU : NVCV_CUDA, 1));
    // Allocate the output image
    BAIL_IF_ERR(err = AllocateBatchBuffer(&m_outVid, num_streams, src_vid_width, src_vid_height, NVCV_BGR, NVCV_U8,
                                          NVCV_CHUNKY, FLAG_useTritonGRPC ? NVCV_CPU : NVCV_CUDA, 1));

  bail:
    return err;
  }

  ~LipsyncApp() {
    NvCVImage_Dealloc(&m_outVid);
    NvCVImage_Dealloc(&m_nthDstImg);
    NvCVImage_Dealloc(&m_nthSrcImg);
    NvCVImage_Dealloc(&m_firstSrcImg);
    NvCVImage_Dealloc(&m_nvTempResult);
    NvCVImage_Dealloc(&m_nthImg);
  }

  // Source image, Generated image
  NvCV_Status SetParameters() {
    NvCV_Status err = NVCV_SUCCESS;
    cv::Mat src_img_cv_buffer;

    BAIL_IF_ERR(err = NvAR_SetObject(m_effect, NvAR_Parameter_Input(Image),
                                     NthImage(0, m_srcVid.height / m_numOfStreams, &m_srcVid, &m_firstSrcImg),
                                     sizeof(NvCVImage)));  // Set the first of the batched images in ...

    BAIL_IF_ERR(err = NvAR_SetObject(m_effect, NvAR_Parameter_Output(Image),
                                     NthImage(0, m_outVid.height / m_numOfStreams, &m_outVid, &m_nthDstImg),
                                     sizeof(NvCVImage)));  // Set the first of the batched images in ...
  bail:
    return err;
  }

  NvCV_Status GenerateNthOutputVizImage(unsigned n, cv::Mat& result) {
    // get NvAR_Parameter_Config(Ready) to check if the video is ready
    NvCV_Status err = NVCV_SUCCESS;
    unsigned* video_generation_ready;
    const void** video_generation_ready_ptr =
        const_cast<const void**>(reinterpret_cast<void**>(&video_generation_ready));
    BAIL_IF_ERR(err = NvAR_GetObject(m_effect, NvAR_Parameter_Output(Ready), video_generation_ready_ptr, 0));
    if (!video_generation_ready[n]) return NVCV_SUCCESS;
    result = cv::Mat(m_outVid.height / m_numOfStreams, m_outVid.width, CV_8UC3);
    NVWrapperForCVMat(&result, &m_nvTempResult);
    BAIL_IF_ERR(err = NvCVImage_Transfer(NthImage(n, m_outVid.height / m_numOfStreams, &m_outVid, &m_nthImg),
                                         &m_nvTempResult, 1, m_cudaStream, &m_tmpImg));

  bail:
    return err;
  }

  NvCV_Status GetNumInitialFrames(unsigned int& num_initial_frames) {
    NvCV_Status err = NVCV_SUCCESS;
    BAIL_IF_ERR(err = NvAR_GetU32(m_effect, NvAR_Parameter_Config(NumInitialFrames), &num_initial_frames));
  bail:
    return err;
  }
};

BaseApp* BaseApp::Create(const char* effect_name) {
  BaseApp* obj;

  if (!strcasecmp(effect_name, NvAR_Feature_LipSync))
    obj = new LipsyncApp;
  else
    return nullptr;
  obj->m_effectName = effect_name;
  return obj;
}

NvCV_Status BatchProcessVideos() {
  NvCV_Status err = NVCV_SUCCESS;
  unsigned num_streams = (unsigned)FLAG_srcAudioFiles.size();
  std::unique_ptr<BaseApp> app(BaseApp::Create(NvAR_Feature_LipSync));
  cv::Mat cv_img;
  NvCVImage nv_img;
  unsigned src_video_width = 0, src_video_height = 0;
  unsigned int init_latency_frame_count = 0;
  std::vector<std::vector<float>*> list_of_audio(num_streams);
  std::vector<cv::Mat> frames(num_streams), frames_t_1(num_streams);
  std::vector<unsigned> audio_nr_chunks(num_streams);
  std::vector<cv::Mat> src_img_buffer(num_streams);
  std::vector<cv::VideoWriter> list_of_writers(num_streams);
  std::vector<cv::VideoCapture> list_of_captures(num_streams);
  std::vector<unsigned> batch_indices(num_streams);
  std::vector<float> audio_frame_batched;  // Contiguous buffer of all audio samples from all streams
  std::vector<unsigned> audio_frame_num_samples(num_streams, 0);  // Number of audio frame samples in each stream
  float* audio_input;
  unsigned batchsize = 0;
  unsigned frame_count = 0;
  double fps;
  unsigned input_num_samples = 0;
  const unsigned int samples_per_second = LipsyncConstants::kInputSampleRate;  // 16000 Hz
  unsigned int last_audio_end_sample = 0;
  float estimated_video_frame_duration = 1.0f / LipsyncConstants::kFPS;
  std::vector<double> frame_timestamp(num_streams);
  std::vector<bool> audio_finished(num_streams, false);
  std::vector<int> flush_frames_remaining;
  float* activation = nullptr;
  const void** activation_ptr = const_cast<const void**>(reinterpret_cast<void**>(&activation));
  BAIL_IF_FALSE(app != nullptr, err, NVCV_ERR_UNIMPLEMENTED);
  BAIL_IF_FALSE(num_streams > 0, err, NVCV_ERR_MISSINGINPUT);
  BAIL_IF_FALSE(FLAG_outputFormat == "mp4" || FLAG_outputFormat == "avi", err, NVCV_ERR_GENERAL);

  // Assert video resolutions are the same
  for (unsigned i = 0; i < num_streams; i++) {
    // Open the video file
    list_of_captures[i].open(FLAG_srcVideoFiles[i], cv::CAP_FFMPEG);
    if (!list_of_captures[i].isOpened()) {
      printf("Error: Could not open %s.\n", FLAG_srcVideoFiles[i].c_str());
      return NVCV_ERR_READ;
    }

    // Retrieve resolution from metadata with implicit conversion
    unsigned width = list_of_captures[i].get(cv::CAP_PROP_FRAME_WIDTH);
    unsigned height = list_of_captures[i].get(cv::CAP_PROP_FRAME_HEIGHT);

    if (width == 0 || height == 0) {
      printf("Error: Could not retrieve resolution for %s.\n", FLAG_srcVideoFiles[i].c_str());
      return NVCV_ERR_READ;
    }

    if (i == 0) {
      src_video_width = width;
      src_video_height = height;
    }
    // For subsequent videos, compare against the first video's resolution
    else if (src_video_width != width || src_video_height != height) {
      printf("Error: Resolution of the videos must be the same.\n");
      return NVCV_ERR_MISMATCH;
    }

    list_of_captures[i].set(cv::CAP_PROP_POS_FRAMES, 0);
  }

  // Read audio
  for (int i = 0; i < num_streams; i++) {
    if (!ReadWavFile(FLAG_srcAudioFiles[i], LipsyncConstants::kInputSampleRate, LipsyncConstants::kAudioNumChannels,
                     &list_of_audio[i], &input_num_samples, nullptr, -1, FLAG_verbose)) {
      printf("Unable to read wav file: %s\n", FLAG_srcAudioFiles[i].c_str());
      return NVCV_ERR_READ;
    }
  }

  BAIL_IF_ERR(err = app->Init(num_streams));                                                // Init effect
  BAIL_IF_ERR(err = app->AllocateBuffers(src_video_width, src_video_height, num_streams));  // Allocate buffers
  BAIL_IF_ERR(err = app->SetParameters());                                                  // Set IO and config
  BAIL_IF_ERR(err = app->Load());                                                           // Load the feature
  BAIL_IF_ERR(err = app->GetNumInitialFrames(init_latency_frame_count));
  flush_frames_remaining.resize(num_streams, init_latency_frame_count);

  for (unsigned i = 0; i < num_streams; i++) {
    if (!list_of_captures[i].isOpened()) continue;  // if video is not opened, we skip
    list_of_captures[i] >> frames[i];
    if (frames[i].empty() || audio_finished[i])  // if nothing read
      list_of_captures[i].release();             // closing the video
    else                                         // if a frame is read
      BAIL_IF_ERR(app->InitStream(i));           // Initialize stream
  }

  // Open video writers
  for (unsigned i = 0; i < num_streams; i++) {
    size_t period_loc = std::string(FLAG_srcVideoFiles[i]).find_last_of(".");
    std::string dst_video = std::string(FLAG_srcVideoFiles[i]).substr(0, period_loc);
    dst_video = dst_video + "_" + FLAG_outputNameTag + "." + FLAG_outputFormat;

    list_of_writers[i].open(dst_video, StringToFourcc(FLAG_outputCodec), LipsyncConstants::kFPS,
                            cv::Size(src_video_width, src_video_height));
    if (!list_of_writers[i].isOpened()) {
      printf("Error: Could not open video writer for video %s.\n", dst_video.c_str());
      return NVCV_ERR_WRITE;
    }
  }

  while (1) {
    // Read inputs
    batchsize = 0;  // batchsize = number of active videos

    for (unsigned i = 0; i < num_streams; i++) {
      frame_timestamp[i] += 1.0f / static_cast<double>(LipsyncConstants::kFPS);
      if (list_of_captures[i].isOpened()) {
        list_of_captures[i] >> frames_t_1[i];  // Reading the next frame to know if the video has ended
                                               // as it is not possible to know if current frame is last
                                               // without reading the next frame
        if (frames_t_1[i].empty() || audio_finished[i]) {
          if (FLAG_verbose) {
            if (frames_t_1[i].empty()) {
              printf("Video Stream %d ending at frame %d\n ", i, frame_count);
              list_of_captures[i].release();  // release the capture if video stream is ending
            }
          }
        }
      }

      if (frames[i].empty() || audio_finished[i]) {
        if (flush_frames_remaining[i] > 0) {
          // The LipSync feature has internal latency/lookahead that requires additional frames
          // to generate complete output for the last few input frames.
          flush_frames_remaining[i]--;
          if (FLAG_verbose) {
            printf("Flush frames remaining for stream %d: %d\n", i, flush_frames_remaining[i]);
          }
          if (flush_frames_remaining[i] == 0) {
            // Current frame is the last frame to be processed for this stream
            app->ReleaseStream(i);  // Triton requires NvAR_DeallocateState() to be called just before the last
                                    // inference for that video stream
          }
        } else {
          if (audio_finished[i]) {
            list_of_captures[i].release();  // release the capture if audio is finished
          }
          continue;
        }
      }
      if (!frames[i].empty()) {
        NVWrapperForCVMat(&frames[i], &nv_img);
        BAIL_IF_ERR(err = TransferToNthImage(batchsize, &nv_img, &app->m_srcVid, 1, app->m_cudaStream, &app->m_tmpImg));
      }
      std::vector<float> audio_frame;
      unsigned int audio_start_sample = last_audio_end_sample;
      unsigned requested_audio_end_sample = static_cast<unsigned int>(frame_timestamp[i] * samples_per_second);
      unsigned int audio_end_sample =
          std::min(static_cast<size_t>(requested_audio_end_sample), list_of_audio[i]->size());
      // Store end sample for next frame.
      last_audio_end_sample = requested_audio_end_sample;
      // Pad with zeros for when audio is finished
      if (audio_finished[i]) {
        audio_frame.insert(audio_frame.end(), requested_audio_end_sample - audio_start_sample, 0.0f);
      } else {
        audio_frame.insert(audio_frame.end(), list_of_audio[i]->begin() + audio_start_sample,
                           list_of_audio[i]->begin() + audio_end_sample);
        // If we need padding, insert the required number of zeros and set audio_finished to true
        if (requested_audio_end_sample >= list_of_audio[i]->size()) {
          unsigned int num_padding = requested_audio_end_sample - audio_end_sample;
          if (FLAG_verbose) {
            printf("Audio Stream %d ending at frame %d\n", i, frame_count);
          }
          audio_finished[i] = true;
          audio_frame.insert(audio_frame.end(), num_padding, 0.0f);
        }
      }

      audio_frame_batched.insert(audio_frame_batched.end(), audio_frame.begin(), audio_frame.end());
      audio_frame_num_samples[batchsize] = audio_frame.size();
      batch_indices[batchsize] = i;  // storing video indices for creating output videos
      batchsize++;                   // counting the number of active videos
    }

    if (batchsize == 0) break;
    if (FLAG_verbose) {
      std::cout << "Batchsize : " << batchsize << std::endl;
    }

    // Run batched inference
    BAIL_IF_ERR(err = app->Run(audio_frame_batched, audio_frame_num_samples, batch_indices.data(), batchsize));

    // Get activation values for the batch
    BAIL_IF_ERR(err = NvAR_GetObject(app->m_effect, NvAR_Parameter_Output(Activation), activation_ptr, 0));

    for (unsigned i = 0; i < batchsize; i++) {
      unsigned video_idx = batch_indices[i];

      // Write Output
      cv::Mat display_frame;
      BAIL_IF_ERR(err = app->GenerateNthOutputVizImage(i, display_frame));
      if (!display_frame.empty()) {
        list_of_writers[video_idx] << display_frame;
      }

      // Update current frame
      frames[video_idx] = frames_t_1[video_idx].clone();  // copying the t+1 frame to current frame
      if (FLAG_verbose && activation) {
        std::cout << "Activation value for video " << video_idx << " for frame " << frame_count << " is "
                  << activation[i] << std::endl;
      }
    }
    if (FLAG_verbose) {
      std::cout << "Finished processing for frame index : " << frame_count << std::endl;
    }
    frame_count++;
    audio_frame_batched.clear();
    audio_frame_num_samples.clear();
  }
bail:
  for (auto& writer : list_of_writers) writer.release();
  return err;
}

int main(int argc, char** argv) {
  int num_errs;
  NvCV_Status nv_err;

  num_errs = ParseMyArgs(argc, argv);
  if (num_errs) return num_errs;

  nv_err = NvAR_ConfigureLogger(FLAG_logLevel, FLAG_log.c_str(), nullptr, nullptr);
  if (NVCV_SUCCESS != nv_err)
    printf("%s: while configuring logger to \"%s\"\n", NvCV_GetErrorStringFromCode(nv_err), FLAG_log.c_str());
  nv_err = BatchProcessVideos();
  if (NVCV_SUCCESS != nv_err) {
    printf("Error: %s\n", NvCV_GetErrorStringFromCode(nv_err));
    num_errs = (int)nv_err;
  }
  return num_errs;
}
