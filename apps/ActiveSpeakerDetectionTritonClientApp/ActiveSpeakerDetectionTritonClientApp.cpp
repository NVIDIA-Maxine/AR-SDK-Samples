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

#include <stdio.h>
#include <string.h>

#include <deque>
#include <sstream>
#include <string>

#include "batchUtilities.h"
#include "nvAR.h"
#include "nvARActiveSpeakerDetection.h"
#include "nvCVOpenCV.h"
#include "waveReadWrite.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Macros                                                                                                             //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// clang-format off
 #define BAIL_IF_ERR(err) do { if (0 != (err)) { goto bail; } } while (0)
 #define BAIL_IF_FALSE(x, err, code) do { if (!(x)) { err = code; goto bail; } } while (0)
 #define BAIL(err, code) do { err = code; goto bail; } while (0)
// clang-format on

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Global variables                                                                                                   //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

char* g_nvARSDKPath = NULL;

bool FLAG_verbose = false;
bool FLAG_useTritonGRPC = false;
std::string FLAG_tritonURL = "localhost:8001";
std::string FLAG_outputNameTag = "output";
std::string FLAG_outputCodec = "mp4v";
std::string FLAG_outputFormat = "mp4";
std::string FLAG_log = "stderr";
std::vector<std::string> FLAG_srcVideoFiles;
std::vector<std::vector<std::string>> FLAG_srcAudioFilesPerVideo;
uint32_t FLAG_logLevel = NVCV_LOG_ERROR;
float FLAG_syncTolerance = -1.0f;  // [0, 1] to set; -1 = unset (use feature default)

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Static function declarations                                                                                       //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static bool GetFlagArgVal(const char* flag, const char* arg, const char** val);
static bool GetFlagArgVal(const char* flag, const char* arg, std::string* val);
static bool GetFlagArgVal(const char* flag, const char* arg, bool* val);
static bool GetFlagArgVal(const char* flag, const char* arg, int64_t* val);
static bool GetFlagArgVal(const char* flag, const char* arg, uint32_t* val);
static bool GetFlagArgVal(const char* flag, const char* arg, float* val);
static bool GetFlagArgValAndSplit(const char* flag, const char* arg, std::vector<std::string>* vals);
static bool GetFlagArgValAndSplitNested(const char* flag, const char* arg, std::vector<std::vector<std::string>>* vals);
static void Usage();
static int32_t StringToFourcc(const std::string& str);
static int ParseMyArgs(int argc, char** argv);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Class definitions                                                                                                ///
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class App {
 public:
  App();
  ~App();

  NvCV_Status Initialize();
  NvCV_Status OpenInputVideo();
  NvCV_Status OpenInputAudio();
  NvCV_Status OpenOutputVideo();
  NvCV_Status Run();
  NvCV_Status CloseInputVideo();
  NvCV_Status CloseOutputVideo();

 private:
  NvCV_Status InitTriton();
  NvCV_Status Load();
  NvCV_Status InitStream(uint32_t n);
  NvCV_Status ReleaseStream(uint32_t n);
  NvCV_Status AllocateBuffers();
  NvCV_Status SetParameters();

  // Triton connection
  std::string m_effectName;
  NvAR_TritonServer m_triton;
  NvAR_FeatureHandle m_effect;
  CUstream m_cudaStream;

  // Stream management
  uint32_t m_numVideoStreams;
  std::vector<NvAR_StateHandle> m_arrayOfAllStateObjects;
  std::vector<NvAR_StateHandle> m_batchOfStateObjects;

  // Video I/O
  std::vector<cv::VideoCapture> m_videoCaptures;
  std::vector<cv::VideoWriter> m_videoWriters;
  uint32_t m_srcVideoWidth;
  uint32_t m_srcVideoHeight;
  float m_videoFps;

  // Image buffers
  NvCVImage m_srcVid;
  NvCVImage m_tmpImg;
  NvCVImage m_nvTempResult;
  NvCVImage m_inputImgView;

  // Audio data - per video, per track
  std::vector<std::vector<std::vector<float>*>> m_audioSamplesPerVideo;  // [video][track] -> samples
  std::vector<uint32_t> m_numAudioStreamsPerVideo;
  uint32_t m_maxAudioStreamsPerVideo;
  uint32_t m_sampleRate;

  // Batched input audio frame data
  std::vector<std::vector<NvAR_AudioFrame>> m_inputAudioFramesBatched;
  std::vector<NvAR_AudioFrameData> m_inputAudioFrameDataBatched;
  std::vector<std::vector<float>> m_audioFrameDataBuffers;

  // Batched active audio IDs
  std::vector<NvAR_ActiveAudioIds> m_activeAudioIdsBatched;
  std::vector<std::vector<uint32_t>> m_activeAudioIdsArrays;

  // Batched output tracking data
  std::vector<std::vector<NvAR_SpeakerTrackingBBox>> m_outputBoxesPerStream;
  std::vector<NvAR_ActiveSpeakerTrackingData> m_outputTrackingDataBatched;
  uint32_t m_maxNumOutputIdentities;

  // Batched input/output parameters (indexed by video stream index)
  std::vector<uint32_t> m_newShotBatched;  // Input: shot change mode per video stream
  std::vector<uint32_t> m_flushBatched;    // Input: flush mode per video stream
  std::vector<uint32_t> m_readyBatched;    // Output: ready status per video stream

  // Batch-position-indexed arrays for remapping (indexed by batch position)
  // These get populated before each NvAR_Run call with data from the active video streams
  std::vector<NvAR_AudioFrameData> m_batchAudioFrameData;                 // Audio data at batch positions
  std::vector<NvAR_ActiveAudioIds> m_batchActiveAudioIds;                 // Active audio IDs at batch positions
  std::vector<std::vector<uint32_t>> m_batchActiveAudioIdsArrays;         // Storage for active audio ID arrays
  std::vector<NvAR_ActiveSpeakerTrackingData> m_batchOutputTrackingData;  // Output at batch positions
  std::vector<std::vector<NvAR_SpeakerTrackingBBox>> m_batchOutputBoxes;  // Output boxes at batch positions
  std::vector<uint32_t> m_batchNewShot;                                   // New shot flags at batch positions
  std::vector<uint32_t> m_batchFlush;                                     // Flush flags at batch positions
  std::vector<uint32_t> m_batchReady;                                     // Ready flags at batch positions

  // Frame caching for output synchronization (per video stream)
  std::vector<std::deque<cv::Mat>> m_frameCachePerVideo;
  std::vector<std::deque<NvAR_ActiveSpeakerTrackingData>> m_outputCachePerVideo;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Static function definitions                                                                                        //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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

static bool GetFlagArgVal(const char* flag, const char* arg, int64_t* val) {
  const char* valStr;
  bool success = GetFlagArgVal(flag, arg, &valStr);
  if (success) *val = strtol(valStr, NULL, 10);
  return success;
}

static bool GetFlagArgVal(const char* flag, const char* arg, uint32_t* val) {
  int64_t longVal;
  bool success = GetFlagArgVal(flag, arg, &longVal);
  if (success) {
    *val = static_cast<uint32_t>(longVal);
  }
  return success;
}

static bool GetFlagArgVal(const char* flag, const char* arg, float* val) {
  const char* valStr;
  bool success = GetFlagArgVal(flag, arg, &valStr);
  if (success) *val = static_cast<float>(atof(valStr));
  return success;
}

static bool GetFlagArgValAndSplit(const char* flag, const char* arg, std::vector<std::string>* vals) {
  const char* valStr;
  if (!GetFlagArgVal(flag, arg, &valStr)) return false;

  if (valStr) {
    std::string value(valStr);
    std::istringstream iss(value);
    std::string part;
    while (std::getline(iss, part, ',')) {
      if (!part.empty()) {
        vals->push_back(part);
      }
    }
  }
  return true;
}

// Parse audio files with comma separation for videos and '+' separation for tracks
// Format: "v1_track1+v1_track2,v2_track1+v2_track2+v2_track3"
static bool GetFlagArgValAndSplitNested(const char* flag, const char* arg,
                                        std::vector<std::vector<std::string>>* vals) {
  const char* valStr;
  if (!GetFlagArgVal(flag, arg, &valStr)) return false;

  if (valStr) {
    std::string value(valStr);
    std::istringstream video_stream(value);
    std::string video_group;
    while (std::getline(video_stream, video_group, ',')) {
      if (video_group.empty()) continue;
      std::vector<std::string> audio_tracks;
      std::istringstream track_stream(video_group);
      std::string track;
      while (std::getline(track_stream, track, '+')) {
        if (!track.empty()) {
          audio_tracks.push_back(track);
        }
      }
      if (!audio_tracks.empty()) {
        vals->push_back(audio_tracks);
      }
    }
  }
  return true;
}

static void Usage() {
  printf(
      "ActiveSpeakerDetectionTritonClientApp [flags ...]\n"
      "  where flags are:\n"
      "  --verbose[=(true|false)]           Print verbose information (default: false)\n"
      "  --url=<URL>                        URL to the Triton server\n"
      "  --grpc[=(true|false)]              Use gRPC for data transfer to the Triton server instead of CUDA shared "
      "memory (default: false)\n"
      "  --output_name_tag=<string>         A string appended to each input video file to create the corresponding "
      "output file name (default: \"output\")\n"
      "  --output_codec=<fourcc>            FourCC code for the desired codec (default: \"mp4v\" -- MPEG-4)\n"
      "  --output_format=<format>           Format of the output video (default: \"mp4\")\n"
      "  --log=<file>                       Log SDK errors to a file, \"stderr\" or \"\" (default: stderr)\n"
      "  --log_level=<N>                    The desired log level: {0, 1, 2, 3} = {FATAL, ERROR, WARNING, INFO} "
      "(default: 1)\n"
      "  --src_videos=<v0,v1,...>           Comma separated list of identically sized source video files\n"
      "  --src_audios=<v0_a0+v0_a1,v1_a0,...>  Audio files per video. Comma separates videos, '+' separates\n"
      "                                        audio tracks within each video.\n"
      "                                        Example: \"audio0_0.wav+audio0_1.wav,audio1_0.wav\"\n"
      "                                        means video0 has 2 audio tracks, video1 has 1 audio track.\n"
      "  --sync_tolerance=<f>               Min sync score [0, 1] to consider speaking (-1 = unset, use SDK default)\n"
      "  --help                             Print out this message\n");
}

static int32_t StringToFourcc(const std::string& str) {
  union chint {
    int32_t i;
    char c[4];
  };
  chint x = {0};
  for (int32_t n = (str.size() < 4) ? static_cast<int32_t>(str.size()) : 4; n--;) x.c[n] = str[n];
  return x.i;
}

static int ParseMyArgs(int argc, char** argv) {
  int errs = 0;
  for (--argc, ++argv; argc--; ++argv) {
    bool help;
    const char* arg = *argv;
    if (arg[0] == '-') {
      if (arg[1] == '-') {                                                 // double-dash
        if (GetFlagArgVal("verbose", arg, &FLAG_verbose) ||                //
            GetFlagArgVal("url", arg, &FLAG_tritonURL) ||                  //
            GetFlagArgVal("grpc", arg, &FLAG_useTritonGRPC) ||             //
            GetFlagArgVal("output_name_tag", arg, &FLAG_outputNameTag) ||  //
            GetFlagArgVal("log", arg, &FLAG_log) ||                        //
            GetFlagArgVal("output_codec", arg, &FLAG_outputCodec) ||       //
            GetFlagArgVal("output_format", arg, &FLAG_outputFormat) ||     //
            GetFlagArgVal("log_level", arg, &FLAG_logLevel) ||
            GetFlagArgVal("sync_tolerance", arg, &FLAG_syncTolerance)) {
          continue;
        } else if (GetFlagArgVal("help", arg, &help)) {
          Usage();
          errs = 1;
        } else if (GetFlagArgValAndSplit("src_videos", arg, &FLAG_srcVideoFiles)) {
          continue;
        } else if (GetFlagArgValAndSplitNested("src_audios", arg, &FLAG_srcAudioFilesPerVideo)) {
          continue;
        }
      } else {  // single dash
        for (++arg; *arg; ++arg) {
          if (*arg == 'v') {
            FLAG_verbose = true;
          } else {
            printf("Unknown flag: \"-%c\"\n", *arg);
          }
        }
      }
    }
  }
  return errs;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Member function definitions                                                                                        //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

App::App()
    : m_triton(nullptr),
      m_effect(nullptr),
      m_cudaStream(0),
      m_numVideoStreams(0),
      m_srcVideoWidth(0),
      m_srcVideoHeight(0),
      m_videoFps(30.0f),
      m_maxAudioStreamsPerVideo(1),
      m_sampleRate(44100),
      m_maxNumOutputIdentities(0) {
  m_effectName = NvAR_Feature_ActiveSpeakerDetection;
}

App::~App() {
  NvCVImage_Dealloc(&m_srcVid);
  NvCVImage_Dealloc(&m_nvTempResult);
  NvCVImage_Dealloc(&m_tmpImg);

  if (m_effect) {
    NvAR_Destroy(m_effect);
    m_effect = nullptr;
  }
  if (m_cudaStream) NvAR_CudaStreamDestroy(m_cudaStream);
  if (m_triton) NvAR_DisconnectTritonServer(m_triton);

  // Close video/audio resources
  CloseInputVideo();
  CloseOutputVideo();
}

NvCV_Status App::InitTriton() {
  NvCV_Status err = NVCV_SUCCESS;
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
  if (FLAG_verbose) {
    printf("Using triton server\n");
  }
bail:
  return err;
}
NvCV_Status App::Load() { return NvAR_Load(m_effect); }

NvCV_Status App::InitStream(uint32_t n) { return NvAR_AllocateState(m_effect, &m_arrayOfAllStateObjects[n]); }

NvCV_Status App::ReleaseStream(uint32_t n) { return NvAR_DeallocateState(m_effect, m_arrayOfAllStateObjects[n]); }

NvCV_Status App::AllocateBuffers() {
  NvCV_Status err = NVCV_SUCCESS;

  // Allocate batched image buffer
  BAIL_IF_ERR(err = AllocateBatchBuffer(&m_srcVid, m_numVideoStreams, m_srcVideoWidth, m_srcVideoHeight, NVCV_BGR,
                                        NVCV_U8, NVCV_CHUNKY, FLAG_useTritonGRPC ? NVCV_CPU : NVCV_CUDA, 1));

  // Get max output identities
  BAIL_IF_ERR(err = NvAR_GetU32(m_effect, NvAR_Parameter_Config(MaxNumOutputIdentities), &m_maxNumOutputIdentities));

  // Allocate output tracking data
  m_outputBoxesPerStream.resize(m_numVideoStreams);
  m_outputTrackingDataBatched.resize(m_numVideoStreams);
  for (uint32_t i = 0; i < m_numVideoStreams; ++i) {
    m_outputBoxesPerStream[i].resize(m_maxNumOutputIdentities);
    m_outputTrackingDataBatched[i].boxes = m_outputBoxesPerStream[i].data();
    m_outputTrackingDataBatched[i].max_boxes = static_cast<uint8_t>(m_maxNumOutputIdentities);
    m_outputTrackingDataBatched[i].num_boxes = 0;
  }

  // Allocate audio frame data buffers
  m_inputAudioFramesBatched.resize(m_numVideoStreams);
  m_inputAudioFrameDataBatched.resize(m_numVideoStreams);
  m_audioFrameDataBuffers.resize(m_numVideoStreams * m_maxAudioStreamsPerVideo);

  for (uint32_t v = 0; v < m_numVideoStreams; ++v) {
    uint32_t num_audio = m_numAudioStreamsPerVideo[v];
    m_inputAudioFramesBatched[v].resize(num_audio);

    for (uint32_t a = 0; a < num_audio; ++a) {
      uint32_t buffer_idx = v * m_maxAudioStreamsPerVideo + a;
      m_audioFrameDataBuffers[buffer_idx].resize(m_sampleRate + 1024, 0.0f);
      m_inputAudioFramesBatched[v][a].audio_data = m_audioFrameDataBuffers[buffer_idx].data();
      m_inputAudioFramesBatched[v][a].audio_id = a;
      m_inputAudioFramesBatched[v][a].num_samples = 0;
    }
    m_inputAudioFrameDataBatched[v].audio_frames = m_inputAudioFramesBatched[v].data();
    m_inputAudioFrameDataBatched[v].num_audio_channels = num_audio;
  }

  // Allocate active audio IDs
  m_activeAudioIdsBatched.resize(m_numVideoStreams);
  m_activeAudioIdsArrays.resize(m_numVideoStreams);
  for (uint32_t i = 0; i < m_numVideoStreams; ++i) {
    uint32_t num_audio = m_numAudioStreamsPerVideo[i];
    m_activeAudioIdsArrays[i].resize(num_audio);
    for (uint32_t j = 0; j < num_audio; ++j) {
      m_activeAudioIdsArrays[i][j] = j;
    }
    m_activeAudioIdsBatched[i].active_audio_ids = m_activeAudioIdsArrays[i].data();
    m_activeAudioIdsBatched[i].num_active_audio_ids = num_audio;
  }
  // Initialize state arrays
  m_arrayOfAllStateObjects.resize(m_numVideoStreams, nullptr);
  m_batchOfStateObjects.resize(m_numVideoStreams, nullptr);

  // Initialize batched parameters (indexed by video stream)
  m_newShotBatched.resize(m_numVideoStreams, NVARACTIVESPEAKERDETECTION_DETECT_SHOT_CHANGE);
  m_flushBatched.resize(m_numVideoStreams, 0);
  m_readyBatched.resize(m_numVideoStreams, 0);

  // Initialize batch-position-indexed arrays
  m_batchAudioFrameData.resize(m_numVideoStreams);
  m_batchActiveAudioIds.resize(m_numVideoStreams);
  m_batchActiveAudioIdsArrays.resize(m_numVideoStreams);
  m_batchOutputTrackingData.resize(m_numVideoStreams);
  m_batchOutputBoxes.resize(m_numVideoStreams);
  m_batchNewShot.resize(m_numVideoStreams);
  m_batchFlush.resize(m_numVideoStreams);
  m_batchReady.resize(m_numVideoStreams);
  for (uint32_t i = 0; i < m_numVideoStreams; ++i) {
    m_batchActiveAudioIdsArrays[i].resize(m_maxAudioStreamsPerVideo);
    m_batchActiveAudioIds[i].active_audio_ids = m_batchActiveAudioIdsArrays[i].data();
    m_batchActiveAudioIds[i].num_active_audio_ids = 0;
    m_batchOutputBoxes[i].resize(m_maxNumOutputIdentities);
    m_batchOutputTrackingData[i].boxes = m_batchOutputBoxes[i].data();
    m_batchOutputTrackingData[i].max_boxes = m_maxNumOutputIdentities;
    m_batchOutputTrackingData[i].num_boxes = 0;
  }

  // Initialize frame caches (one per video stream)
  m_frameCachePerVideo.resize(m_numVideoStreams);
  m_outputCachePerVideo.resize(m_numVideoStreams);

bail:
  return err;
}

NvCV_Status App::SetParameters() {
  NvCV_Status err = NVCV_SUCCESS;

  // Set batch size first
  BAIL_IF_ERR(err = NvAR_SetU32(m_effect, NvAR_Parameter_Config(BatchSize), m_numVideoStreams));

  // Set config parameters
  BAIL_IF_ERR(err = NvAR_SetU32(m_effect, NvAR_Parameter_Config(NumAudioStreams), m_maxAudioStreamsPerVideo));
  BAIL_IF_ERR(err = NvAR_SetU32(m_effect, NvAR_Parameter_Config(SampleRate), m_sampleRate));
  BAIL_IF_ERR(err = NvAR_SetF32(m_effect, NvAR_Parameter_Config(VideoFPS), m_videoFps));
  if (FLAG_syncTolerance >= 0.0f) {
    BAIL_IF_ERR(err = NvAR_SetF32(m_effect, NvAR_Parameter_Config(SyncTolerance), FLAG_syncTolerance));
  } else if (FLAG_verbose) {
    printf("Sync tolerance not set; using SDK default.\n");
  }

  // Set input image (first image in batch, pixels pointer gives access to full buffer)
  BAIL_IF_ERR(err = NvAR_SetObject(m_effect, NvAR_Parameter_Input(Image),
                                   NthImage(0, m_srcVid.height / m_numVideoStreams, &m_srcVid, &m_inputImgView),
                                   sizeof(NvCVImage)));

  // Set batch-position-indexed arrays (these may get remapped before each Run call)
  BAIL_IF_ERR(err = NvAR_SetObject(m_effect, NvAR_Parameter_Input(AudioFrameData), m_batchAudioFrameData.data(),
                                   sizeof(NvAR_AudioFrameData)));
  BAIL_IF_ERR(err = NvAR_SetObject(m_effect, NvAR_Parameter_Input(ActiveAudioIDs), m_batchActiveAudioIds.data(),
                                   sizeof(NvAR_ActiveAudioIds)));
  BAIL_IF_ERR(err = NvAR_SetObject(m_effect, NvAR_Parameter_Output(ActiveSpeakerTrackingData),
                                   m_batchOutputTrackingData.data(), sizeof(NvAR_ActiveSpeakerTrackingData)));
  BAIL_IF_ERR(err = NvAR_SetU32Array(m_effect, NvAR_Parameter_Input(NewShot), m_batchNewShot.data(),
                                     static_cast<int32_t>(m_numVideoStreams)));
  BAIL_IF_ERR(err = NvAR_SetU32Array(m_effect, NvAR_Parameter_Input(Flush), m_batchFlush.data(),
                                     static_cast<int32_t>(m_numVideoStreams)));
  BAIL_IF_ERR(err = NvAR_SetU32Array(m_effect, NvAR_Parameter_Output(Ready), m_batchReady.data(),
                                     static_cast<int32_t>(m_numVideoStreams)));

bail:
  return err;
}

NvCV_Status App::Initialize() {
  NvCV_Status err = NVCV_SUCCESS;
  BAIL_IF_ERR(err = InitTriton());
  BAIL_IF_ERR(err = AllocateBuffers());
  BAIL_IF_ERR(err = SetParameters());
  BAIL_IF_ERR(err = Load());

  if (FLAG_verbose) {
    printf("Number of video streams: %u\n", m_numVideoStreams);
  }

bail:
  return err;
}

NvCV_Status App::OpenInputVideo() {
  NvCV_Status err = NVCV_SUCCESS;

  m_numVideoStreams = static_cast<uint32_t>(FLAG_srcVideoFiles.size());
  BAIL_IF_FALSE(m_numVideoStreams > 0, err, NVCV_ERR_MISSINGINPUT);
  BAIL_IF_FALSE(FLAG_srcAudioFilesPerVideo.size() == m_numVideoStreams, err, NVCV_ERR_MISMATCH);
  BAIL_IF_FALSE(FLAG_outputFormat == "mp4" || FLAG_outputFormat == "avi", err, NVCV_ERR_PARAMETER);

  m_videoCaptures.resize(m_numVideoStreams);

  for (uint32_t i = 0; i < m_numVideoStreams; i++) {
    m_videoCaptures[i].open(FLAG_srcVideoFiles[i], cv::CAP_FFMPEG);
    if (!m_videoCaptures[i].isOpened()) {
      printf("Error: Could not open %s.\n", FLAG_srcVideoFiles[i].c_str());
      BAIL(err, NVCV_ERR_READ);
    }

    uint32_t width = static_cast<uint32_t>(m_videoCaptures[i].get(cv::CAP_PROP_FRAME_WIDTH));
    uint32_t height = static_cast<uint32_t>(m_videoCaptures[i].get(cv::CAP_PROP_FRAME_HEIGHT));

    if (width == 0 || height == 0) {
      printf("Error: Could not retrieve resolution for %s.\n", FLAG_srcVideoFiles[i].c_str());
      BAIL(err, NVCV_ERR_READ);
    }

    if (i == 0) {
      m_srcVideoWidth = width;
      m_srcVideoHeight = height;
      m_videoFps = static_cast<float>(m_videoCaptures[i].get(cv::CAP_PROP_FPS));
    } else if (m_srcVideoWidth != width || m_srcVideoHeight != height) {
      printf("Error: Resolution of all videos must be the same.\n");
      BAIL(err, NVCV_ERR_MISMATCH);
    }

    m_videoCaptures[i].set(cv::CAP_PROP_POS_FRAMES, 0);

    if (FLAG_verbose) {
      printf("Opened video %u: %s (%ux%u @ %.2f fps)\n", i, FLAG_srcVideoFiles[i].c_str(), width, height, m_videoFps);
    }
  }

bail:
  return err;
}

NvCV_Status App::OpenInputAudio() {
  NvCV_Status err = NVCV_SUCCESS;

  m_numAudioStreamsPerVideo.resize(m_numVideoStreams);
  m_audioSamplesPerVideo.resize(m_numVideoStreams);
  m_maxAudioStreamsPerVideo = 0;

  for (uint32_t v = 0; v < m_numVideoStreams; v++) {
    uint32_t num_audio_tracks = static_cast<uint32_t>(FLAG_srcAudioFilesPerVideo[v].size());
    m_numAudioStreamsPerVideo[v] = num_audio_tracks;
    m_audioSamplesPerVideo[v].resize(num_audio_tracks, nullptr);

    if (num_audio_tracks > m_maxAudioStreamsPerVideo) {
      m_maxAudioStreamsPerVideo = num_audio_tracks;
    }
    if (FLAG_verbose) {
      printf("Video %u: %u audio tracks\n", v, num_audio_tracks);
    }

    for (uint32_t a = 0; a < num_audio_tracks; a++) {
      const std::string& audio_file = FLAG_srcAudioFilesPerVideo[v][a];
      uint32_t input_num_samples = 0;

      CWaveFileRead wave_reader(audio_file);
      if (!wave_reader.isValid()) {
        printf("Error: Audio file \"%s\" could not be opened\n", audio_file.c_str());
        BAIL(err, NVCV_ERR_READ);
      }

      if (v == 0 && a == 0) {
        m_sampleRate = wave_reader.GetSampleRate();
      }
      if (!ReadWavFile(audio_file, m_sampleRate, 1, &m_audioSamplesPerVideo[v][a], &input_num_samples, nullptr, -1,
                       FLAG_verbose)) {
        printf("Unable to read wav file: %s\n", audio_file.c_str());
        BAIL(err, NVCV_ERR_READ);
      }
      if (FLAG_verbose) {
        printf("  Audio %u.%u: %s (%u samples)\n", v, a, audio_file.c_str(), input_num_samples);
      }
    }
  }

bail:
  return err;
}

NvCV_Status App::OpenOutputVideo() {
  NvCV_Status err = NVCV_SUCCESS;

  m_videoWriters.resize(m_numVideoStreams);

  for (uint32_t i = 0; i < m_numVideoStreams; i++) {
    size_t period_loc = FLAG_srcVideoFiles[i].find_last_of(".");
    std::string dst_video = FLAG_srcVideoFiles[i].substr(0, period_loc);
    dst_video = dst_video + "_" + FLAG_outputNameTag + "." + FLAG_outputFormat;

    m_videoWriters[i].open(dst_video, StringToFourcc(FLAG_outputCodec), m_videoFps,
                           cv::Size(m_srcVideoWidth, m_srcVideoHeight));
    if (!m_videoWriters[i].isOpened()) {
      printf("Error: Could not open video writer for %s.\n", dst_video.c_str());
      BAIL(err, NVCV_ERR_WRITE);
    }

    if (FLAG_verbose) {
      printf("Output video %u: %s\n", i, dst_video.c_str());
    }
  }

bail:
  return err;
}

NvCV_Status App::Run() {
  NvCV_Status err = NVCV_SUCCESS;

  std::vector<uint32_t> batch_indices(m_numVideoStreams);
  std::vector<double> frame_timestamp(m_numVideoStreams, 0.0);
  std::vector<bool> video_finished(m_numVideoStreams, false);
  std::vector<bool> stream_initialized(m_numVideoStreams, false);
  std::vector<std::vector<uint32_t>> last_audio_end_samples(m_numVideoStreams);
  std::vector<std::vector<bool>> audio_finished(m_numVideoStreams);

  std::vector<cv::Mat> frames(m_numVideoStreams);
  std::vector<bool> got_first_frame(m_numVideoStreams, false);
  NvCVImage nv_img, tmp_img;
  uint32_t frame_count = 0;
  uint32_t batch_size_to_process = 0;
  std::vector<uint32_t> output_frame_count_per_video(m_numVideoStreams, 0);
  bool is_flushing = false;

  // Initialize per-video audio tracking
  for (uint32_t v = 0; v < m_numVideoStreams; v++) {
    last_audio_end_samples[v].resize(m_numAudioStreamsPerVideo[v], 0);
    audio_finished[v].resize(m_numAudioStreamsPerVideo[v], false);
  }

  // Sometimes the first video frame is not read, so we need to retry it in that case.
  for (uint32_t v = 0; v < m_numVideoStreams; v++) {
    if (m_videoCaptures[v].isOpened()) {
      got_first_frame[v] = m_videoCaptures[v].read(frames[v]);
    }
  }

  // Main processing loop
  while (true) {
    batch_size_to_process = 0;

    for (uint32_t v = 0; v < m_numVideoStreams; v++) {
      if (video_finished[v]) continue;

      // Read frame
      if (!m_videoCaptures[v].isOpened()) {
        video_finished[v] = true;
        continue;
      }
      bool got_frame = got_first_frame[v] || m_videoCaptures[v].read(frames[v]);
      got_first_frame[v] = false;  // Reset the flag after the first time
      if (!got_frame || frames[v].empty()) {
        if (FLAG_verbose) {
          printf("Video stream %u ending at frame %u\n", v, frame_count);
        }
        video_finished[v] = true;
        continue;
      }

      // Initialize stream state on first valid frame
      if (!stream_initialized[v]) {
        BAIL_IF_ERR(err = InitStream(v));
        stream_initialized[v] = true;
      }

      frame_timestamp[v] += 1.0 / static_cast<double>(m_videoFps);

      // Cache the input frame
      m_frameCachePerVideo[v].push_back(frames[v].clone());

      // Transfer frame to batch buffer
      NVWrapperForCVMat(&frames[v], &nv_img);
      BAIL_IF_ERR(err = TransferToNthImage(batch_size_to_process, &nv_img, &m_srcVid, 1, m_cudaStream, &tmp_img));

      // Prepare audio for all tracks of this video
      // Update active audio IDs based on which tracks still have data
      m_activeAudioIdsArrays[v].clear();
      for (uint32_t a = 0; a < m_numAudioStreamsPerVideo[v]; a++) {
        uint32_t audio_start_sample = last_audio_end_samples[v][a];
        uint32_t requested_audio_end_sample =
            static_cast<uint32_t>(frame_timestamp[v] * static_cast<double>(m_sampleRate));
        uint32_t audio_end_sample =
            std::min(requested_audio_end_sample, static_cast<uint32_t>(m_audioSamplesPerVideo[v][a]->size()));
        last_audio_end_samples[v][a] = requested_audio_end_sample;

        uint32_t audio_frame_length = requested_audio_end_sample - audio_start_sample;
        m_inputAudioFramesBatched[v][a].num_samples = audio_frame_length;

        // Clear and copy audio samples
        uint32_t buffer_idx = v * m_maxAudioStreamsPerVideo + a;
        std::fill(m_audioFrameDataBuffers[buffer_idx].begin(), m_audioFrameDataBuffers[buffer_idx].end(), 0.0f);

        if (!audio_finished[v][a]) {
          size_t valid_start = std::min<size_t>(audio_start_sample, m_audioSamplesPerVideo[v][a]->size());
          size_t valid_end = std::min<size_t>(audio_end_sample, m_audioSamplesPerVideo[v][a]->size());

          if (valid_start < valid_end) {
            std::copy(m_audioSamplesPerVideo[v][a]->begin() + valid_start,
                      m_audioSamplesPerVideo[v][a]->begin() + valid_end, m_audioFrameDataBuffers[buffer_idx].begin());
          }

          // Only add to active audio IDs if audio still has data
          m_activeAudioIdsArrays[v].push_back(a);

          if (requested_audio_end_sample >= m_audioSamplesPerVideo[v][a]->size()) {
            if (FLAG_verbose) {
              printf("Audio stream %u.%u ending at frame %u\n", v, a, frame_count);
            }
            audio_finished[v][a] = true;
          }
        }
      }
      // Update the active audio IDs count for this video
      m_activeAudioIdsBatched[v].num_active_audio_ids = static_cast<uint32_t>(m_activeAudioIdsArrays[v].size());

      batch_indices[batch_size_to_process] = v;
      batch_size_to_process++;
    }

    // Check if we need to start flushing
    if (batch_size_to_process == 0) {
      // Check if any video still has cached frames to flush
      bool any_frames_to_flush = false;
      for (uint32_t v = 0; v < m_numVideoStreams; v++) {
        if (!m_frameCachePerVideo[v].empty()) {
          any_frames_to_flush = true;
          break;
        }
      }
      if (!any_frames_to_flush) break;

      // Flushing mode: add all videos with cached frames to batch
      if (!is_flushing && FLAG_verbose) {
        printf("End of input. Flushing remaining frames...\n");
      }
      is_flushing = true;
      for (uint32_t v = 0; v < m_numVideoStreams; v++) {
        if (!m_frameCachePerVideo[v].empty() && stream_initialized[v]) {
          // Set flush flag for this video
          m_flushBatched[v] = 1;

          // Prepare zero-filled audio for flushing
          for (uint32_t a = 0; a < m_numAudioStreamsPerVideo[v]; a++) {
            uint32_t buffer_idx = v * m_maxAudioStreamsPerVideo + a;
            std::fill(m_audioFrameDataBuffers[buffer_idx].begin(), m_audioFrameDataBuffers[buffer_idx].end(), 0.0f);
            m_inputAudioFramesBatched[v][a].num_samples =
                static_cast<uint32_t>(m_sampleRate / m_videoFps);  // One frame worth
          }

          batch_indices[batch_size_to_process] = v;
          batch_size_to_process++;
        }
      }
      if (batch_size_to_process == 0) break;
    }

    if (FLAG_verbose) {
      printf("Frame %u, batch size: %u\n", frame_count, batch_size_to_process);
    }

    // Remap inputs from video indices to batch positions
    // This is critical: the Triton client reads data at batch positions 0, 1, 2...
    // but our per-video arrays are indexed by video stream ID
    for (uint32_t i = 0; i < batch_size_to_process; i++) {
      uint32_t video_idx = batch_indices[i];

      // Copy audio frame data from video index to batch position
      m_batchAudioFrameData[i] = m_inputAudioFrameDataBatched[video_idx];

      // Copy active audio IDs from video index to batch position
      uint32_t num_active = m_activeAudioIdsBatched[video_idx].num_active_audio_ids;
      for (uint32_t j = 0; j < num_active && j < m_maxAudioStreamsPerVideo; ++j) {
        m_batchActiveAudioIdsArrays[i][j] = m_activeAudioIdsBatched[video_idx].active_audio_ids[j];
      }
      m_batchActiveAudioIds[i].active_audio_ids = m_batchActiveAudioIdsArrays[i].data();
      m_batchActiveAudioIds[i].num_active_audio_ids = num_active;

      // Copy input parameters from video index to batch position
      m_batchNewShot[i] = m_newShotBatched[video_idx];
      m_batchFlush[i] = m_flushBatched[video_idx];

      // Set up state objects
      m_batchOfStateObjects[i] = m_arrayOfAllStateObjects[video_idx];
    }

    BAIL_IF_ERR(err = NvAR_SetU32(m_effect, NvAR_Parameter_Config(BatchSize), batch_size_to_process));
    BAIL_IF_ERR(err = NvAR_SetObject(m_effect, NvAR_Parameter_InOut(State), m_batchOfStateObjects.data(),
                                     batch_size_to_process));

    // Re-set input/output pointers after batch size change to update internal state
    BAIL_IF_ERR(err = NvAR_SetObject(m_effect, NvAR_Parameter_Input(Image),
                                     NthImage(0, m_srcVid.height / m_numVideoStreams, &m_srcVid, &m_inputImgView),
                                     sizeof(NvCVImage)));
    BAIL_IF_ERR(err = NvAR_SetObject(m_effect, NvAR_Parameter_Input(AudioFrameData), m_batchAudioFrameData.data(),
                                     sizeof(NvAR_AudioFrameData)));
    BAIL_IF_ERR(err = NvAR_SetObject(m_effect, NvAR_Parameter_Input(ActiveAudioIDs), m_batchActiveAudioIds.data(),
                                     sizeof(NvAR_ActiveAudioIds)));
    BAIL_IF_ERR(err = NvAR_SetObject(m_effect, NvAR_Parameter_Output(ActiveSpeakerTrackingData),
                                     m_batchOutputTrackingData.data(), sizeof(NvAR_ActiveSpeakerTrackingData)));
    BAIL_IF_ERR(err = NvAR_SetU32Array(m_effect, NvAR_Parameter_Input(NewShot), m_batchNewShot.data(),
                                       static_cast<int32_t>(batch_size_to_process)));
    BAIL_IF_ERR(err = NvAR_SetU32Array(m_effect, NvAR_Parameter_Input(Flush), m_batchFlush.data(),
                                       static_cast<int32_t>(batch_size_to_process)));
    BAIL_IF_ERR(err = NvAR_SetU32Array(m_effect, NvAR_Parameter_Output(Ready), m_batchReady.data(),
                                       static_cast<int32_t>(batch_size_to_process)));

    // Triton requires NvAR_DeallocateState() - Called in ReleaseStream() - to be called just before the last
    // inference (NvAR_Run()) for that video stream to indicate that the sequence is complete.
    for (uint32_t i = 0; i < batch_size_to_process; i++) {
      uint32_t video_idx = batch_indices[i];
      if (is_flushing && m_frameCachePerVideo[video_idx].size() == 1) {
        ReleaseStream(video_idx);
      }
    }

    BAIL_IF_ERR(err = NvAR_Run(m_effect));
    BAIL_IF_ERR(err = NvAR_SynchronizeTriton(m_effect));

    // Cache output for each video in batch if ready
    // Read from batch positions (i) and store to video indices (video_idx)
    for (uint32_t i = 0; i < batch_size_to_process; i++) {
      uint32_t video_idx = batch_indices[i];

      // Read ready status from batch position, not video index
      if (m_batchReady[i] != 0) {
        // Read output from batch position
        NvAR_ActiveSpeakerTrackingData output_copy = m_batchOutputTrackingData[i];
        output_copy.boxes = new NvAR_SpeakerTrackingBBox[output_copy.num_boxes];
        std::copy(m_batchOutputBoxes[i].begin(), m_batchOutputBoxes[i].begin() + output_copy.num_boxes,
                  output_copy.boxes);
        m_outputCachePerVideo[video_idx].push_back(output_copy);
      }
    }

    // Write outputs when both frame and output are available (per video)
    for (uint32_t v = 0; v < m_numVideoStreams; v++) {
      while (!m_frameCachePerVideo[v].empty() && !m_outputCachePerVideo[v].empty()) {
        cv::Mat& cached_frame = m_frameCachePerVideo[v].front();
        NvAR_ActiveSpeakerTrackingData& cached_output = m_outputCachePerVideo[v].front();

        // Generate visualization
        cv::Mat display_frame = cached_frame.clone();

        // Scale font for high-resolution videos (1440p and above)
        const bool high_res = display_frame.rows >= 1440;
        const double font_scale = high_res ? 1.4 : 0.7;
        const int font_thickness = high_res ? 4 : 2;
        const int box_thickness = high_res ? 6 : 3;
        const int label_padding = high_res ? 20 : 10;
        const int label_offset = high_res ? 10 : 5;

        // Draw frame counter at top left
        std::string frame_text = cv::format("Frame: %d", static_cast<int>(output_frame_count_per_video[v]));
        const int frame_y = high_res ? 60 : 30;
        cv::putText(display_frame, frame_text, cv::Point(10, frame_y), cv::FONT_HERSHEY_SIMPLEX, font_scale,
                    cv::Scalar(0, 255, 0), font_thickness, cv::LINE_AA);

        // Draw bounding boxes with three-color scheme
        for (uint32_t i = 0; i < cached_output.num_boxes; ++i) {
          const NvAR_SpeakerTrackingBBox& bbox = cached_output.boxes[i];

          // Color scheme: Red=tracked, Blue=audio assigned, Green=speaking
          cv::Scalar color = cv::Scalar(0, 0, 255);  // Red: tracked, no audio assigned
          if (bbox.audio_id != -1) {
            color = cv::Scalar(255, 0, 0);  // Blue: audio assigned (previous speaker)
          }
          if (bbox.is_speaking) {
            color = cv::Scalar(0, 255, 0);  // Green: currently speaking
          }

          // Text color: black for active speaker (green), white otherwise
          cv::Scalar text_color = bbox.is_speaking ? cv::Scalar(0, 0, 0) : cv::Scalar(255, 255, 255);

          // Draw bounding box
          cv::Rect rect(static_cast<int32_t>(bbox.bbox.x), static_cast<int32_t>(bbox.bbox.y),
                        static_cast<int32_t>(bbox.bbox.width), static_cast<int32_t>(bbox.bbox.height));
          cv::rectangle(display_frame, rect, color, box_thickness);

          // Draw track label with background at top-left
          std::string label_text = cv::format("Track:%d (%.2f)", bbox.tracking_id, bbox.confidence);
          int baseline = 0;
          cv::Size text_size =
              cv::getTextSize(label_text, cv::FONT_HERSHEY_SIMPLEX, font_scale, font_thickness, &baseline);
          int box_x = static_cast<int32_t>(bbox.bbox.x);
          int box_y = static_cast<int32_t>(bbox.bbox.y);
          cv::rectangle(display_frame, cv::Point(box_x, box_y - text_size.height - label_padding),
                        cv::Point(box_x + text_size.width, box_y), color, -1);
          cv::putText(display_frame, label_text, cv::Point(box_x, box_y - label_offset), cv::FONT_HERSHEY_SIMPLEX,
                      font_scale, text_color, font_thickness);

          // Draw audio ID at bottom-right corner if assigned
          if (bbox.audio_id != -1) {
            std::string audio_label = cv::format("Audio:%d", bbox.audio_id);
            cv::Size audio_text_size =
                cv::getTextSize(audio_label, cv::FONT_HERSHEY_SIMPLEX, font_scale, font_thickness, &baseline);
            int audio_x = static_cast<int32_t>(bbox.bbox.x + bbox.bbox.width) - audio_text_size.width;
            int audio_y = static_cast<int32_t>(bbox.bbox.y + bbox.bbox.height);
            cv::rectangle(display_frame, cv::Point(audio_x, audio_y),
                          cv::Point(audio_x + audio_text_size.width, audio_y + audio_text_size.height + label_padding),
                          color, -1);
            cv::putText(display_frame, audio_label, cv::Point(audio_x, audio_y + audio_text_size.height + label_offset),
                        cv::FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness);
          }
        }

        if (!display_frame.empty()) {
          m_videoWriters[v] << display_frame;
        }

        delete[] cached_output.boxes;
        m_frameCachePerVideo[v].pop_front();
        m_outputCachePerVideo[v].pop_front();
        output_frame_count_per_video[v]++;
      }
    }

    // Increment frame count only when processing actual input frames
    if (!is_flushing) {
      frame_count++;
    }
  }

  // Clean up any remaining cached outputs
  for (uint32_t v = 0; v < m_numVideoStreams; v++) {
    for (auto& cached_output : m_outputCachePerVideo[v]) {
      delete[] cached_output.boxes;
    }
    m_outputCachePerVideo[v].clear();
  }

  printf("Processing complete! Total frames: %u\n", frame_count);

bail:
  return err;
}

NvCV_Status App::CloseOutputVideo() {
  for (auto& writer : m_videoWriters) {
    writer.release();
  }
  m_videoWriters.clear();
  return NVCV_SUCCESS;
}

NvCV_Status App::CloseInputVideo() {
  for (auto& capture : m_videoCaptures) {
    capture.release();
  }
  m_videoCaptures.clear();
  return NVCV_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Main                                                                                                               //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {
  int num_errs;
  NvCV_Status nv_err = NVCV_SUCCESS;
  App app;

  num_errs = ParseMyArgs(argc, argv);
  if (num_errs) return num_errs;

  // Validate required inputs
  if (FLAG_srcVideoFiles.empty() || FLAG_srcAudioFilesPerVideo.empty()) {
    printf("ERROR: --src_videos and --src_audios are required\n");
    Usage();
    return 1;
  }

  if (FLAG_srcAudioFilesPerVideo.size() != FLAG_srcVideoFiles.size()) {
    printf("ERROR: Number of audio groups (%zu) must match number of videos (%zu)\n", FLAG_srcAudioFilesPerVideo.size(),
           FLAG_srcVideoFiles.size());
    printf("Use ',' to separate audio groups for each video, '+' for tracks within a video.\n");
    printf("Example: --src_audios=\"audio0_0.wav+audio0_1.wav,audio1_0.wav\"\n");
    return 1;
  }

  nv_err = NvAR_ConfigureLogger(FLAG_logLevel, FLAG_log.c_str(), nullptr, nullptr);
  if (NVCV_SUCCESS != nv_err)
    printf("%s: while configuring logger to \"%s\"\n", NvCV_GetErrorStringFromCode(nv_err), FLAG_log.c_str());

  BAIL_IF_ERR(nv_err = app.OpenInputVideo());
  BAIL_IF_ERR(nv_err = app.OpenInputAudio());
  BAIL_IF_ERR(nv_err = app.Initialize());
  BAIL_IF_ERR(nv_err = app.OpenOutputVideo());
  BAIL_IF_ERR(nv_err = app.Run());
  BAIL_IF_ERR(nv_err = app.CloseOutputVideo());
  BAIL_IF_ERR(nv_err = app.CloseInputVideo());

bail:
  if (nv_err != NVCV_SUCCESS) {
    printf("Error: %s\n", NvCV_GetErrorStringFromCode(nv_err));
    return static_cast<int>(nv_err);
  }

  return 0;
}
