/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <deque>
#include <iostream>
#include <sstream>
#include <vector>

#include "diarizationReader.h"
#include "nvAR.h"
#include "nvARActiveSpeakerDetection.h"
#include "nvAR_defs.h"
#include "nvCVOpenCV.h"
#include "waveReadWrite.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Macros                                                                                                             //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#if CV_MAJOR_VERSION >= 4
#define CV_CAP_PROP_FRAME_WIDTH cv::CAP_PROP_FRAME_WIDTH
#define CV_CAP_PROP_FRAME_HEIGHT cv::CAP_PROP_FRAME_HEIGHT
#define CV_CAP_PROP_FPS cv::CAP_PROP_FPS
#define CV_WINDOW_AUTOSIZE cv::WINDOW_AUTOSIZE
#endif

#ifdef _MSC_VER
#define strcasecmp _stricmp
#endif

#define BAIL_IF_ERR(err)                        \
  do {                                          \
    if (!CheckResult(err, __LINE__)) goto bail; \
  } while (0)

#define BAIL(err, code) \
  do {                  \
    err = code;         \
    goto bail;          \
  } while (0)

#define SAMPLE_RATE_UNINITIALIZED 0

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Global variables                                                                                                   //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

char* g_nvARSDKPath = nullptr;

bool FLAG_verbose = false;
bool FLAG_show = false;
unsigned FLAG_logLevel = NVCV_LOG_ERROR;
std::string FLAG_inVideo;
std::vector<std::string> FLAG_inAudios;
std::string FLAG_outVideo;
std::string FLAG_modelPath;
std::string FLAG_log;
std::string FLAG_codec = "mp4v";   // FourCC for output video
float FLAG_syncTolerance = -1.0f;  // [0, 1] to set; -1 = unset (use SDK default)
unsigned FLAG_maxSyncFaces = 0;    // max faces for sync discrimination (0 = no limit)
std::string FLAG_diarization;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Types                                                                                                            ///
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class App {
 public:
  App();
  ~App() { Stop(); }

  NvCV_Status SetInputVideo(const std::string& file);
  NvCV_Status SetInputAudio(const std::string& file);
  NvCV_Status SetOutputVideo(const std::string& file);

  NvCV_Status Init();
  NvCV_Status Run();
  NvCV_Status Stop();

 private:
  CUstream m_cudaStream;
  NvAR_FeatureHandle m_speakerDetectionHandle;
  std::vector<NvAR_AudioFrame> m_inputAudioFrames;  // One per input audio stream
  NvAR_AudioFrameData m_inputAudioFrameData;
  NvAR_ActiveAudioIds m_inputActiveAudioIds;
  std::vector<NvAR_SpeakerTrackingBBox> m_outputBoxes;
  NvAR_ActiveSpeakerTrackingData m_outputTrackingData;
  std::vector<std::vector<float>> m_audioFrameDataBuffers;  // One buffer per audio stream

  NvCVImage m_InputImageCpu;
  NvCVImage m_InputImageGpu;
  NvCVImage m_tmpImage;

  cv::VideoCapture m_vidIn;
  cv::VideoWriter m_vidOut;
  cv::Mat m_imgIn;
  cv::Mat m_imgOut;

  std::string m_outFile;
  std::string m_inVideoFile;
  std::vector<std::string> m_inAudioFiles;

  // Audio data
  std::vector<std::vector<float>*> m_audioSamples;  // One vector per audio stream
  std::vector<unsigned int> m_audioNumSamples;

  // Feature configuration
  unsigned int m_videoWidth;
  unsigned int m_videoHeight;
  float m_videoFps;
  uint32_t m_sampleRate;
  unsigned int m_numAudioStreams;
  unsigned int m_maxNumOutputIdentities;

  // Processing state
  unsigned int m_frameCount;
  unsigned int m_outputFrameCount;
  std::vector<unsigned int> m_lastAudioEndSample;  // One per audio stream

  std::vector<uint32_t> m_activeAudioIds;  // Buffer for active audio IDs to pass to API

  // Diarization data for active audio ID determination
  DiarizationReader m_diarizationReader;
  bool m_hasDiarization;

  // Batched parameters (single element for non-batched use)
  uint32_t m_newShot;  // Input: shot change mode
  uint32_t m_flush;    // Input: flush mode (not used yet)
  uint32_t m_flagsVal; // Input: flags
  uint32_t m_ready;    // Output: ready status

  // Frame caching for output synchronization
  std::deque<cv::Mat> m_frameCache;
  std::deque<NvAR_ActiveSpeakerTrackingData> m_outputCache;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Static function declarations                                                                                       //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static bool CheckResult(NvCV_Status nvErr, unsigned line);

static bool GetFlagArgVal(const char* flag, const char* arg, const char** val);

static bool GetFlagArgVal(const char* flag, const char* arg, std::string* val);

static bool GetFlagArgVal(const char* flag, const char* arg, bool* val);

static bool GetFlagArgVal(const char* flag, const char* arg, unsigned* val);

static bool GetFlagArgVal(const char* flag, const char* arg, float* val);

static bool GetFlagArgValAndSplit(const char* flag, const char* arg, std::vector<std::string>* vals);

static int StringToFourcc(const std::string& str);

static void Usage();

static int ParseMyArgs(int argc, char** argv);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Static function definitions                                                                                        //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static bool CheckResult(const NvCV_Status nvErr, const unsigned line) {
  if (NVCV_SUCCESS == nvErr) return true;
  std::cout << "ERROR: " << NvCV_GetErrorStringFromCode(nvErr) << ", line " << line << std::endl;
  return false;
}

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
  if ((strlen(flag) != n) || (strncmp(flag, arg, n) != 0)) {
    return false;
  }
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

static bool GetFlagArgVal(const char* flag, const char* arg, unsigned* val) {
  const char* valStr;
  bool success = GetFlagArgVal(flag, arg, &valStr);
  if (success) *val = (unsigned)atoi(valStr);
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
    try {
      std::string value(valStr);
      std::istringstream iss(value);
      std::string part;
      while (std::getline(iss, part, ',')) {
        if (!part.empty()) {
          vals->push_back(part);
        }
      }
    } catch (const std::ios_base::failure&) {
      return false;
    }
  }
  return true;
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

static void Usage() {
  printf(
      "ActiveSpeakerDetectionApp <args...>\n"
      "where <args...> is one or more of\n"
      " --verbose[=(true|false)]   enable verbose output\n"
      " --show[=(true|false)]      show output video window\n"
      " --in_video=<path>          input video file path (required)\n"
      " --in_audios=<a0,a1,...>    comma-separated WAV paths (required). With --diarization, exactly one mix-down "
      "file is expected; logical channels are derived from diarization speaker IDs.\n"
      " --out_video=<path>         output video file path (default: activeSpeakerDetectionOutput.mp4)\n"
      " --codec=<fourcc>           FOURCC for output video (default: mp4v; e.g. avc1 for H.264)\n"
      " --model_path=<path>        directory containing the TRT models (required)\n"
      " --log=<file>               log SDK errors to file, \"stderr\" or \"\" (default: stderr)\n"
      " --log_level=<N>            log level: {0, 1, 2, 3} = {FATAL, ERROR, WARNING, INFO} (default: 1)\n"
      " --sync_tolerance=<f>       min sync score [0, 1] to consider speaking (-1 = unset, use SDK default)\n"
      " --max_sync_faces=<N>       max faces for sync discrimination (default: 0 = no limit)\n"
      " --diarization=<path>       path to diarization JSON file for active audio ID determination\n");
}

static int ParseMyArgs(int argc, char** argv) {
  int errs = 0;
  for (--argc, ++argv; argc--; ++argv) {
    bool help;
    const char* arg = *argv;
    if (arg[0] != '-') {
      continue;
    } else if ((arg[1] == '-') &&                                             //
               (GetFlagArgVal("verbose", arg, &FLAG_verbose) ||               //
                GetFlagArgVal("show", arg, &FLAG_show) ||                     //
                GetFlagArgVal("in_video", arg, &FLAG_inVideo) ||              //
                GetFlagArgValAndSplit("in_audios", arg, &FLAG_inAudios) ||    //
                GetFlagArgVal("out_video", arg, &FLAG_outVideo) ||            //
                GetFlagArgVal("codec", arg, &FLAG_codec) ||                   //
                GetFlagArgVal("model_path", arg, &FLAG_modelPath) ||          //
                GetFlagArgVal("log", arg, &FLAG_log) ||                       //
                GetFlagArgVal("log_level", arg, &FLAG_logLevel) ||            //
                GetFlagArgVal("sync_tolerance", arg, &FLAG_syncTolerance) ||  //
                GetFlagArgVal("max_sync_faces", arg, &FLAG_maxSyncFaces) ||   //
                GetFlagArgVal("diarization", arg, &FLAG_diarization))) {
      continue;
    } else if (GetFlagArgVal("help", arg, &help)) {
      Usage();
    }
  }
  return errs;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Member function definitions                                                                                        //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

App::App()
    : m_cudaStream(nullptr),
      m_speakerDetectionHandle(nullptr),
      m_inputAudioFrameData({}),
      m_inputActiveAudioIds({}),
      m_outputTrackingData({}),
      m_InputImageCpu({}),
      m_InputImageGpu({}),
      m_tmpImage({}),
      m_videoWidth(0),
      m_videoHeight(0),
      m_videoFps(0.0f),
      m_sampleRate(SAMPLE_RATE_UNINITIALIZED),
      m_numAudioStreams(0),
      m_maxNumOutputIdentities(0),
      m_frameCount(0),
      m_outputFrameCount(0),
      m_newShot(NVARACTIVESPEAKERDETECTION_DETECT_SHOT_CHANGE),
      m_flush(0),
      m_flagsVal(0),
      m_ready(0),
      m_hasDiarization(false) {}

NvCV_Status App::SetInputVideo(const std::string& file) {
  NvCV_Status err = NVCV_SUCCESS;
  m_inVideoFile = file;

  if (!m_vidIn.open(file)) {
    std::cout << "ERROR: Unable to open video file: " << file << std::endl;
    BAIL(err, NVCV_ERR_READ);
  }

  m_videoWidth = static_cast<unsigned int>(m_vidIn.get(CV_CAP_PROP_FRAME_WIDTH));
  m_videoHeight = static_cast<unsigned int>(m_vidIn.get(CV_CAP_PROP_FRAME_HEIGHT));
  m_videoFps = static_cast<float>(m_vidIn.get(CV_CAP_PROP_FPS));

  if (FLAG_verbose) {
    std::cout << "Video resolution: " << m_videoWidth << "x" << m_videoHeight << " @ " << m_videoFps << " fps"
              << std::endl;
  }

bail:
  return err;
}

NvCV_Status App::SetInputAudio(const std::string& file) {
  NvCV_Status err = NVCV_SUCCESS;
  m_inAudioFiles.push_back(file);

  uint32_t actual_sample_rate = 0;
  std::vector<float>* audio_samples;
  unsigned int num_samples = 0;

  // Get the actual sample rate from the file
  {
    CWaveFileRead wave_reader(file);
    if (!wave_reader.isValid()) {
      std::cout << "ERROR: Unable to open audio file: " << file << std::endl;
      BAIL(err, NVCV_ERR_READ);
    }
    actual_sample_rate = wave_reader.GetSampleRate();
  }

  if (!ReadWavFile(file, actual_sample_rate, 1, &audio_samples, &num_samples, nullptr, -1, FLAG_verbose)) {
    std::cout << "ERROR: Unable to read audio file: " << file << std::endl;
    BAIL(err, NVCV_ERR_READ);
  }

  if (m_sampleRate != SAMPLE_RATE_UNINITIALIZED && m_sampleRate != actual_sample_rate) {
    std::cout << "ERROR: Sample rate mismatch. Expected " << m_sampleRate << " Hz, got " << actual_sample_rate << " Hz"
              << std::endl;
    BAIL(err, NVCV_ERR_MISMATCH);
  }
  m_sampleRate = actual_sample_rate;
  m_audioSamples.push_back(audio_samples);
  m_audioNumSamples.push_back(num_samples);

  if (FLAG_verbose) {
    std::cout << "Audio loaded: " << num_samples << " samples @ " << actual_sample_rate << " Hz" << std::endl;
  }

bail:
  return err;
}

NvCV_Status App::SetOutputVideo(const std::string& file) {
  m_outFile = file;
  return NVCV_SUCCESS;
}

NvCV_Status App::Init() {
  NvCV_Status err = NVCV_SUCCESS;

  int codec_fourcc;
  codec_fourcc = StringToFourcc(FLAG_codec);

  // Update number of audio streams based on loaded audio files
  m_numAudioStreams = static_cast<unsigned int>(m_audioSamples.size());

  // Load diarization data if provided
  if (!FLAG_diarization.empty()) {
    err = m_diarizationReader.Load(FLAG_diarization);
    BAIL_IF_ERR(err);
    if (m_diarizationReader.maxSpeakerId() < 0) {
      std::cerr << "ERROR: Diarization must contain at least one word with speaker_id" << std::endl;
      BAIL(err, NVCV_ERR_PARSE);
    }
    if (m_audioSamples.size() != 1u) {
      std::cerr << "ERROR: --diarization requires exactly one audio file in --in_audios, got "
                << m_audioSamples.size() << std::endl;
      BAIL(err, NVCV_ERR_PARAMETER);
    }
    m_hasDiarization = true;
    m_numAudioStreams = static_cast<unsigned int>(m_diarizationReader.maxSpeakerId()) + 1u;
    m_audioSamples.assign(m_numAudioStreams, m_audioSamples[0]);
    m_audioNumSamples.assign(m_numAudioStreams, m_audioNumSamples[0]);
    if (FLAG_verbose) {
      std::cout << "Loaded diarization with " << m_diarizationReader.numSpeakers()
                << " speakers (max speaker_id=" << m_diarizationReader.maxSpeakerId() << ")" << std::endl;
    }
  }

  // Create feature
  err = NvAR_Create(NvAR_Feature_ActiveSpeakerDetection, &m_speakerDetectionHandle);
  BAIL_IF_ERR(err);

  // Create CUDA stream
  err = NvAR_CudaStreamCreate(&m_cudaStream);
  BAIL_IF_ERR(err);

  // Configure feature
  err = NvAR_SetString(m_speakerDetectionHandle, NvAR_Parameter_Config(ModelDir), FLAG_modelPath.c_str());
  BAIL_IF_ERR(err);

  err = NvAR_SetCudaStream(m_speakerDetectionHandle, NvAR_Parameter_Config(CUDAStream), m_cudaStream);
  BAIL_IF_ERR(err);

  err = NvAR_SetF32(m_speakerDetectionHandle, NvAR_Parameter_Config(VideoFPS), m_videoFps);
  BAIL_IF_ERR(err);

  err = NvAR_SetU32(m_speakerDetectionHandle, NvAR_Parameter_Config(NumAudioStreams), m_numAudioStreams);
  BAIL_IF_ERR(err);

  err = NvAR_SetU32(m_speakerDetectionHandle, NvAR_Parameter_Config(SampleRate), m_sampleRate);
  BAIL_IF_ERR(err);

  // Load feature
  err = NvAR_Load(m_speakerDetectionHandle);
  BAIL_IF_ERR(err);

  // Set up batched parameters (single element for non-batched use)
  m_newShot = NVARACTIVESPEAKERDETECTION_DETECT_SHOT_CHANGE;
  m_flush = 0;
  m_ready = 0;
  err = NvAR_SetU32Array(m_speakerDetectionHandle, NvAR_Parameter_Input(NewShot), &m_newShot, 1);
  BAIL_IF_ERR(err);
  err = NvAR_SetU32Array(m_speakerDetectionHandle, NvAR_Parameter_Input(Flush), &m_flush, 1);
  BAIL_IF_ERR(err);
  err = NvAR_SetU32Array(m_speakerDetectionHandle, NvAR_Parameter_Output(Ready), &m_ready, 1);
  BAIL_IF_ERR(err);

  err = NvAR_GetU32(m_speakerDetectionHandle, NvAR_Parameter_Config(MaxNumOutputIdentities), &m_maxNumOutputIdentities);
  BAIL_IF_ERR(err);

  if (FLAG_syncTolerance >= 0.0f) {
    err = NvAR_SetF32(m_speakerDetectionHandle, NvAR_Parameter_Config(SyncTolerance), FLAG_syncTolerance);
    BAIL_IF_ERR(err);
  } else if (FLAG_verbose) {
    std::cout << "Sync tolerance not set; using SDK default." << std::endl;
  }

  err = NvAR_SetU32(m_speakerDetectionHandle, NvAR_Parameter_Config(MaxSyncFaces), FLAG_maxSyncFaces);
  BAIL_IF_ERR(err);

  m_flagsVal = m_hasDiarization ? 0u : NVARACTIVESPEAKERDETECTION_FLAG_FILTER_SILENT_TRACKS;
  // NewShot API is still used for shot-change; no shot bits set here.
  err = NvAR_SetU32Array(m_speakerDetectionHandle, NvAR_Parameter_Config(Flags), &m_flagsVal, 1);
  BAIL_IF_ERR(err);

  if (FLAG_verbose) {
    std::cout << "Feature loaded. Max output identities: " << m_maxNumOutputIdentities << std::endl;
  }

  // Allocate images
  err = NvCVImage_Realloc(&m_InputImageCpu, m_videoWidth, m_videoHeight, NVCV_BGR, NVCV_U8, NVCV_CHUNKY, NVCV_CPU, 1);
  BAIL_IF_ERR(err);
  err = NvCVImage_Realloc(&m_InputImageGpu, m_videoWidth, m_videoHeight, NVCV_BGR, NVCV_U8, NVCV_CHUNKY, NVCV_CUDA, 1);
  BAIL_IF_ERR(err);

  // Set up speaker data input for each audio stream
  m_inputAudioFrames.resize(m_numAudioStreams);
  m_audioFrameDataBuffers.resize(m_numAudioStreams);

  for (unsigned int i = 0; i < m_numAudioStreams; ++i) {
    m_audioFrameDataBuffers[i].resize(static_cast<size_t>(m_sampleRate) + 1024, 0.0f);
    m_inputAudioFrames[i].audio_data = m_audioFrameDataBuffers[i].data();
    m_inputAudioFrames[i].num_samples = 0;
    m_inputAudioFrames[i].audio_id = i;
  }

  m_inputAudioFrameData.audio_frames = m_inputAudioFrames.data();
  m_inputAudioFrameData.num_audio_channels = m_numAudioStreams;

  // Set up tracking data output
  m_outputBoxes.resize(m_maxNumOutputIdentities);
  m_outputTrackingData.boxes = m_outputBoxes.data();
  m_outputTrackingData.max_boxes = static_cast<uint8_t>(m_outputBoxes.size());

  // Open output video writer
  if (!m_vidOut.open(m_outFile, codec_fourcc, m_videoFps, cv::Size(m_videoWidth, m_videoHeight))) {
    std::cout << "ERROR: Unable to open output video file: " << m_outFile << std::endl;
    return NVCV_ERR_WRITE;
  }

bail:
  return err;
}

NvCV_Status App::Run() {
  NvCV_Status err = NVCV_SUCCESS;

  const float estimated_video_frame_duration = 1.0f / m_videoFps;
  bool is_flushing = false;

  m_frameCount = 0;
  m_outputFrameCount = 0;
  m_lastAudioEndSample.resize(m_numAudioStreams, 0);

  while (true) {
    // Skip reading new frames when flushing
    bool got_video_frame = false;
    if (!is_flushing) {
      got_video_frame = m_vidIn.read(m_imgIn);
    }

    // Get timestamp from video frame in seconds
    const double frame_timestamp = m_frameCount * estimated_video_frame_duration;

    m_newShot = NVARACTIVESPEAKERDETECTION_DETECT_SHOT_CHANGE;

    // Prepare audio frame for each track
    // When audio ends before video, continue with zero-filled audio
    m_activeAudioIds.clear();
    std::vector<bool> has_audio_data(m_numAudioStreams, false);

    for (unsigned int track_idx = 0; track_idx < m_numAudioStreams; ++track_idx) {
      // Calculate audio window based on video timestamp for this track
      const unsigned int audio_start_sample = m_lastAudioEndSample[track_idx];
      const unsigned int audio_end_sample =
          static_cast<unsigned int>((frame_timestamp + estimated_video_frame_duration) * m_sampleRate);
      const unsigned audio_frame_length = audio_end_sample - audio_start_sample;
      m_lastAudioEndSample[track_idx] = audio_end_sample;

      // Create audio frame initialized to zero (generates silence if audio file has finished)
      std::fill(m_audioFrameDataBuffers[track_idx].begin(), m_audioFrameDataBuffers[track_idx].end(), 0.0f);

      // Copy valid audio into the frame
      const size_t valid_audio_start_sample = std::min<size_t>(audio_start_sample, m_audioSamples[track_idx]->size());
      const size_t valid_audio_end_sample = std::min<size_t>(audio_end_sample, m_audioSamples[track_idx]->size());

      std::copy(m_audioSamples[track_idx]->begin() + valid_audio_start_sample,
                m_audioSamples[track_idx]->begin() + valid_audio_end_sample,
                m_audioFrameDataBuffers[track_idx].begin());

      has_audio_data[track_idx] = audio_start_sample < m_audioSamples[track_idx]->size();

      // Update audio frame size
      m_inputAudioFrames[track_idx].num_samples = audio_frame_length;
    }

    if (m_hasDiarization) {
      // Activate per diarization at this time interval
      double frame_time_end = frame_timestamp + estimated_video_frame_duration;
      std::vector<uint32_t> diarization_active =
          m_diarizationReader.GetActiveAudioIdsAndAdvanceCursor(frame_timestamp, frame_time_end);
      for (uint32_t active_audio_idx : diarization_active) {
        if (active_audio_idx < m_numAudioStreams && has_audio_data[active_audio_idx]) {
          m_activeAudioIds.push_back(m_inputAudioFrames[active_audio_idx].audio_id);
        }
      }
    } else {
      for (unsigned int active_audio_idx = 0; active_audio_idx < m_numAudioStreams; ++active_audio_idx) {
        if (has_audio_data[active_audio_idx]) {
          m_activeAudioIds.push_back(m_inputAudioFrames[active_audio_idx].audio_id);
        }
      }
    }

    // Enter flush mode if video ends (audio can continue with zeros)
    if (!is_flushing && !got_video_frame) {
      is_flushing = true;
      m_flush = 1;
      if (FLAG_verbose) {
        std::cout << "End of video input. Flushing " << m_frameCache.size() << " frames..." << std::endl;
      }
    }

    // Exit if flushing and frame cache is empty
    if (is_flushing && m_frameCache.empty()) break;

    // Normal processing: cache frame and prepare inputs
    if (!is_flushing) {
      m_frameCache.push_back(m_imgIn.clone());

      // Transfer video frame to GPU
      (void)NVWrapperForCVMat(&m_imgIn, &m_InputImageCpu);
      err = NvCVImage_Transfer(&m_InputImageCpu, &m_InputImageGpu, 1.0f, m_cudaStream, &m_tmpImage);
      BAIL_IF_ERR(err);

      // Set inputs
      err = NvAR_SetObject(m_speakerDetectionHandle, NvAR_Parameter_Input(Image), &m_InputImageGpu, sizeof(NvCVImage));
      BAIL_IF_ERR(err);

      m_frameCount++;
    } else {
      // During flushing, prepare zero-filled audio with all channels active
      for (unsigned int track_idx = 0; track_idx < m_numAudioStreams; ++track_idx) {
        std::fill(m_audioFrameDataBuffers[track_idx].begin(), m_audioFrameDataBuffers[track_idx].end(), 0.0f);
        m_inputAudioFrames[track_idx].num_samples =
            static_cast<unsigned int>(m_sampleRate / m_videoFps);  // One frame worth of samples
      }
      m_activeAudioIds.clear();
      for (unsigned int track_idx = 0; track_idx < m_numAudioStreams; ++track_idx) {
        m_activeAudioIds.push_back(m_inputAudioFrames[track_idx].audio_id);
      }
    }

    // Set active audio IDs
    m_inputActiveAudioIds.active_audio_ids = m_activeAudioIds.data();
    m_inputActiveAudioIds.num_active_audio_ids = static_cast<uint32_t>(m_activeAudioIds.size());
    err = NvAR_SetObject(m_speakerDetectionHandle, NvAR_Parameter_Input(ActiveAudioIDs), &m_inputActiveAudioIds,
                         sizeof(NvAR_ActiveAudioIds));
    BAIL_IF_ERR(err);

    err = NvAR_SetObject(m_speakerDetectionHandle, NvAR_Parameter_Input(AudioFrameData), &m_inputAudioFrameData,
                         sizeof(NvAR_AudioFrameData));
    BAIL_IF_ERR(err);

    err = NvAR_SetObject(m_speakerDetectionHandle, NvAR_Parameter_Output(ActiveSpeakerTrackingData),
                         &m_outputTrackingData, sizeof(NvAR_ActiveSpeakerTrackingData));
    BAIL_IF_ERR(err);

    // Run the feature
    err = NvAR_Run(m_speakerDetectionHandle);
    BAIL_IF_ERR(err);

    err = NvAR_CudaStreamSynchronize(m_cudaStream);
    BAIL_IF_ERR(err);

    // Cache output if ready
    if (m_ready != 0) {
      NvAR_ActiveSpeakerTrackingData output_copy;
      output_copy.num_boxes = m_outputTrackingData.num_boxes;
      output_copy.max_boxes = m_outputTrackingData.num_boxes;
      output_copy.boxes = new NvAR_SpeakerTrackingBBox[output_copy.num_boxes];
      std::copy(m_outputBoxes.begin(), m_outputBoxes.begin() + output_copy.num_boxes, output_copy.boxes);
      m_outputCache.push_back(output_copy);
    } else if (FLAG_verbose && m_frameCount == 0) {
      std::cout << "Waiting for enough frames to accumulate..." << std::endl;
    }

    // Write outputs when both frame and output are available
    while (!m_frameCache.empty() && !m_outputCache.empty()) {
      cv::Mat& cached_frame = m_frameCache.front();
      NvAR_ActiveSpeakerTrackingData& cached_output = m_outputCache.front();

      m_imgOut = cached_frame.clone();

      // Scale font for high-resolution videos (1440p and above)
      const bool high_res = m_imgOut.rows >= 1440;
      const double font_scale = high_res ? 1.4 : 0.7;
      const int font_thickness = high_res ? 4 : 2;
      const int box_thickness = high_res ? 6 : 3;
      const int label_padding = high_res ? 20 : 10;
      const int label_offset = high_res ? 10 : 5;

      // Draw frame counter at top left
      std::string frame_text = cv::format("Frame: %d", static_cast<int>(m_outputFrameCount));
      const int frame_y = high_res ? 60 : 30;
      cv::putText(m_imgOut, frame_text, cv::Point(10, frame_y), cv::FONT_HERSHEY_SIMPLEX, font_scale,
                  cv::Scalar(0, 255, 0), font_thickness, cv::LINE_AA);

      // Draw bounding boxes with three-color scheme
      for (unsigned int i = 0; i < cached_output.num_boxes; ++i) {
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
        cv::Rect rect(static_cast<int>(bbox.bbox.x), static_cast<int>(bbox.bbox.y), static_cast<int>(bbox.bbox.width),
                      static_cast<int>(bbox.bbox.height));
        cv::rectangle(m_imgOut, rect, color, box_thickness);

        // Draw track label with background at top-left
        std::string label_text = cv::format("Track:%d (%.2f)", bbox.tracking_id, bbox.confidence);
        int baseline = 0;
        cv::Size text_size =
            cv::getTextSize(label_text, cv::FONT_HERSHEY_SIMPLEX, font_scale, font_thickness, &baseline);
        int box_x = static_cast<int>(bbox.bbox.x);
        int box_y = static_cast<int>(bbox.bbox.y);
        cv::rectangle(m_imgOut, cv::Point(box_x, box_y - text_size.height - label_padding),
                      cv::Point(box_x + text_size.width, box_y), color, -1);
        cv::putText(m_imgOut, label_text, cv::Point(box_x, box_y - label_offset), cv::FONT_HERSHEY_SIMPLEX, font_scale,
                    text_color, font_thickness);

        // Draw audio ID at bottom-right corner if assigned
        if (bbox.audio_id != -1) {
          std::string audio_label = cv::format("Audio:%d", bbox.audio_id);
          cv::Size audio_text_size =
              cv::getTextSize(audio_label, cv::FONT_HERSHEY_SIMPLEX, font_scale, font_thickness, &baseline);
          int audio_x = static_cast<int>(bbox.bbox.x + bbox.bbox.width) - audio_text_size.width;
          int audio_y = static_cast<int>(bbox.bbox.y + bbox.bbox.height);
          cv::rectangle(m_imgOut, cv::Point(audio_x, audio_y),
                        cv::Point(audio_x + audio_text_size.width, audio_y + audio_text_size.height + label_padding),
                        color, -1);
          cv::putText(m_imgOut, audio_label, cv::Point(audio_x, audio_y + audio_text_size.height + label_offset),
                      cv::FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness);
        }
      }

      // Write output frame
      m_vidOut.write(m_imgOut);

      // Show output if requested
      if (FLAG_show) {
        cv::imshow("ActiveSpeakerDetection Output", m_imgOut);
        if (cv::waitKey(1) == 27) {
          if (!is_flushing) {
            is_flushing = true;
            m_flush = 1;
            if (FLAG_verbose) {
              std::cout << "Flushing " << m_frameCache.size() << " frames..." << std::endl;
            }
          }
        }
      }

      delete[] cached_output.boxes;
      m_frameCache.pop_front();
      m_outputCache.pop_front();
      m_outputFrameCount++;
    }

    if (FLAG_verbose && (m_frameCount % 30 == 0)) {
      std::cout << "Processed " << m_frameCount << " frames..." << std::endl;
    }
  }

  // Clean up any remaining cached outputs
  for (auto& cached_output : m_outputCache) {
    delete[] cached_output.boxes;
  }
  m_outputCache.clear();

  std::cout << "Processing complete!" << std::endl;
  std::cout << "Total frames processed: " << m_frameCount << std::endl;
  std::cout << "Output written to: " << m_outFile << std::endl;

bail:
  return err;
}

NvCV_Status App::Stop() {
  // Clean up
  if (m_vidOut.isOpened()) m_vidOut.release();
  if (m_vidIn.isOpened()) m_vidIn.release();
  if (FLAG_show) cv::destroyAllWindows();

  NvCVImage_Dealloc(&m_InputImageCpu);
  NvCVImage_Dealloc(&m_InputImageGpu);
  NvCVImage_Dealloc(&m_tmpImage);

  if (m_cudaStream) {
    NvAR_CudaStreamDestroy(m_cudaStream);
    m_cudaStream = nullptr;
  }

  if (m_speakerDetectionHandle) {
    NvAR_Destroy(m_speakerDetectionHandle);
    m_speakerDetectionHandle = nullptr;
  }

  return NVCV_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Main                                                                                                               //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {
  int errs = 0;
  NvCV_Status err = NVCV_SUCCESS;
  App app;

  // Parse command-line arguments
  errs = ParseMyArgs(argc, argv);
  if (errs) {
    Usage();
    return errs;
  }

  // Validate required arguments
  if (FLAG_inVideo.empty() || FLAG_inAudios.empty()) {
    std::cout << "ERROR: --in_video and --in_audios are required" << std::endl;
    Usage();
    return 100;
  }

  if (FLAG_outVideo.empty()) FLAG_outVideo = "activeSpeakerDetectionOutput.mp4";
  if (FLAG_log.empty()) FLAG_log = "stderr";

  // Configure logger
  err = NvAR_ConfigureLogger(FLAG_logLevel, FLAG_log.c_str(), nullptr, nullptr);
  BAIL_IF_ERR(err);

  if (FLAG_verbose) {
    std::cout << "Input video: " << FLAG_inVideo << std::endl;
    for (size_t i = 0; i < FLAG_inAudios.size(); ++i) {
      std::cout << "Input audio " << i << ": " << FLAG_inAudios[i] << std::endl;
    }
    std::cout << "Output video: " << FLAG_outVideo << std::endl;
    std::cout << "Model path: " << FLAG_modelPath << std::endl;
  }

  // Input / Output
  err = app.SetInputVideo(FLAG_inVideo);
  BAIL_IF_ERR(err);

  // Load all audio streams
  for (const std::string& audio_file : FLAG_inAudios) {
    err = app.SetInputAudio(audio_file);
    BAIL_IF_ERR(err);
  }

  err = app.SetOutputVideo(FLAG_outVideo);
  BAIL_IF_ERR(err);

  // Initialize and run
  err = app.Init();
  BAIL_IF_ERR(err);
  err = app.Run();
  BAIL_IF_ERR(err);

bail:
  err = app.Stop();
  return static_cast<int>(err);
}
