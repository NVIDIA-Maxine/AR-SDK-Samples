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

#include <memory>
#include <string>

#include "batchUtilities.h"
#include "nvAR.h"
#include "nvARFaceBoxDetection.h"
#include "nvARLandmarkDetection.h"
#include "nvCVOpenCV.h"
#include "opencv2/opencv.hpp"

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

bool FLAG_verbose = false;
bool FLAG_isLandmarks126 = false;
bool FLAG_useTritonGRPC = false;
std::string FLAG_tritonURL = "localhost:8001";
std::string FLAG_modelPath;
std::string FLAG_effect;
std::string FLAG_outputNameTag = "output";
std::string FLAG_log = "stderr";
std::vector<const char*> FLAG_inSrcVideoFiles;
std::vector<std::string> FLAG_srcImages;
unsigned FLAG_landmarksMode = 0;
unsigned FLAG_temporal = 0xFFFFFFFF;
unsigned FLAG_logLevel = NVCV_LOG_ERROR;

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
      "FaceTrackTritonClient [flags ...] inVideoFile1 [inVideoFileN ...]\n"
      "  where flags is:\n"
      "  --effect=<effect>                  the effect to apply (supported: FaceBoxDetection, LandmarkDetection).\n"
      "  --url=<URL>                        URL to the Triton server\n"
      "  --grpc[=(true|false)]              use gRPC for data transfer to the Triton server instead of CUDA shared "
      "memory.\n"
      "  --output_name_tag=<string>         a string appended to each inFile to create the corresponding output file "
      "name\n"
      "  --log=<file>                       log SDK errors to a file, \"stderr\" or \"\" (default stderr)\n"
      "  --log_level=<N>                    the desired log level: {0, 1, 2} = {FATAL, ERROR, WARNING}, respectively "
      "(default 1)\n"
      "  --temporal                         temporal flag (default 0xFFFFFFFF)\n"
      "\n  Landmark detection only:\n"
      "    --landmarks_126[=(true|false)]     set the number of facial landmark points to 126, otherwise default to "
      "68\n"
      "    --landmark_mode                    select Landmark Detection Model. 0: Performance (Default),  1: "
      "Quality\n");
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
            GetFlagArgVal("effect", arg, &FLAG_effect) ||                  //
            GetFlagArgVal("model_path", arg, &FLAG_modelPath) ||           //
            GetFlagArgVal("output_name_tag", arg, &FLAG_outputNameTag) ||  //
            GetFlagArgVal("landmarks_126", arg, &FLAG_isLandmarks126) ||   //
            GetFlagArgVal("landmark_mode", arg, &FLAG_landmarksMode) ||    //
            GetFlagArgVal("log", arg, &FLAG_log) ||                        //
            GetFlagArgVal("log_level", arg, &FLAG_logLevel) ||             //
            GetFlagArgVal("temporal", arg, &FLAG_temporal)) {
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
            printf("Unknown flag ignored: \"-%c\"\n", *arg);
          }
        }
        continue;
      }
    } else {  // no dash
      FLAG_inSrcVideoFiles.push_back(arg);
    }
  }
  return errs;
}

class BaseApp {
 public:
  std::string m_effectName;
  NvAR_TritonServer m_triton;
  NvAR_FeatureHandle m_effect;
  NvCVImage m_srcVidFrame, m_firstSrc, m_stg;
  CUstream m_cudaStream;
  unsigned m_numOfVideoStreams;
  unsigned m_outputImgVizWidth, m_outputImgVizHeight;
  std::vector<NvAR_StateHandle> m_arrayOfAllStateObjects;
  std::vector<NvAR_StateHandle> m_batchOfStateObjects;

  static BaseApp* Create(const char* effect_name);
  virtual ~BaseApp() {
    if (m_effect) NvAR_Destroy(m_effect);
    if (m_cudaStream) NvAR_CudaStreamDestroy(m_cudaStream);
    if (m_triton) NvAR_DisconnectTritonServer(m_triton);
  }
  virtual NvCV_Status Init(unsigned num_video_streams) {
    NvCV_Status err = NVCV_SUCCESS;
    m_numOfVideoStreams = num_video_streams;
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
    m_arrayOfAllStateObjects.resize(m_numOfVideoStreams, nullptr);
    m_batchOfStateObjects.resize(m_numOfVideoStreams, nullptr);
    if (FLAG_verbose) {
      printf("Using triton server\n");
    }
  bail:
    return err;
  }
  virtual NvCV_Status AllocateBuffers(unsigned src_vid_width, unsigned src_vid_height) = 0;
  virtual NvCV_Status SetParametersBeforeLoad() = 0;
  virtual NvCV_Status SetParametersAfterLoad() { return NVCV_SUCCESS; }
  virtual NvCV_Status GenerateNthOutputVizImage(unsigned n, const cv::Mat& input, cv::Mat& result) = 0;
  virtual NvCV_Status Load() { return NvAR_Load(m_effect); }
  virtual NvCV_Status Run(const unsigned* batch_indices, unsigned batchsize) {
    NvCV_Status err = NVCV_SUCCESS;
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
  virtual NvCV_Status InitVideoStream(unsigned n) { return NvAR_AllocateState(m_effect, &m_arrayOfAllStateObjects[n]); }
  virtual NvCV_Status ReleaseVideoStream(unsigned n) {
    return NvAR_DeallocateState(m_effect, m_arrayOfAllStateObjects[n]);
  }

 protected:
  BaseApp() : m_triton(nullptr), m_effect(nullptr), m_cudaStream(0), m_numOfVideoStreams(0) {}
};

class FaceDetectionApp : public BaseApp {
 public:
  std::vector<std::vector<NvAR_Rect>> m_outputBboxData;
  std::vector<NvAR_BBoxes> m_outputBboxes;
  static constexpr int kMaxBoxes = 25;

  NvCV_Status AllocateBuffers(unsigned src_vid_width, unsigned src_vid_height) {
    NvCV_Status err = NVCV_SUCCESS;
    BAIL_IF_ERR(err = AllocateBatchBuffer(&m_srcVidFrame, m_numOfVideoStreams, src_vid_width, src_vid_height, NVCV_BGR,
                                          NVCV_U8, NVCV_CHUNKY, FLAG_useTritonGRPC ? NVCV_CPU : NVCV_CUDA, 1));
    m_outputBboxes.resize(m_numOfVideoStreams);
    m_outputBboxData.resize(m_numOfVideoStreams);
    for (unsigned i = 0; i < m_numOfVideoStreams; i++) {
      m_outputBboxData[i].resize(kMaxBoxes);
      m_outputBboxData[i].assign(kMaxBoxes, {0.f, 0.f, 0.f, 0.f});
      m_outputBboxes[i].boxes = m_outputBboxData[i].data();
      m_outputBboxes[i].max_boxes = kMaxBoxes;
      m_outputBboxes[i].num_boxes = 1;
    }
    m_outputImgVizWidth = src_vid_width;
    m_outputImgVizHeight = src_vid_height;
  bail:
    return err;
  }
  NvCV_Status SetParametersBeforeLoad() {
    NvCV_Status err = NVCV_SUCCESS;
    BAIL_IF_ERR(err =
                    NvAR_SetObject(m_effect, NvAR_Parameter_Input(Image),
                                   NthImage(0, m_srcVidFrame.height / m_numOfVideoStreams, &m_srcVidFrame, &m_firstSrc),
                                   sizeof(NvCVImage)));  // Set the first of the batched images in ...
    BAIL_IF_ERR(err = NvAR_SetObject(m_effect, NvAR_Parameter_Output(BoundingBoxes), m_outputBboxes.data(),
                                     sizeof(NvAR_BBoxes)));
    BAIL_IF_ERR(err = NvAR_SetU32(m_effect, NvAR_Parameter_Config(Temporal), FLAG_temporal));  // Set temporal
  bail:
    return err;
  }
  NvCV_Status GenerateNthOutputVizImage(unsigned n, const cv::Mat& input, cv::Mat& result) {
    NvCV_Status err = NVCV_SUCCESS;
    result = input.clone();
    cv::rectangle(result, cv::Point(lround(m_outputBboxes[n].boxes->x), lround(m_outputBboxes[n].boxes->y)),
                  cv::Point(lround(m_outputBboxes[n].boxes->x + m_outputBboxes[n].boxes->width),
                            lround(m_outputBboxes[n].boxes->y + m_outputBboxes[n].boxes->height)),
                  cv::Scalar(255, 0, 0), 2);
  bail:
    return err;
  }
};

class FacialLandmarksApp : public BaseApp {
 public:
  std::vector<NvAR_Point2f> m_facialLandmarks;
  std::vector<float> m_facialLandmarksConfidence;
  std::vector<NvAR_Quaternion> m_facialPose;
  unsigned m_landmarksMode;
  unsigned m_numLandmarks;

  NvCV_Status AllocateBuffers(unsigned src_vid_width, unsigned src_vid_height) {
    NvCV_Status err = NVCV_SUCCESS;
    m_numLandmarks = FLAG_isLandmarks126 ? 126 : 68;
    m_landmarksMode = FLAG_landmarksMode;
    BAIL_IF_ERR(err = AllocateBatchBuffer(&m_srcVidFrame, m_numOfVideoStreams, src_vid_width, src_vid_height, NVCV_BGR,
                                          NVCV_U8, NVCV_CHUNKY, FLAG_useTritonGRPC ? NVCV_CPU : NVCV_CUDA, 1));
    m_facialLandmarks.resize(m_numOfVideoStreams * m_numLandmarks);
    m_facialLandmarksConfidence.resize(m_numOfVideoStreams * m_numLandmarks);
    m_facialPose.resize(m_numOfVideoStreams);
    BAIL_IF_ERR(err = NvAR_SetF32Array(m_effect, NvAR_Parameter_Output(LandmarksConfidence),
                                       m_facialLandmarksConfidence.data(), m_numOfVideoStreams * m_numLandmarks));
    m_outputImgVizWidth = src_vid_width;
    m_outputImgVizHeight = src_vid_height;
  bail:
    return err;
  }
  NvCV_Status SetParametersBeforeLoad() {
    NvCV_Status err = NVCV_SUCCESS;
    BAIL_IF_ERR(err =
                    NvAR_SetObject(m_effect, NvAR_Parameter_Input(Image),
                                   NthImage(0, m_srcVidFrame.height / m_numOfVideoStreams, &m_srcVidFrame, &m_firstSrc),
                                   sizeof(NvCVImage)));  // Set the first of the batched images in ...
    BAIL_IF_ERR(err = NvAR_SetObject(m_effect, NvAR_Parameter_Output(Landmarks), m_facialLandmarks.data(),
                                     sizeof(NvAR_Point2f)));
    BAIL_IF_ERR(
        err = NvAR_SetObject(m_effect, NvAR_Parameter_Output(Pose), m_facialPose.data(), sizeof(NvAR_Quaternion)));
    BAIL_IF_ERR(err = NvAR_SetU32(m_effect, NvAR_Parameter_Config(Landmarks_Size), m_numLandmarks));
    BAIL_IF_ERR(err = NvAR_SetU32(m_effect, NvAR_Parameter_Config(Mode), m_landmarksMode));
    BAIL_IF_ERR(err = NvAR_SetU32(m_effect, NvAR_Parameter_Config(Temporal), FLAG_temporal));  // Set temporal
  bail:
    return err;
  }
  NvCV_Status GenerateNthOutputVizImage(unsigned n, const cv::Mat& input, cv::Mat& result) {
    NvCV_Status err = NVCV_SUCCESS;
    result = input.clone();
    for (int j = 0; j < m_numLandmarks; j++) {
      cv::circle(result,
                 cv::Point(lround(m_facialLandmarks[m_numLandmarks * n + j].x),
                           lround(m_facialLandmarks[m_numLandmarks * n + j].y)),
                 3, cv::Scalar(255, 0, 0), -1);
    }
  bail:
    return err;
  }
};

BaseApp* BaseApp::Create(const char* effect_name) {
  BaseApp* obj;
  if (!strcasecmp(effect_name, NvAR_Feature_FaceBoxDetection))
    obj = new FaceDetectionApp;
  else if (!strcasecmp(effect_name, NvAR_Feature_LandmarkDetection))
    obj = new FacialLandmarksApp;
  else
    return nullptr;
  obj->m_effectName = effect_name;
  return obj;
}

NvCV_Status BatchProcessVideos() {
  NvCV_Status err = NVCV_SUCCESS;
  unsigned num_videos = (unsigned)FLAG_inSrcVideoFiles.size();
  std::unique_ptr<BaseApp> app(BaseApp::Create(FLAG_effect.c_str()));
  cv::Mat cv_img;
  std::vector<cv::Mat> frames(num_videos), frames_t_1(num_videos);
  NvCVImage nv_img;
  unsigned src_video_width = 0, src_video_height = 0;
  std::vector<cv::VideoCapture> list_of_captures(num_videos);
  std::vector<cv::Mat> src_img_buffer(num_videos);
  std::vector<cv::VideoWriter> list_of_writers(num_videos);
  std::vector<unsigned> batch_indices(num_videos);
  double fps;

  BAIL_IF_FALSE(app != nullptr, err, NVCV_ERR_UNIMPLEMENTED);
  BAIL_IF_FALSE(num_videos > 0, err, NVCV_ERR_MISSINGINPUT);

  for (unsigned i = 0; i < num_videos; i++) {
    list_of_captures[i].open(FLAG_inSrcVideoFiles[i]);
    if (!list_of_captures[i].isOpened()) {
      printf("Error: Could not open %s.\n", FLAG_inSrcVideoFiles[i]);
      return NVCV_ERR_READ;
    }
    list_of_captures[i] >> cv_img;
    if (cv_img.empty()) {
      printf("Error: Could not read %s.\n", FLAG_inSrcVideoFiles[i]);
      return NVCV_ERR_READ;
    }
    if (i == 0) {
      src_video_width = cv_img.cols;
      src_video_height = cv_img.rows;
    } else if (int(src_video_width) != cv_img.cols && int(src_video_height) != cv_img.rows) {
      printf("Error: Resolution of the videos must be same.\n");
      return NVCV_ERR_MISMATCH;
    }
    list_of_captures[i].set(cv::CAP_PROP_POS_FRAMES, 0);
  }

  BAIL_IF_ERR(err = app->Init(num_videos));                                    // Init effect
  BAIL_IF_ERR(err = app->AllocateBuffers(src_video_width, src_video_height));  // Allocate buffers
  BAIL_IF_ERR(err = app->SetParametersBeforeLoad());                           // Set IO and config
  BAIL_IF_ERR(err = app->Load());                                              // Load the feature
  BAIL_IF_ERR(err = app->SetParametersAfterLoad());                            // Set IO and config

  for (unsigned i = 0; i < num_videos; i++) {
    if (!list_of_captures[i].isOpened()) continue;  // if video is not opened, we skip
    list_of_captures[i] >> frames[i];
    if (frames[i].empty())                   // if nothing read
      list_of_captures[i].release();         // closing the video
    else                                     // if a frame is read
      BAIL_IF_ERR(app->InitVideoStream(i));  // initialize video stream
  }

  // Open video writers
  for (unsigned i = 0; i < num_videos; i++) {
    size_t period_loc = std::string(FLAG_inSrcVideoFiles[i]).find_last_of(".");
    std::string dst_video = std::string(FLAG_inSrcVideoFiles[i]).substr(0, period_loc);
    dst_video = dst_video + "_" + FLAG_outputNameTag + ".mp4";
    fps = list_of_captures[i].get(cv::CAP_PROP_FPS);

    list_of_writers[i].open(dst_video, cv::VideoWriter::fourcc('a', 'v', 'c', '1'), fps,
                            cv::Size(app->m_outputImgVizWidth, app->m_outputImgVizHeight));
    if (!list_of_writers[i].isOpened()) {
      printf("Error: Could not open video writer for video %s.\n", dst_video.c_str());
      return NVCV_ERR_WRITE;
    }
  }

  while (1) {
    // Read inputs
    unsigned batchsize = 0;  // batchsize = number of active videos
    for (unsigned i = 0; i < num_videos; i++) {
      if (list_of_captures[i].isOpened()) {
        list_of_captures[i] >> frames_t_1[i];  // Reading the next frame to know if the video has ended
                                               // as it is not possible to know if current frame is last
                                               // without reading the next frame
        if (frames_t_1[i].empty()) {
          BAIL_IF_ERR(app->ReleaseVideoStream(i));  // Trition requires NvAR_DeallocateState() is called just before the
                                                    // last inference for that video stream
          list_of_captures[i].release();            // closing the video
        }
      }
      if (frames[i].empty()) continue;
      NVWrapperForCVMat(&frames[i], &nv_img);
      BAIL_IF_ERR(err = TransferToNthImage(batchsize, &nv_img, &app->m_srcVidFrame, 1, app->m_cudaStream, &app->m_stg));
      batch_indices[batchsize] = i;  // storing video indices for creating output videos
      batchsize++;                   // counting the number of active videos
    }
    if (batchsize == 0) goto bail;  // if all videos have ended

    // Run batch
    BAIL_IF_ERR(err = app->Run(batch_indices.data(), batchsize));

    // Write Output
    for (unsigned i = 0; i < batchsize; i++) {
      unsigned video_idx = batch_indices[i];
      cv::Mat display_frame;
      BAIL_IF_ERR(err = app->GenerateNthOutputVizImage(i, frames[video_idx], display_frame));
      if (!display_frame.empty()) {
        list_of_writers[video_idx] << display_frame;
      }
      frames[video_idx] = frames_t_1[video_idx].clone();  // copying the t+1 frame to current frame
    }

    // Update current frame
    for (unsigned i = 0; i < batchsize; i++) {
      unsigned video_idx = batch_indices[i];
      frames[video_idx] = frames_t_1[video_idx].clone();  // copying the t+1 frame to current frame
                                                          // for the videos that were processed
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
