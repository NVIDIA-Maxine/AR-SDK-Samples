/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef __GAZE_ENGINE__
#define __GAZE_ENGINE__

#include <random>
#include "featureVertexName.h"
#include "nvAR.h"
#include "nvCVOpenCV.h"

class KalmanFilter1D {
 private:
  float Q_;          // Covariance of the process noise
  float xhat_;       // Current prediction
  float xhatminus_;  // Previous prediction
  float P_;          // Estimated accuracy of xhat_
  float Pminus_;     // Previous P_
  float K_;          // Kalman gain
  float R_;          // Covariance of the observation noise
  bool bFirstUse;

 public:
  KalmanFilter1D() { reset(); }

  KalmanFilter1D(float Q, float R) { reset(Q, R); }

  void reset() {
    R_ = 0.005f * 0.005f;
    Q_ = 1e-5f;
    xhat_ = 0.0f;
    xhatminus_ = 0.0f;
    P_ = 1;
    bFirstUse = true;
    Pminus_ = 0.0f;
    K_ = 0.0f;
  }

  void reset(float Q, float R) {
    reset();
    Q_ = Q;
    R_ = R;
  }

  float update(float val) {
    if (bFirstUse) {
      xhat_ = val;
      bFirstUse = false;
    }

    xhatminus_ = xhat_;
    Pminus_ = P_ + Q_;
    K_ = Pminus_ / (Pminus_ + R_);
    xhat_ = xhatminus_ + K_ * (val - xhatminus_);
    P_ = (1 - K_) * Pminus_;

    return xhat_;
  }
};

bool CheckResult(NvCV_Status nvErr, unsigned line);

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

typedef struct LandmarksProperties {
  int numPoints;
  float confidence_threshold;
} LandmarksProperties;

/********************************************************************************
 * GazeEngine
 ********************************************************************************/

class GazeEngine {
 public:
  enum Err { errNone, errGeneral, errRun, errInitialization, errRead, errEffect, errParameter, errNoFace };
  unsigned int input_image_width, input_image_height, input_image_pitch;
  const LandmarksProperties LANDMARKS_INFO[2] = {{68, 0.15f},  // number of landmark points, confidence threshold value
                                                 {126, 0.15f}};

  void setInputImageWidth(int width) { input_image_width = (unsigned int)width; }
  void setInputImageHeight(int height) { input_image_height = (unsigned int)height; }

  Err createGazeRedirectionFeature(const char* modelPath, unsigned int _batchSize = 1);
  void destroyGazeRedirectionFeature();
  Err initGazeRedirectionIOParams();

  unsigned findFaceBoxes();
  NvAR_Rect* getLargestBox();
  NvCV_Status findLandmarks();
  NvAR_BBoxes* getBoundingBoxes();
  
  /**
   * Landmarks corresponding to facial keypoints
   *
   * @returns Pointer to the landmarks array 
   */
  NvAR_Point2f* getLandmarks();

  /**
   * Output landmarks corresponding to the redirected eyes from the gaze redirection network
   *
   * @returns Pointer to the landmarks array
   */
  NvAR_Point2f* getGazeOutputLandmarks();

  NvAR_Quaternion* getPose();
  float* getHeadTranslation();
  float* getGazeVector();
  float* getLandmarksConfidence();
  float getAverageLandmarksConfidence();
  void enlargeAndSquarifyImageBox(float enlarge, NvAR_Rect& box, int FLAG_variant);
  unsigned findLargestFaceBox(NvAR_Rect& faceBox, int variant = 0);
  unsigned acquireFaceBox(cv::Mat& src, NvAR_Rect& faceBox, int variant = 0);
  unsigned acquireFaceBoxAndLandmarks(cv::Mat& src, NvAR_Point2f* refMarks, NvAR_Rect& faceBox);
  Err acquireGazeRedirection(cv::Mat& frame, cv::Mat& outputFrame);
  NvAR_RenderingParams* getRenderingParams();
  void setFaceStabilization(bool);
  Err setNumLandmarks(int);
  void setGazeRedirect(bool _bGazeRedirect);
  void setUseCudaGraph(bool _bUseCudaGraph);
  void setEyeSizeSensitivity(unsigned);
  // Set and get functions for look away
  void setEnableLookAway(unsigned);  // Set the enable look away parameter. 
  void setLookAwayOffsetMax(unsigned); // Set the look away offset max.
  void setLookAwayIntervalRange(unsigned); // Set the maximum lookaway interval.
  void setLookAwayIntervalMin(unsigned); // Set the minimum lookaway interval.
  void setGazePitchThresholdLow(float); // Set the lower threshold for gaze pitch transition.
  void setGazeYawThresholdLow(float);  // Set the lower threshold for gaze yaw transition.
  void setHeadPitchThresholdLow(float);  // Set the lower threshold for head pitch transition.
  void setHeadYawThresholdLow(float);    // Set the lower threshold for head yaw transition.
  void setGazePitchThresholdHigh(float);  // Set the higher threshold for gaze pitch transition.
  void setGazeYawThresholdHigh(float);    // Set the higher threshold for gaze yaw transition.
  void setHeadPitchThresholdHigh(float);  // Set the higher threshold for head pitch transition.
  void setHeadYawThresholdHigh(float);    // Set the higher threshold for head yaw transition.
  unsigned getEnableLookAway();       // Get the enable look away parameter.
  unsigned getLookAwayOffsetMax();  // Get the look away offset max.
  unsigned getLookAwayIntervalRange();  // Get the maximum lookaway interval.
  unsigned getLookAwayIntervalMin();  // Get the minimum lookaway interval.
  float getGazePitchThresholdLow();  // Get the lower threshold for gaze pitch transition.
  float getGazeYawThresholdLow();   // Get the lower threshold for gaze yaw transition.
  float getHeadPitchThresholdLow();  // Get the lower threshold for head pitch transition.
  float getHeadYawThresholdLow();   // Get the lower threshold for head yaw transition.
  float getGazePitchThresholdHigh();  // Get the higher threshold for gaze pitch transition.
  float getGazeYawThresholdHigh();    // Get the higher threshold for gaze yaw transition.
  float getHeadPitchThresholdHigh();  // Get the higher threshold for head pitch transition.
  float getHeadYawThresholdHigh();    // Get the higher threshold for head yaw transition.

  GazeEngine::Err setLookAwayParameters(); // Set all look away parameters in gaze feature. 
 // Toggle the state of enable look away parameter.
  GazeEngine::Err toggleEnableLookAway();
  // Increment or decrement look away parameters
  GazeEngine::Err incrementLookAwayOffsetMax();
  GazeEngine::Err decrementLookAwayOffsetMax();

  int getNumLandmarks() { return numLandmarks; }
  int getNumGazeOutputLandmarks() { return num_output_landmarks; }
  void DrawPose(const cv::Mat& src, const NvAR_Quaternion* pose) const;
  std::array<float, 2> GetAverageLandmarkPositionInGlSpace() const;
  void DrawEstimatedGaze(const cv::Mat& src);
  NvAR_Point3f* getGazeDirectionPoints();

  NvCVImage inputImageBuffer{}, tmpImage{}, outputImageBuffer{};

  NvAR_FeatureHandle faceDetectHandle{}, landmarkDetectHandle{}, gazeRedirectHandle{};
  std::vector<NvAR_Point2f> facial_landmarks;
  std::vector<NvAR_Point2f> gaze_output_landmarks;
  std::vector<float> facial_landmarks_confidence;
  NvAR_Point3f gaze_direction[2] = {{0.f, 0.f, 0.f}};
  NvAR_Quaternion head_pose;
  float gaze_angles_vector[2] = {0.f};
  float head_translation[3] = {0.f};
  NvAR_RenderingParams* rendering_params{};
  CUstream stream{};
  std::vector<NvAR_Rect> output_bbox_data;
  std::vector<float> output_bbox_conf_data;
  NvAR_BBoxes output_bboxes{};
  int batchSize;
  std::mt19937 ran;
  int numLandmarks;
  int num_output_landmarks = 24;
  int eyeSizeSensitivity;
  unsigned int lookAwayOffsetMax, lookAwayIntervalRange, lookAwayIntervalMin;
  float gazePitchThresholdLow, gazeYawThresholdLow, headPitchThresholdLow,
      headYawThresholdLow;
  float gazePitchThresholdHigh, gazeYawThresholdHigh, headPitchThresholdHigh, headYawThresholdHigh;
  const unsigned int lookAwayOffsetLimit = 10;
  float confidenceThreshold;
  std::string face_model;

  bool bStabilizeFace;
  bool bUseOTAU;
  bool bGazeRedirect;
  bool bUseCudaGraph;
  bool bEnableLookAway;
  char *fdOTAModelPath, *ldOTAModelPath;

  GazeEngine() {
    batchSize = 1;
    bStabilizeFace = true;
    bGazeRedirect = true;
    bUseCudaGraph = true;
    numLandmarks = LANDMARKS_INFO[0].numPoints;
    num_output_landmarks = 12;
    confidenceThreshold = LANDMARKS_INFO[0].confidence_threshold;
    input_image_width = 640;
    input_image_height = 480;
    input_image_pitch = 3 * input_image_width * sizeof(unsigned char);  // RGB
    bUseOTAU = false;
    fdOTAModelPath = NULL;
    ldOTAModelPath = NULL;
    eyeSizeSensitivity = 3;
    lookAwayOffsetMax = 5;
    lookAwayIntervalRange = 250;
    lookAwayIntervalMin = 100;
    gazePitchThresholdLow = 20.0;
    gazePitchThresholdHigh = 25.0;
    gazeYawThresholdLow = 30.0;
    gazeYawThresholdHigh = 35.0;
    headPitchThresholdLow = 25.0;
    headPitchThresholdHigh = 30.0;
    headYawThresholdLow = 30.0;
    headYawThresholdHigh = 35.0;
  }
};
#endif
