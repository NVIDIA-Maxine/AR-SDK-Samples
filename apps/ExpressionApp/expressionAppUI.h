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

#pragma once

#if _ENABLE_UI
#include <set>
#include <atomic>
#include <thread>
#include <mutex>
#include <string>
#include <vector>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <misc/cpp/imgui_stdlib.h>

enum exprType {
  BROW = 1,
  CHEEK,
  EYE,
  JAW,
  MOUTH,
  NOSE
};

enum {
  DISPLAY_MESH = (1 << 0),
  DISPLAY_IMAGE = (1 << 1),
  DISPLAY_PLOT = (1 << 2),
  DISPLAY_LM = (1 << 3)
};

struct ExpressionState {
  int                           input_filter;
  float                         global_parameter;
  int                           expr_mode;
  unsigned long                 counter;
  bool                          calibrate;
  bool                          uncalibrate;
  bool                          landmark_display;
  bool                          mesh_display;
  bool                          image_display;
  bool                          bargraph_display;
  bool                          show_fps;
  bool                          kill_app_;
  std::vector<float>            expr;
  std::vector<float>            expr_scale;
  std::vector<float>            expr_offset;
  std::vector<float>            expr_exponent;
  ExpressionState() {
    input_filter = -1;
    counter = 0;
    global_parameter = 1.0f;
    expr_mode = 1;
    calibrate = false;
    uncalibrate = false;
    landmark_display = false;
    mesh_display = false;
    image_display = false;
    bargraph_display = false;
    show_fps = false;
    kill_app_ = false;
  }
};

class ExpressionAppUI {
public:
  //============= State Management =================
  void init(int numExpr, int filter, int exprMode, int display, int showFPS);
  void cleanup();
  
  void stateQuerybyCore(unsigned int& displayMode, unsigned int& exprMode, unsigned int& filter, bool& calibrate, bool& uncalibrate, bool& showFPS,
     float& globalParam, std::vector<float>& expressionOffset, std::vector<float>& expressionScale, std::vector<float>& expressionExponent, bool& killApp);
  void stateSetbyCore(std::vector<float> expression,
    std::vector<float> expressionOffset, std::vector<float> expressionScale, std::vector<float> expressionExponent, bool isCalibrated = false, int key = -1);

private:
  
  void uiRenderThread();
  void getStateFromLocal();
  void setStateToLocal();
  //============= UI calls ====================
  void CreateUIElements(); // main API to create UI compoenents
  void showMLPSetting();
  void showStreamingSetting();
  void showFilterSetting();
  void showExpressionPane();
  void showExpressionWindow();
  void showCalibrationSetting();
  void showImageDisplaySettings();
  void showLandmarkOption();
  void showFPSSetting();
  void showSaveSettingsOption();
  void saveConfigToFile();
  void loadConfgFromFile(const char* filePath = NULL);
  void openFileLoadSettings();
  void closeAppSettings();
  void checkForKeyInput();
  //================= Filter =======================
  bool filter_face_box_;
  bool filter_face_landmark_;
  bool filter_face_rot_pose_;
  bool filter_face_expr_;
  bool filter_face_gaze_;
  bool filter_enhance_expr_;

  //================= Expression ===================
  bool show_expr_;
  bool brow_expr_;
  bool cheek_expr_;
  bool eye_expr_ ;
  bool jaw_expr_ ;
  bool mouth_expr_;
  bool nose_expr_;

  std::set<int>      ui_expression_list_;
  //=================================================
  std::atomic_bool  omniverse_interface_window_;
  std::atomic_int   keyboard_input_;
  bool  load_from_file_;
  bool  show_filter_window_;
  int   expr;

  ExpressionState  ui_state_;
  ExpressionState  curr_state_;
  std::atomic_bool ui_keep_running_;
  std::thread      ui_thread_;
  std::mutex       ui_mutex_;
  std::string file_name_;
  unsigned long internal_get_state_counter_;
  unsigned long internal_set_state_counter_;
  int32_t     num_expressions_;
};
#endif