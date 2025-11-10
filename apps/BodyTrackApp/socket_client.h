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

#include <chrono>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#ifdef _MSC_VER
  #include <WinSock2.h>
  #include <ws2tcpip.h>
#endif // _MSC_VER
#include "nvAR_defs.h"

#pragma comment(lib, "ws2_32.lib")

namespace socket_communication {
class Client {
 public:
  Client();
  Client(const std::string ip, unsigned short port);
  ~Client();

  void Init(const std::string ip = "127.0.0.1", unsigned short port = 5001);

  void Send(std::string message);

  void SendIntVec(std::vector<int> intvec);
  void SendFloatArr(float *floatvec, size_t arr_size);
  void SendKeyPoints(NvAR_Point3f* keypoints, int numKeyPoints);

  std::string Receive();
  void ReceivePing();

 private:
  const int size_message_length_ = 16;  // Buffer size for the length
#ifdef _MSC_VER
  SOCKET client_;
  WSADATA wsa_data;
  SOCKADDR_IN addr;
#endif // _MSC_VER
};

} // namespace socket_communication
