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

#include <assert.h>
#include "socket_client.h"

namespace socket_communication {
Client::Client() {}
Client::Client(const std::string ip, unsigned short port) { Init(ip, port); }
Client::~Client() { closesocket(client_); WSACleanup(); }

void Client::Init(const std::string ip, unsigned short port) {
  client_ = socket(AF_INET, SOCK_STREAM, 0);
  if (client_ < 0) {
    std::cout << "[Client]: ERROR establishing socket\n" << std::endl;
    exit(1);
  }

  bool connected = false;
  int connection_attempts = 5;

  while ((!connected) && (connection_attempts > 0)) {
    WSAStartup(MAKEWORD(2, 0), &wsa_data);
    client_ = socket(AF_INET, SOCK_STREAM, 0);
    InetPton(AF_INET, "127.0.0.1", &addr.sin_addr.s_addr);

    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<u_short>(port));

    // TODO: verify connection
    connect(client_ , reinterpret_cast<SOCKADDR*>(&addr), sizeof(addr));
    connected = true;
    std::cout << "[Client]: Cpp socket client connected." << std::endl;      
  }
}

void Client::Send(std::string message) {
  // Send length of the message
  size_t length = message.length();
  std::string length_str = std::to_string(length);
  std::string message_length =
      std::string(size_message_length_ - length_str.length(), '0') + length_str;
  send(client_, message_length.c_str(), size_message_length_, 0);

  // Send message
  send(client_, message.c_str(), static_cast<int>(length), 0);
}

std::string Client::Receive() {
  // TODO: try catch, if connection dropped print notification and try
  // to reconnect

  // Receive length of the message
  char message_length[16] = {0};
  int n = recv(client_, message_length, size_message_length_, 0);
  std::string message_length_string(message_length);
  int length = std::stoi(message_length_string);
  // if (length == 0) return "";

  // receive message
  char message[16] = {0};
  n = recv(client_, message, length, 0);
  return message;
}

void Client::ReceivePing() {
  char message_length[16] = {0};
  recv(client_, message_length, size_message_length_, 0);
}

void Client::SendFloatArr(float *floatvec, size_t arr_size) {
  size_t buf_size = arr_size*sizeof(float);

  size_t length = buf_size;
  std::string length_str = std::to_string(length);
  std::string message_length =
      std::string(size_message_length_ - length_str.length(), '0') + length_str;

  send(client_, (char*)floatvec, static_cast<int>(length), 0);
}

void Client::SendKeyPoints(NvAR_Point3f* keypoints, int numKeyPoints) {
  size_t buf_size = numKeyPoints * 3 * sizeof(float);
  send(client_, (char*)keypoints, static_cast<int>(buf_size), 0);
}

void Client::SendIntVec(std::vector<int> intvec) {
  size_t buf_size = intvec.size()*sizeof(int);
  unsigned char buf[10]; 
  std::memcpy(buf, intvec.data(), buf_size);

  size_t length = buf_size;
  std::string length_str = std::to_string(length);
  std::string message_length =
      std::string(size_message_length_ - length_str.length(), '0') + length_str;

  send(client_, (char*)buf, static_cast<int>(length), 0);
}

}
