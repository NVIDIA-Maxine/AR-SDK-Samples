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

#define _CRT_SECURE_NO_DEPRECATE

#include <sys/stat.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>

#include "wave.h"

#ifdef _WIN32
#include <Shlwapi.h>
#include <io.h>
#pragma comment(lib, "Shlwapi.lib")
#else
#include <unistd.h>
#endif

#include "waveReadWrite.h"
// #include <misc.hpp>

const float* CWaveFileRead::GetFloatPCMData() {
  int8_t* audioDataPtr = reinterpret_cast<int8_t*>(m_WaveData.get());

  if (m_floatWaveData.size()) return m_floatWaveData.data();

  m_floatWaveData.resize(m_nNumSamples);
  float* outputWaveData = m_floatWaveData.data();

  if (m_WaveFormatEx.wFormatTag == WAVE_FORMAT_IEEE_FLOAT) {
    memcpy(outputWaveData, audioDataPtr, m_nNumSamples * sizeof(float));
    return outputWaveData;
  }

  for (uint32_t i = 0; i < m_nNumSamples; i++) {
    switch (m_WaveFormatEx.wBitsPerSample) {
      case 8: {
        uint8_t audioSample = *(reinterpret_cast<uint8_t*>(audioDataPtr));
        outputWaveData[i] = (audioSample - 128) / 128.0f;
      } break;
      case 16: {
        int16_t audioSample = *(reinterpret_cast<int16_t*>(audioDataPtr));
        outputWaveData[i] = audioSample / 32768.0f;
      } break;
      case 24: {
        int32_t audioSample = *(reinterpret_cast<int32_t*>(audioDataPtr));
        uint8_t data0 = audioSample & 0x000000ff;
        uint8_t data1 = static_cast<uint8_t>((audioSample & 0x0000ff00) >> 8);
        uint8_t data2 = static_cast<uint8_t>((audioSample & 0x00ff0000) >> 16);
        int32_t Value = ((data2 << 24) | (data1 << 16) | (data0 << 8)) >> 8;
        outputWaveData[i] = Value / 8388608.0f;
      } break;
      case 32: {
        int32_t audioSample = *(reinterpret_cast<int32_t*>(audioDataPtr));
        outputWaveData[i] = audioSample / 2147483648.0f;
      } break;
    }
    audioDataPtr += m_WaveFormatEx.nBlockAlign;
  }

  return outputWaveData;
}

const float* CWaveFileRead::GetFloatPCMDataAligned(int alignSamples) {
  if (!GetFloatPCMData()) return nullptr;

  int totalAlignedSamples;
  if (!(m_nNumSamples % alignSamples))
    totalAlignedSamples = m_nNumSamples;
  else
    totalAlignedSamples = m_nNumSamples + (alignSamples - (m_nNumSamples % alignSamples));

  m_floatWaveDataAligned.reset(new float[totalAlignedSamples]());

  for (uint32_t i = 0; i < m_nNumSamples; i++) m_floatWaveDataAligned[i] = m_floatWaveData[i];

  m_NumAlignedSamples = totalAlignedSamples;
  return m_floatWaveDataAligned.get();
}

int CWaveFileRead::GetBitsPerSample() {
  if (m_WaveFormatEx.wBitsPerSample == 0) assert(0);

  return m_WaveFormatEx.wBitsPerSample;
}

const RiffChunk* CWaveFileRead::FindChunk(const uint8_t* data, size_t sizeBytes, uint32_t fourcc) {
  if (!data) return nullptr;

  const uint8_t* ptr = data;
  const uint8_t* end = data + sizeBytes;

  while (end > (ptr + sizeof(RiffChunk))) {
    const RiffChunk* header = reinterpret_cast<const RiffChunk*>(ptr);
    if (header->chunkId == fourcc) return header;

    ptr += (header->chunkSize + sizeof(RiffChunk));
  }

  return nullptr;
}

CWaveFileRead::CWaveFileRead(std::string wavFile)
    : m_wavFile(wavFile), m_nNumSamples(0), validFile(false), m_WaveDataSize(0), m_NumAlignedSamples(0) {
  memset(&m_WaveFormatEx, 0, sizeof(m_WaveFormatEx));
#ifdef __linux__
  if (access(m_wavFile.c_str(), R_OK) == 0)
#else
  if (PathFileExistsA(m_wavFile.c_str()))
#endif
  {
    if (readPCM(m_wavFile.c_str()) == 0) validFile = true;
  }
}

inline bool loadFile(std::string const& infilename, std::string* outData) {
  std::string result;
  std::string filename = infilename;

  errno = 0;
  std::ifstream stream(filename.c_str(), std::ios::binary | std::ios::in);
  if (!stream.is_open()) {
    return false;
  }

  stream.seekg(0, std::ios::end);
  result.reserve(stream.tellg());
  stream.seekg(0, std::ios::beg);

  result.assign((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());

  *outData = result;

  return true;
}

inline std::string loadFile(const std::string& infilename) {
  std::string result;
  loadFile(infilename, &result);

  return result;
}

int CWaveFileRead::readPCM(const char* szFileName) {
  std::string fileData;
  if (loadFile(std::string(szFileName), &fileData) != true) {
    return -1;
  }

  const uint8_t* waveData = reinterpret_cast<const uint8_t*>(fileData.data());
  size_t waveDataSize = fileData.length();
  const uint8_t* waveEnd = waveData + waveDataSize;

  // Locate RIFF 'WAVE'
  const RiffChunk* riffChunk = FindChunk(waveData, waveDataSize, MAKEFOURCC('R', 'I', 'F', 'F'));
  if (!riffChunk || riffChunk->chunkSize < 4) {
    return -1;
  }

  const RiffHeader* riffHeader = reinterpret_cast<const RiffHeader*>(riffChunk);
  if (riffHeader->fileTag != MAKEFOURCC('W', 'A', 'V', 'E')) {
    return -1;
  }

  // Locate 'fmt '
  const uint8_t* ptr = reinterpret_cast<const uint8_t*>(riffHeader) + sizeof(RiffHeader);
  if ((ptr + sizeof(RiffChunk)) > waveEnd) {
    return -1;
  }

  const RiffChunk* fmtChunk = FindChunk(ptr, riffHeader->chunkSize, MAKEFOURCC('f', 'm', 't', ' '));
  if (!fmtChunk || fmtChunk->chunkSize < sizeof(waveFormat_basic)) {
    return -1;
  }

  ptr = reinterpret_cast<const uint8_t*>(fmtChunk) + sizeof(RiffChunk);
  if (ptr + fmtChunk->chunkSize > waveEnd) {
    return -1;
  }

  const waveFormat_basic_nopcm* wf = reinterpret_cast<const waveFormat_basic_nopcm*>(ptr);

  if (!(wf->formatTag == WAVE_FORMAT_PCM || wf->formatTag == WAVE_FORMAT_IEEE_FLOAT)) {
    if (wf->formatTag == WAVE_FORMAT_EXTENSIBLE) {
      printf("WAVE_FORMAT_EXTENSIBLE is not supported. Please convert\n");
    }

    return -1;
  }

  ptr = reinterpret_cast<const uint8_t*>(riffHeader) + sizeof(RiffHeader);
  if ((ptr + sizeof(RiffChunk)) > waveEnd) {
    return -1;
  }

  const RiffChunk* dataChunk = FindChunk(ptr, riffChunk->chunkSize, MAKEFOURCC('d', 'a', 't', 'a'));
  if (!dataChunk || !dataChunk->chunkSize) {
    return -1;
  }

  ptr = reinterpret_cast<const uint8_t*>(dataChunk) + sizeof(RiffChunk);
  if (ptr + dataChunk->chunkSize > waveEnd) {
    return -1;
  }

  m_WaveData = std::unique_ptr<uint8_t[]>(new uint8_t[dataChunk->chunkSize]);
  m_WaveDataSize = dataChunk->chunkSize;
  memcpy(m_WaveData.get(), ptr, dataChunk->chunkSize);
  if (wf->formatTag == WAVE_FORMAT_PCM) {
    memcpy(&m_WaveFormatEx, reinterpret_cast<const waveFormat_basic*>(wf), sizeof(waveFormat_basic));
    m_WaveFormatEx.cbSize = 0;
  } else {
    memcpy(&m_WaveFormatEx, reinterpret_cast<const waveFormat_ext*>(wf), sizeof(waveFormat_ext));
  }

  m_nNumSamples = m_WaveDataSize / (m_WaveFormatEx.nBlockAlign / m_WaveFormatEx.nChannels);

  return 0;
}

CWaveFileWrite::CWaveFileWrite(std::string wavFile, uint32_t samplesPerSec, uint32_t numChannels,
                               uint16_t bitsPerSample, bool isFloat)
    : m_wavFile(wavFile) {
  wfx.wFormatTag = isFloat ? WAVE_FORMAT_IEEE_FLOAT : WAVE_FORMAT_PCM;
  wfx.nChannels = static_cast<uint16_t>(numChannels);
  wfx.nSamplesPerSec = samplesPerSec;
  wfx.nBlockAlign = static_cast<uint16_t>((numChannels * bitsPerSample) / 8);
  wfx.nAvgBytesPerSec = samplesPerSec * wfx.nBlockAlign;
  wfx.wBitsPerSample = bitsPerSample;
  wfx.cbSize = 0;

  m_validState = true;
}

CWaveFileWrite::~CWaveFileWrite() {
  if (m_commitDone == false) commitFile();

  if (m_fp) {
    fclose(m_fp);
    m_fp = nullptr;
  }
}

bool CWaveFileWrite::initFile() {
  if (!m_fp) {
    errno = 0;
    m_fp = fopen(m_wavFile.c_str(), "wb");
    if (!m_fp) return false;

    int64_t offset = sizeof(RiffHeader) + sizeof(RiffChunk) + sizeof(waveFormat_basic) + sizeof(RiffChunk);
    if (fseek(m_fp, static_cast<long>(offset), SEEK_SET) != 0) {
      fclose(m_fp);
      m_fp = nullptr;
      return false;
    }
  }

  return true;
}

bool CWaveFileWrite::writeChunk(const void* data, uint32_t len) {
  if (!m_validState) {
    return false;
  }

  if (!initFile()) {
    return false;
  }

  size_t written = fwrite(data, len, 1, m_fp);
  if (written != 1) return false;

  m_cumulativeCount += len;
  return true;
}

bool CWaveFileWrite::commitFile() {
  if (!m_validState) return false;

  if (!m_fp) return false;

  // pull fp to start of file to write headers.
  fseek(m_fp, 0, SEEK_SET);

  // write the riff chunk header
  uint32_t fmtChunkSize = sizeof(waveFormat_basic);
  RiffHeader riffHeader;
  riffHeader.chunkId = MAKEFOURCC('R', 'I', 'F', 'F');
  riffHeader.chunkSize = 4 + sizeof(RiffChunk) + sizeof(RiffChunk) + fmtChunkSize + m_cumulativeCount;
  riffHeader.fileTag = MAKEFOURCC('W', 'A', 'V', 'E');
  if (fwrite(&riffHeader, sizeof(riffHeader), 1, m_fp) != 1) return false;

  // fmt riff chunk
  RiffChunk fmtChunk;
  fmtChunk.chunkId = MAKEFOURCC('f', 'm', 't', ' ');
  fmtChunk.chunkSize = sizeof(waveFormat_basic);
  if (fwrite(&fmtChunk, sizeof(RiffChunk), 1, m_fp) != 1) return false;

  // fixme: try using WAVEFORMATEX for size
  if (fwrite(&wfx, sizeof(waveFormat_basic), 1, m_fp) != 1) return false;

  // data riff chunk
  RiffChunk dataChunk;
  dataChunk.chunkId = MAKEFOURCC('d', 'a', 't', 'a');
  dataChunk.chunkSize = m_cumulativeCount;
  if (fwrite(&dataChunk, sizeof(RiffChunk), 1, m_fp) != 1) return false;

  fclose(m_fp);
  m_fp = nullptr;

  m_commitDone = true;
  m_validState = false;
  return true;
}

std::vector<float>* CWaveFileRead::GetFloatVector() {
  if (!m_floatWaveData.size()) (void)GetFloatPCMData();
  return &m_floatWaveData;
}

std::map<std::string, std::unique_ptr<CWaveFileRead>> read_file_cache;
bool ReadWavFile(const std::string& filename, uint32_t expected_sample_rate, int expected_num_channels,
                 std::vector<float>** data, unsigned* original_num_samples, std::vector<int>* file_end_offset,
                 int align_samples, bool enable_debug) {
  std::vector<std::string> files;

  const std::string kDelimiter = ";";

  auto delim = filename.find(kDelimiter);
  if (delim != std::string::npos) {
    unsigned int start = 0;
    do {
      files.push_back(filename.substr(start, delim - start));
      start = delim + 1;
    } while ((delim = filename.find(kDelimiter, delim + 1)) != std::string::npos);
    if (start < filename.length()) files.push_back(filename.substr(start));
  } else {
    files.push_back(filename);
  }

  *original_num_samples = 0;

  int offset = 0;
  std::vector<float>* ret = nullptr;
  for (auto& file : files) {
    CWaveFileRead* wave_file = nullptr;
    auto cached_file = read_file_cache.find(file);
    if (cached_file == read_file_cache.end()) {
      wave_file = new CWaveFileRead(file);
      read_file_cache.emplace(file, std::unique_ptr<CWaveFileRead>(wave_file));
    } else {
      wave_file = cached_file->second.get();
    }

    if (wave_file->isValid() == false) {
      std::cerr << "Invalid wave file" << std::endl;
      delete ret;
      *data = nullptr;
      return false;
    }

    if (enable_debug) {
      std::cout << "Total number of samples: " << wave_file->GetNumSamples() << std::endl;
      std::cout << "Size in bytes: " << wave_file->GetRawPCMDataSizeInBytes() << std::endl;
      std::cout << "Sample rate: " << wave_file->GetSampleRate() << std::endl;
      std::cout << "Number of Channels : " << wave_file->GetWaveFormat().nChannels << std::endl;

      auto bits_per_sample = wave_file->GetBitsPerSample();
      std::cout << "Bits/sample: " << bits_per_sample << std::endl;
    }

    if (wave_file->GetSampleRate() != expected_sample_rate) {
      std::cerr << "Sample rate mismatch. Sample rate of file " << filename << ": " << wave_file->GetSampleRate()
                << "v/s expected value: " << expected_sample_rate << std::endl;
      delete ret;
      *data = nullptr;
      return false;
    }
    if (wave_file->GetWaveFormat().nChannels != expected_num_channels) {
      std::cerr << "Channel count needs to be " << expected_num_channels << std::endl;
      delete ret;
      *data = nullptr;
      return false;
    }

    *original_num_samples += wave_file->GetNumSamples();

    int num_samples;
    if (align_samples != -1) {
      uint32_t pad = align_samples - (wave_file->GetNumSamples() % align_samples);
      num_samples = wave_file->GetNumSamples() + pad;
    } else {
      num_samples = wave_file->GetNumSamples();
    }

    if (file_end_offset) {
      offset += num_samples;
      file_end_offset->push_back(offset);
    }

    if (files.size() > 1) {
      // If using reset, ignore cache
      // Vector is not resized here, will be resized at end
      auto local = wave_file->GetFloatVector();
      if (!ret) {
        ret = new std::vector<float>();
        *data = ret;
      }
      ret->insert(ret->end(), local->begin(), local->end());
    } else {
      *data = wave_file->GetFloatVector();
      // Align if using multiple inputs
      (*data)->resize(num_samples, 0.f);
    }
  }

  if (files.size() > 1 && align_samples != -1) {
    // Align if using multiple inputs
    uint32_t pad = ret->size() % align_samples;
    if (pad) ret->resize(ret->size() + pad, 0.f);
  }
  return true;
}
