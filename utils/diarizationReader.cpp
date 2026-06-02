/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "diarizationReader.h"

#include <algorithm>
#include <fstream>

#include "nlohmann/json.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Macros                                                                                                             //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// clang-format off
#define BAIL_IF_ERR(err)              do { if (0 != (err)) { goto bail; } } while (0)
#define BAIL_IF_FALSE(x, err, code)   do { if (!(x)) { err = code; goto bail; } } while (0)
#define BAIL(err, code)               do { err = code; goto bail; } while (0)
// clang-format on

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Member definitions                                                                                               ///
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

NvCV_Status DiarizationReader::Load(const std::string& file_path) {
  NvCV_Status err = NVCV_SUCCESS;

  std::ifstream file(file_path);
  BAIL_IF_FALSE(file.is_open(), err, NVCV_ERR_FILE);

  {
    nlohmann::json json_data;
    try {
      file >> json_data;
    } catch (const nlohmann::json::parse_error&) {
      BAIL(err, NVCV_ERR_PARSE);
    }

    BAIL_IF_FALSE(json_data.contains("words") && json_data["words"].is_array(), err, NVCV_ERR_PARSE);

    const auto& words = json_data["words"];

    // First pass: collect valid words and find max speaker_id
    struct ParsedWord {
      double start;
      double end;
      int32_t speaker_id;
    };
    std::vector<ParsedWord> valid_words;
    int32_t max_speaker_id = -1;

    for (size_t i = 0; i < words.size(); ++i) {
      const auto& word = words[i];

      BAIL_IF_FALSE(word.contains("speaker_id"), err, NVCV_ERR_PARSE);
      BAIL_IF_FALSE(word["speaker_id"].is_number_integer(), err, NVCV_ERR_PARSE);
      int32_t speaker_id = word["speaker_id"].get<int32_t>();
      BAIL_IF_FALSE(speaker_id >= 0, err, NVCV_ERR_PARSE);
      BAIL_IF_FALSE(word.contains("start") && word.contains("end"), err, NVCV_ERR_PARSE);

      double start = word["start"].get<double>();
      double end = word["end"].get<double>();
      BAIL_IF_FALSE(start < end, err, NVCV_ERR_PARSE);

      valid_words.push_back({start, end, speaker_id});
      max_speaker_id = std::max(max_speaker_id, speaker_id);
    }

    // If no valid words, return success with empty data
    if (max_speaker_id < 0) {
      m_wordsPerSpeaker.clear();
      m_currentWordIndex.clear();
      return NVCV_SUCCESS;
    }

    // Group words by speaker_id
    m_wordsPerSpeaker.resize(static_cast<size_t>(max_speaker_id) + 1);
    for (const auto& w : valid_words) {
      m_wordsPerSpeaker[static_cast<size_t>(w.speaker_id)].push_back({w.start, w.end});
    }

    // Verify sort order per speaker
    for (size_t speaker_id = 0; speaker_id < m_wordsPerSpeaker.size(); ++speaker_id) {
      const auto& speaker_words = m_wordsPerSpeaker[speaker_id];
      for (size_t j = 1; j < speaker_words.size(); ++j) {
        if (speaker_words[j].start < speaker_words[j - 1].start) {
          m_wordsPerSpeaker.clear();
          m_currentWordIndex.clear();
          BAIL(err, NVCV_ERR_PARSE);
        }
      }
    }

    // Initialize cursors
    m_currentWordIndex.assign(m_wordsPerSpeaker.size(), 0);
    m_lastTimeStart = -1.0;
  }

bail:
  return err;
}

std::vector<uint32_t> DiarizationReader::GetActiveAudioIdsAndAdvanceCursor(double time_start, double time_end) {
  if (time_start < m_lastTimeStart) {
    return {};
  }
  m_lastTimeStart = time_start;

  std::vector<uint32_t> active_ids;

  for (size_t speaker_id = 0; speaker_id < m_wordsPerSpeaker.size(); ++speaker_id) {
    const auto& speaker_words = m_wordsPerSpeaker[speaker_id];
    if (speaker_words.empty()) continue;

    size_t& cursor = m_currentWordIndex[speaker_id];

    // Advance cursor past words that are fully before the query window
    while (cursor < speaker_words.size() && speaker_words[cursor].end <= time_start) {
      ++cursor;
    }

    // Check for overlapping words starting from cursor
    bool found = false;
    for (size_t j = cursor; j < speaker_words.size(); ++j) {
      const auto& word = speaker_words[j];
      // If word starts at or after time_end, no more overlaps possible
      if (word.start >= time_end) break;
      // Overlap: word.start < time_end && word.end > time_start
      if (word.end > time_start) {
        found = true;
        break;
      }
    }

    if (found) {
      active_ids.push_back(static_cast<uint32_t>(speaker_id));
    }
  }

  return active_ids;
}

uint32_t DiarizationReader::numSpeakers() const {
  uint32_t count = 0;
  for (const auto& words : m_wordsPerSpeaker) {
    if (!words.empty()) ++count;
  }
  return count;
}

int32_t DiarizationReader::maxSpeakerId() const {
  if (m_wordsPerSpeaker.empty()) return -1;
  return static_cast<int32_t>(m_wordsPerSpeaker.size()) - 1;
}

void DiarizationReader::ResetCursor() {
  m_currentWordIndex.assign(m_wordsPerSpeaker.size(), 0);
  m_lastTimeStart = -1.0;
}
