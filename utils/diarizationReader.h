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

#ifndef __DIARIZATION_READER_H__
#define __DIARIZATION_READER_H__

#include <cstdint>
#include <string>
#include <vector>

#include "nvCVStatus.h"

//! A single word interval from the diarization data.
struct DiarizationWord {
  double start;  //!< Start time in seconds
  double end;    //!< End time in seconds
};

//! Loads diarization JSON and provides per-frame active audio ID lookups.
//!
//! The speaker_id field in the diarization JSON is an integer that directly
//! indexes the audio tracks. No string-to-integer mapping is needed.
//!
//! Owns both the pre-processed diarization data and the per-speaker cursor for
//! sequential frame-by-frame lookup. Each DiarizationReader instance corresponds
//! to one diarization file (and hence one audio/video stream).
//!
//! Not thread-safe: the cursor is mutated on each GetActiveAudioIdsAndAdvanceCursor() call.
class DiarizationReader {
 public:
  DiarizationReader() = default;

  //! Load and pre-process a diarization JSON file.
  //!
  //! Parses the JSON, extracts word-level speaker timestamps, and groups words
  //! by speaker_id. The speaker_id is used directly as the audio track index.
  //!
  //! Returns NVCV_ERR_PARSE if any word has:
  //! - speaker_id missing, not an integer, or negative
  //! - start or end missing
  //! - start >= end
  //! - per-speaker words not sorted by start time
  //!
  //! \param[in] file_path  Path to the diarization JSON file.
  //! \return NVCV_SUCCESS on success.
  //!         NVCV_ERR_FILE if the file cannot be opened.
  //!         NVCV_ERR_PARSE if the JSON is malformed, missing required fields, or has invalid data.
  NvCV_Status Load(const std::string& file_path);

  //! Get active audio IDs for a given time window.
  //!
  //! Advances the internal per-speaker cursor forward to find words overlapping with
  //! the specified time range. Frames must be queried in chronological order
  //! (non-decreasing time_start).
  //!
  //! \param[in] time_start  Start of the query window in seconds.
  //! \param[in] time_end    End of the query window in seconds.
  //! \return Vector of audio_ids (= speaker_ids) that have at least one word
  //!         overlapping [time_start, time_end). Values are guaranteed non-negative.
  std::vector<uint32_t> GetActiveAudioIdsAndAdvanceCursor(double time_start, double time_end);

  //! Get the number of unique speakers found in the diarization data.
  uint32_t numSpeakers() const;

  //! Get the maximum speaker_id found in the diarization data.
  //!
  //! Returns -1 if no words were loaded. The app layer should validate that
  //! maxSpeakerId() < numAudioTracks.
  int32_t maxSpeakerId() const;

  //! Reset the per-speaker cursor to the beginning.
  //!
  //! Call this to replay from the start (e.g., if looping the video).
  void ResetCursor();

 private:
  /// Per speaker_id list of word intervals (ordered by start time).
  /// Index: speaker_id. Value: vector of word intervals.
  /// Speakers with no words have empty vectors.
  std::vector<std::vector<DiarizationWord>> m_wordsPerSpeaker;

  /// Per-speaker cursor for sequential lookup. Index: speaker_id.
  std::vector<size_t> m_currentWordIndex;

  /// Last queried time_start, used to detect backwards-in-time queries.
  double m_lastTimeStart = -1.0;
};

#endif  // __DIARIZATION_READER_H__
