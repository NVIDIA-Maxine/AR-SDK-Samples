ActiveSpeakerDetectionApp
=========================

This application demonstrates the use of NVIDIA AR SDK's Active Speaker Detection feature to identify and track active speakers in a video with synchronized audio.

Overview
--------

The ActiveSpeakerDetectionApp processes video and multiple audio track inputs to:
- Detect speakers in each video frame
- Track speaker identities across frames
- Identify which speakers are actively speaking
- Draw color-coded bounding boxes indicating speaker state
- Display tracking IDs and confidence scores

Input expectations
-----------------
- **Video:** Resolution between 360x360 and 4096x2160 (BGR, 8-bit).
- **Audio:** Use diarized audio, frame-accurate with the video, one track per speaker. Each track should contain only that speaker's speech (or silence when not talking); clean (no background audio or noise), isolated mono. Supported sample rates: 16 kHz, 44.1 kHz, 48 kHz.
- **Performance:** Expect a few seconds of startup latency; the feature is designed for real-time processing of a single speaker.

Required Features
-----------------
This app requires the following features to be installed. Make sure to install them using *install_feature.ps1* (Windows) or *install_feature.sh* (Linux) in your AR SDK features directory before building it.
- nvARActiveSpeakerDetection

Usage
-----

### Basic Usage

```bash
./run_activespeakerdetectionapp.bat  # Windows
./run_activespeakerdetectionapp.sh   # Linux
```

### Command Line Arguments

| Argument                    | Description |
|-----------------------------|-------------|
| `--in_video=<path>`         | Input video file path (required) |
| `--in_audios=<a0,a1,...>`   | Comma-separated WAV paths (required). With `--diarization`, exactly one mix-down file is expected; logical channels are derived from diarization speaker IDs. |
| `--out_video=<path>`        | Output video file path (default: `activeSpeakerDetectionOutput.mp4`) |
| `--codec=<fourcc>`          | FOURCC for output video (default: `mp4v`; e.g. `avc1` for H.264) |
| `--model_path=<path>`       | Directory containing the TRT models (required) |
| `--show[={true\|false}]`    | Show output video window (default: false) |
| `--verbose[={true\|false}]` | Enable verbose output (default: false) |
| `--log=<file>`              | Log SDK errors to file, `"stderr"` or `""` (default: `stderr`) |
| `--log_level=<N>`           | Log level: `{0, 1, 2, 3}` = `{FATAL, ERROR, WARNING, INFO}` (default: 1) |
| `--sync_tolerance=<f>`      | Min sync score [0, 1] to consider speaking (`-1` = unset, use SDK default) |
| `--max_sync_faces=<N>`      | Max faces for sync discrimination (default: `0` = no limit) |
| `--diarization=<path>`      | Path to diarization JSON file for active audio ID determination |

### Example

```bash
# Multiple audio tracks (comma-separated)
ActiveSpeakerDetectionApp \
  --in_video=activeSpeakerDetectionSampleVideo.mp4 \
  --in_audios=activeSpeakerDetectionSampleAudio0.wav,activeSpeakerDetectionSampleAudio1.wav \
  --out_video=activeSpeakerDetectionOutput.mp4 \
  --model_path=/path/to/models \
  --show \
  --verbose
```

## Diarization Input

Use `--diarization=<path>` to provide a diarization JSON file that controls which
audio tracks are active at each point in time. The JSON must contain a `words` array
with `start`, `end`, and `speaker_id` fields per word.

When `--diarization` is used, exactly one audio file must be provided in `--in_audios`.
The app creates `max(speaker_id) + 1` logical channels that all share the same waveform;
which channels are active each frame is determined by diarization only.

Words must be ordered by start time per speaker.

**Example:**
```bash
ActiveSpeakerDetectionApp \
  --in_video=activeSpeakerDetectionSampleVideo.mp4 \
  --in_audios=activeSpeakerDetectionSampleAudioMixed.wav \
  --diarization=activeSpeakerDetectionSampleDiarization.json \
  --out_video=activeSpeakerDetectionOutput.mp4 \
  --model_path=/path/to/models \
  --show \
  --verbose
```

## Output

**Output file size and codec:** Output video uses **`mp4v` (MPEG-4)** by default for better compression.
Use `--codec=<fourcc>` (e.g. `avc1` for H.264) for quality/size tradeoffs.

The application generates an output video file with:
- **Green bounding boxes**: Active speakers (currently speaking)
- **Blue bounding boxes**: Previously identified speakers (audio ID assigned, not currently speaking)
- **Red bounding boxes**: Tracked faces with no audio assigned yet
- **Labels**: 
  - Top-left: `"Track:Y (confidence)"` showing tracking ID and confidence score
  - Bottom-right: `"Audio:X"` showing audio ID (only displayed when audio is assigned)

## FFmpeg Video Backend (Optional)

### Windows

To use the FFmpeg video backend instead of the default Microsoft Media Foundation decoder, you need to install `opencv_ffmpeg346_64.dll`:

1. Download `opencv-3.4.6-vc14_vc15.exe` from https://sourceforge.net/projects/opencvlibrary/files/3.4.6/opencv-3.4.6-vc14_vc15.exe/download
2. Double-click `opencv-3.4.6-vc14_vc15.exe` to extract files to a local folder
3. In the local folder, locate `opencv_ffmpeg346_64.dll` in "opencv\build\bin" folder
4. Copy `opencv_ffmpeg346_64.dll` into the same folder as `ActiveSpeakerDetectionApp.exe`

The app will default to the FFmpeg backend if it is available.

### Linux

The FFmpeg backend should be available if ffmpeg is installed. If not, install it using:

```bash
apt install ffmpeg
```
