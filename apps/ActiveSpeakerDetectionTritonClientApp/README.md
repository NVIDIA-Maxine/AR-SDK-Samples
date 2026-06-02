ActiveSpeakerDetectionTritonClientApp
=====================================

The ActiveSpeakerDetectionTritonClientApp is a sample app, only for the Triton enabled AR SDK, which can be used to run AR SDK Active Speaker Detection feature on the server.

It processes multiple input video files with multiple audio streams per video for batched speaker detection.

Per-speaker WAVs usage:

```
ActiveSpeakerDetectionTritonClientApp [flags ...] --src_videos=video0.mp4[,video1.mp4,...] --src_audios=audio1_track1.wav+audio1_track2.wav[,audio2_track1.wav,...]
```

Diarization (one mix-down WAV per video, logical channels from diarization):

```
ActiveSpeakerDetectionTritonClientApp [flags ...] --src_videos=v0.mp4,v1.mp4 --src_audios=mix0.wav,mix1.wav --diarization=d0.json,d1.json
```

Input expectations
-----------------
The application requires at least one video file and per-video audio tracks via `--src_audios`. WAV format; all audio files must share the same sample rate.

- **Video:** Resolution between 360x360 and 4096x2160 (BGR, 8-bit).
- **Audio:** Use diarized audio, frame-accurate with the video, one track per speaker. Each track should contain only that speaker's speech (or silence when not talking); clean (no background audio or noise), isolated mono. Supported sample rates: 16 kHz, 44.1 kHz, 48 kHz.
- **Performance:** Expect a few seconds of startup latency; the feature is designed for real-time processing of a single speaker.

Speaker detection requires:
- `--src_videos`: Comma-separated list of identically sized video files
- `--src_audios`: per video, commas separate video groups and '+' separates per-speaker tracks.
  Example: `--src_audios="audio0_0.wav+audio0_1.wav,audio1_0.wav"` means video0 has 2 audio tracks, video1 has 1 audio track.
  With `--diarization`, exactly one mix-down WAV is expected per video; logical channels are derived from diarization speaker IDs.

The videos will be processed in batch for speaker detection using the provided audio streams, producing video outputs with bounding boxes drawn around detected speakers.

The output videos will show:
- **Green bounding boxes** around speakers who are currently speaking
- **Blue bounding boxes** around previously identified speakers (audio ID assigned, not currently speaking)
- **Red bounding boxes** around tracked faces with no audio assigned yet
- **Text labels**:
  - Top-left: `"Track:Y (confidence)"` showing tracking ID and confidence score
  - Bottom-right: `"Audio:X"` showing audio ID (only displayed when audio is assigned)

Required Features
-----------------
This app requires the following features to be installed. Make sure to install them using *install_feature.ps1* (Windows) or *install_feature.sh* (Linux) in your AR SDK features directory before building it.
- nvARActiveSpeakerDetection

Run the Triton Client Application
---------------------------------

First make sure you have the Triton server application running. See the base README.md for information on this.

The following sets up the AR SDK library path and then runs speaker detection to produce output video files.

**Single video with multiple audio tracks:**
```
source setup_env.sh

./ActiveSpeakerDetectionTritonClientApp --src_videos=video0.mp4 --src_audios=audio0_0.wav+audio0_1.wav
```

**Multiple videos with audio tracks:**
```
source setup_env.sh

./ActiveSpeakerDetectionTritonClientApp --src_videos=video0.mp4,video1.mp4 --src_audios=audio0_0.wav+audio0_1.wav,audio1_0.wav
```

Multi-View Mode (Cross-Stream Face Identification)
---------------------------------------------------

When running Active Speaker Detection on multiple cameras showing the same scene, the default model assigns independent face track IDs per stream. To get **consistent face IDs across streams**, use the `ActiveSpeakerDetectionMultiView` model:

```
./ActiveSpeakerDetectionTritonClientApp --multiview --src_videos=cam0.mp4,cam1.mp4 --src_audios=audio0.wav,audio1.wav
```

Command-Line Arguments for the Speaker Detection Triton Client Application
--------------------------------------------------------------------------

| Argument                               | Description |
|----------------------------------------|-------------|
| `--src_videos=<v0,v1,...>`             | Comma separated list of identically sized source video files |
| `--src_audios=<v0_a0+v0_a1,v1_a0,...>` | Audio files per video. Comma separates videos, `+` separates tracks. With `--diarization`, exactly one mix-down WAV is expected per video; logical channels are derived from diarization speaker IDs. Example: `"audio0_0.wav+audio0_1.wav,audio1_0.wav"` means video0 has 2 audio tracks, video1 has 1 audio track. |
| `--output_name_tag=<string>`           | A string appended to each input video file to create the corresponding output file name (default: `output`) |
| `--output_codec=<fourcc>`              | FourCC code for the desired codec (default: `mp4v` -- MPEG-4) |
| `--output_format=<format>`             | Format of the output video (default: `mp4`) |
| `--sync_tolerance=<f>`                 | Min sync score [0, 1] to consider speaking (`-1` = unset, use SDK default) |
| `--max_sync_faces=<N>`                 | Max faces for sync discrimination (default: `0` = no limit) |
| `--multiview[={true\|false}]`          | Enable cross-stream face identification with consistent tracking IDs across all camera streams (default: false) |
| `--diarization=<d0,d1,...>`            | Comma separated list of diarization JSON files (one per video) |
| `--verbose[={true\|false}]`            | Print verbose information (default: false) |
| `--url=<URL>`                          | URL to the Triton server |
| `--grpc[={true\|false}]`               | Use gRPC for data transfer to the Triton server instead of CUDA shared memory (default: false) |
| `--log=<file>`                         | Log SDK errors to a file, `"stderr"` or `""` (default: `stderr`) |
| `--log_level=<N>`                      | The desired log level: `{0, 1, 2, 3}` = `{FATAL, ERROR, WARNING, INFO}` (default: 1) |


Output
------

**Output file size and codec:** Output video uses **`mp4v` (MPEG-4)** by default for better compression.
Use `--output_codec=<fourcc>` (e.g. `avc1` for H.264) for quality/size tradeoffs.
See the arguments table for `--output_codec` and `--output_format`.

For each input video `input_video.mp4`, the application will generate:
- `input_video_output.mp4` - The input video with bounding boxes and labels overlaid

Example output shows:
- Green rectangles around active speakers
- Blue rectangles around previously identified speakers (not currently speaking)
- Red rectangles around tracked faces with no audio assigned yet
- Text labels:
  - Top-left: `"Track:Y (confidence)"` showing tracking ID and confidence score
  - Bottom-right: `"Audio:X"` showing audio ID (only displayed when audio is assigned)
