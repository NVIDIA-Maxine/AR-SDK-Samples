LipSyncApp
==========

This application demonstrates the use of NVIDIA AR SDK's Lip Sync feature to perform lip syncing on a video to match an audio file.

Overview
--------

The LipSyncApp requires a video file and an audio file, as specified with command-line arguments, enumerated by executing: `LipSyncApp.exe --help` (on Windows) or `./LipSyncApp --help` (on Linux).

Required Features
-----------------

This app requires the following features to be installed. Make sure to install them using *install_feature.ps1* (Windows) or *install_feature.sh* (Linux) in your AR SDK features directory before building it.
- nvARLipSync
- nvARLandmarkDetection
- nvARFaceBoxDetection
- nvARFaceExpressions

Additional Dependencies
-----------------------

In order for the sample app and scripts to work correctly, the following dependencies must be installed for FFmpeg support.

### Windows

#### FFmpeg command-line tools

1. Download <ffmpeg-release-essentials.zip> from https://www.gyan.dev/ffmpeg/builds/
2. Unzip the file and and locate <ffmpeg.exe> and <ffprobe.exe> in "ffmpeg-6.0-essentials_build\bin"
3. Copy <ffmpeg.exe> and <ffprobe.exe> into the same folder as LipSyncApp.exe

#### OpenCV FFmpeg backend library

1. Download `opencv-3.4.6-vc14_vc15.exe` from https://sourceforge.net/projects/opencvlibrary/files/3.4.6/opencv-3.4.6-vc14_vc15.exe/download
2. Double-click `opencv-3.4.6-vc14_vc15.exe` to extract files to a local folder
3. In the local folder, locate `opencv_ffmpeg346_64.dll` in "opencv\build\bin" folder
4. Copy <opencv_ffmpeg346_64.dll> into the "external\opencv\bin" folder, which already contains <opencv_world346.dll>

### Linux

#### FFmpeg command-line tools and libraries

```bash
apt install ffmpeg
```

Usage
-----

The `run_lipsyncapp_offline.bat` (Windows) or `run_lipsyncapp_offline.sh` (Linux) script will set up the AR SDK library path and then run the app to produce an output video file. The output video file will contain the lip synced video and the original audio.

The script will print the name of the output video file to the console.

### Basic Usage

To run LipSyncApp with the default source video and driving audio:
```bash
# Windows
./run_lipsyncapp_offline.bat
# Linux
./run_lipsyncapp_offline.sh
```

* Default source video is "lipsyncSampleVideo.mp4" from the "samples/resources" directory
* Default driving audio is "lipsyncSampleAudio.wav" from the "samples/resources" directory
* Default output file is "lipsync_final.mp4" in the current working directory

You can pass in your own input and output files using the optional script arguments:
```bash
# Windows
./run_lipsyncapp_offline.bat [<INPUT_VIDEO>] [<INPUT_AUDIO>] [<OUTPUT_VIDEO>]
# Linux
./run_lipsyncapp_offline.sh [<INPUT_VIDEO>] [<INPUT_AUDIO>] [<OUTPUT_VIDEO>]
```

Notes:
- Input video:
  - 360x360 to 4096x2160
  - Frontal facing
  - Up to +/- 30 degrees head movement for Yaw and Roll, up to +/- 15 degrees for pitch
  - Single person in the video
  - The face should be present always in the input video, and not truncated or occluded
  - Moderate to good lighting condition
- Input audio:
  - 16 kHz sample rate
  - Mono Channel
  - Only one speaker
  - Little-to-no background noise


### Command Line Arguments

The LipSyncApp supports various following command line arguments. These can be passed to the app by editing
`run_lipsyncapp_offline.bat` or `run_lipsyncapp_offline.sh` and adding the arguments to the LipSyncApp command line.

#### Basic Arguments

| Argument | Description |
|----------|-------------|
| `--in_video=<path>` | Input video file path (required) |
| `--in_audio=<path>` | Input audio file path (required) |
| `--model_path=<path>` | Directory containing the TRT models (required) |
| `--capture_outputs[={true\|false}]` | Write generated video to file if set to true (default: true) |
| `--out=<path>` | Output video file path. Applies only if --capture_outputs is true. (default: `lipsync_final.mp4`) |
| `--codec=<fourcc>`                  | FourCC code for the desired codec (default `avc1`). |
| `--log=<file>` | Log SDK errors to file, `"stderr"` or `""` (default: `stderr`) |
| `--log_level=<N>` | Log level: `{0, 1, 2, 3}` = `{FATAL, ERROR, WARNING, INFO}` (default: 1) |
| `--verbose[={true\|false}]` | Enable verbose application output (default: false) |
| `--debug[={true\|false}]` | Print and annotate output video with debug information (default: false) |

#### Algorithm Tuning Arguments

| Argument | Description |
|----------|-------------|
| `--head_movement_speed=<N>` | Specifies the expected speed of head motion in the input video. The default value is 0.<br>- `0`: slow<br>- `1`: fast
| `--bypass_factor=<[0.0,..1.0]>` | Specifies the bypass factor, a value between 0.0 and 1.0 for partial bypass.<br>- `0.0`: effect fully enabled<br>- `1.0`: effect fully bypassed (default: 0.0) |
| `--roi_rect=<x,y,w,h>` | Specifies a region of interest rectangle as x,y,width,height (no space allowed after comma). If this is specified, face detection will only be applied to this region. |
| `--roi_skip_fd[={true\|false}]` | If `--roi_rect` is specified, this flag specifies whether to skip face detection and use the ROI rectangle directly as the face bounding box (default: false) |

#### Handling Video and Audio Duration Mismatches

If the video and audio input files have different duration, by default the sample app will stop processing when the shortest input finishes.
To change this behavior, you can edit `run_lipsyncapp_offline.bat` or `run_lipsyncapp_offline.sh` and add arguments to the LipSyncApp command as follows:

| Argument                            | Description |
|-------------------------------------|-------------|
| `--extend_short_video=<str>`        | Specifies the desired behavior when the input video is shorter than the input audio<br>- `off`: Truncate the output when the input video ends (default).<br>- `forward_loop`: Extend the video by restarting it from the beginning.<br>- `reverse_loop`: Extend the video by reversing it and playing frames backward from the end. |
| `--extend_short_audio=<str>`        | Specifies the desired behavior when the input audio is shorter than the input video<br>- `off`: Truncate the output when the input audio ends (default).<br>- `silence`: Extend the audio by adding silence. |
