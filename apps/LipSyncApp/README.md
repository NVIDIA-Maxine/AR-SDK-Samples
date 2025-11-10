LipSyncApp
==========

LipSyncApp is a sample application that demonstrates the lipsyncing ability of the NVIDIA AR SDK.

The application requires a video file and an audio file, as specified with command-line arguments, enumerated by executing: `LipSyncApp.exe --help` (on Windows) or `./LipSyncApp --help` (on Linux).

To run LipSync app with the default source video and driving audio:
On Windows:
```
  run_lipsync_offline.bat
```
On Linux:
```
  run_lipsync_offline.sh
```

* Default source video is "lipsyncSampleVideo.mp4" from the "samples/resources" directory
* Default driving audio is "lipsyncSampleAudio.wav" from the "samples/resources" directory
* Default output file is "lipsync_final.mp4" in the current working directory

You can pass in your own input and output files using the optional script arguments:
On Windows:
```
  run_lipsync_offline.bat [<INPUT_VIDEO>] [<INPUT_AUDIO>] [<OUTPUT_VIDEO>]
```
On Linux:
```
  run_lipsync_offline.sh [<INPUT_VIDEO>] [<INPUT_AUDIO>] [<OUTPUT_VIDEO>]
```

If the video and audio input files have different duration, by default the sample app will stop processing when the shortest input finishes.
To change this behavior, you can edit `run_lipsync_offline.bat` or `run_lipsync_offline.sh` and add arguments to the LipSyncApp command as follows:

| Argument                            | Description |
|-------------------------------------|-------------|
| `--extend_short_video=off`          | Do not extend the video (default) |
| `--extend_short_video=forward_loop` | Extend the video by restarting the video from the beginning |
| `--extend_short_video=reverse_loop` | Extend the video by playing frames backwards from the end until the beginning (sometimes called "bounce"). Once it gets to the beginning it will start going forwards again etc.<br><br>**Warning:** This may increase execution time compared to `forward_loop` |
| `--extend_short_audio=off`          | Do not extend the audio (default) |
| `--extend_short_audio=silence`      | Extend the audio by adding silence |

If the input video contains fast head motions, it may cause artifacts in the output. In this case you can tune the
algorithm's behavior by adding the argument `--head_movement_speed=1` to the LipSyncApp command.

For a full list of command line arguments, see *Command-Line Arguments for the Lip Sync Sample Application* below

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

README - Windows
----------------

Dependencies required for the scipt and sample app to work correctly:
1. ffmpeg.exe
   - 1.1 Download <ffmpeg-release-essentials.zip> from https://www.gyan.dev/ffmpeg/builds/
   - 1.2 Unzip the file and and locate <ffmpeg.exe> and <ffprobe.exe> in "ffmpeg-6.0-essentials_build\bin"
   - 1.3 Copy <ffmpeg.exe> and <ffprobe.exe> into the same folder as LipSyncApp.exe

2. opencv_ffmpeg346_64.dll
   - 2.1 Download <opencv-3.4.6-vc14_vc15.exe> from https://sourceforge.net/projects/opencvlibrary/files/3.4.6/opencv-3.4.6-vc14_vc15.exe/download
   - 2.2 Double-click <opencv-3.4.6-vc14_vc15.exe> to extract files to a local folder
   - 2.3 In the local folder, locate <opencv_ffmpeg346_64.dll> in "opencv\build\bin" folder
   - 2.4 Copy <opencv_ffmpeg346_64.dll> into the "external\opencv\bin" folder, which already contains <opencv_world346.dll>

README - Linux
--------------

Dependencies required for the scipt and sample app to work correctly:
1. ffmpeg
   - 1.1 The build_samples.sh should have already installed ffmpeg for you
   - 1.2 If ffmpeg is not installed, please install it using "apt install ffmpeg"

Required Features
-----------------
This app requires the following features to be installed. Make sure to install them using *install_features.ps1* (Windows) or *install_features.sh* (Linux) in your AR SDK features directory before building it.
- nvARLipSync
- nvARLandmarkDetection
- nvARFaceBoxDetection
- nvARFaceExpressions

Command-Line Arguments for the Lip Sync Sample Application
----------------------------------------------------------

| Argument                            | Description |
|-------------------------------------|-------------|
| `--in_video=<file>`                 | Specifies the input video file. |
| `--in_audio=<file>`                 | Specifies the input audio file. |
| `--out=<file>`                      | Specifies the output file. Applies only if `--capture_outputs` is true. |
| `--extend_short_video=<str>`        | Specifies the desired behavior when the input video is shorter than the input audio<br><br>- `off`: Truncate the output when the input video ends (default).<br>- `forward_loop`: Extend the video by restarting it from the beginning.<br>- `reverse_loop`: Extend the video by reversing it and playing frames backward from the end. |
| `--extend_short_audio=<str>`        | Specifies the desired behavior when the input audio is shorter than the input video<br><br>- `off`: Truncate the output when the input audio ends (default).<br>- `silence`: Extend the audio by adding silence. |
| `--head_movement_speed=<N>`         | Specifies the expected speed of head motion in the input video. The default value is 0.<br><br>- `0`: slow<br>- `1`: fast
| `--model_path=<path>`               | Specifies the directory that contains the TRT models. |
| `--capture_outputs[={true\|false}]` | Write generated video to file if set to true. |
| `--codec=<fourcc>`                  | FourCC code for the desired codec (default `H264`). |
| `--log=<file>`                      | Log SDK errors to a file, "stderr" (default), or "" (empty string). |
| `--log_level=<N>`                   | Specifies the desired log level: `0` (fatal), `1` (error; default), `2` (warning), or `3` (info). |
| `--verbose[={true\|false}]`         | Reports interesting information. |
