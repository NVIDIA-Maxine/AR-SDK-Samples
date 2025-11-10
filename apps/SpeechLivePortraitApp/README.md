SpeechLivePortraitApp
=====================

SpeechLivePortraitApp is a sample application that demonstrates the live portrait animation of the NVIDIA AR SDK. The application requires an audio file, as specified with command-line arguments enumerated by executing: `SpeechLivePortraitApp.exe --help` (on Windows) or `./SpeechLivePortraitApp --help` (on Linux).

Notes:
- The speech live portrait app support 3 modes. Mode 1 is the default if not specified. However user can pass in `--mode=[1|2|3]` to the run-sript.
  - mode `1`: face mode. The generated video contains whole head and part of the neck. Fixed output size as 512x512.
  - mode `2`: blending mode. The generated video is the same resolution as the source image being animated.
  - mode `3`: inset mode. The generated video is the same resolution as the source image being animated, but light-weighted compared to mode2
- Good source image:
  - 540p - 4K resolution; >720p is recommended 
  - Full face included and centered in the image
  - Neutral expression (no smiling or any expression)
  - Mouth is closed
  - Front facing pose and gaze
  - Clear face features without occlusion (e.g, scarf on the face)
  - Large accessories (e.g., headphones) are not recommended
  - Good lighting condition 
  - Solid color or simple background
  - Only RGB is supported. RGBA is not supported 
- Good driving audio:
  - 16KHz sample rate
  - Mono Channel
  - Only one speaker
  - Little-to-no background noise

For a full list of command line arguments, see *Command-Line Arguments for the Speech Live Portrait Sample Application* below.

README - Windows
----------------

- Step 1. Add required .dll files to generate audio-video sync speech live portrait output
  - 1.1 openh264-1.7.0-win64.dll
    - 1.1.1 Download **openh264-1.7.0-win64.dll.bz2** from http://ciscobinary.openh264.org/openh264-1.7.0-win64.dll.bz2
    - 1.1.2 Unzip **openh264-1.7.0-win64.dll.bz2** and locate the **openh264-1.7.0-win64.dll** file
    - 1.1.3 Copy **openh264-1.7.0-win64.dll** into the same folder as **SpeechLivePortraitApp.exe**

  - 1.2. opencv_ffmpeg346_64.dll 
    - 1.2.1 Download **opencv-3.4.6-vc14_vc15.exe** from https://sourceforge.net/projects/opencvlibrary/files/3.4.6/opencv-3.4.6-vc14_vc15.exe/download
    - 1.2.2 Double-click **opencv-3.4.6-vc14_vc15.exe** to extract files to a local folder
    - 1.2.3 In the local folder, locate **opencv_ffmpeg346_64.dll** in "opencv\build\bin" folder 
    - 1.2.4 Copy **opencv_ffmpeg346_64.dll** into the "external\opencv\bin" folder, which already contains **opencv_world346.dll**

- Step 2. Generate speech live portrait results 

By default, the SDK sample app generates an output video WITHOUT audio. If you would like to use ffmpeg to mux the audio into the generated video, please follow step 2.2 instead of 2.1

  - 2.1 Generate video-only results from **SpeechLivePortraitApp.exe**
    - 2.1.1 Make sure the source image path is `$PATH_TO_SRC_IMG`, the audio path is `$PATH_TO_DRIVING_AUDIO `
    - 2.1.2 Run speech live portrait app using command line directly 
          ```SpeechLivePortraitApp.exe --in_src=$PATH_TO_SRC_IMG --in_drv=$PATH_TO_DRIVING_AUDIO --out=$PATH_TO_GENERATED_VIDEO```
    - 2.1.3 The video-only result will be saved in `$PATH_TO_GENERATED_VIDEO`

  - 2.2 Generate speech live portrait with audio, using ffmpeg to mux audio
    - 2.2.1 Add ffmpeg.exe
      - 2.2.1.1 Download **ffmpeg-release-essentials.zip** from https://www.gyan.dev/ffmpeg/builds/ 
      - 2.2.1.2 Unzip the file and and locate **ffmpeg.exe** and **ffprobe.exe** in "ffmpeg-6.0-essentials_build\bin"
      - 2.2.1.3 Copy **ffmpeg.exe** and **ffprobe.exe** into the same folder as **SpeechLivePortraitApp.exe**
  
    - 2.2.2 Run scripts to generate the result
      - 2.2.2.1 Make sure **speech_live_portrait_sample_audio.wav** and **speech_live_portrait_sample_portrait.png** are both in the same folder as **SpeechLivePortraitApp.exe**
      - 2.2.2.2 Run **run_speechliveportrait.bat** will generate a **speech_live_portrait_final.mp4** file

    - 2.2.3 Verify the result has audio-video sync
      - 2.2.3.1 Open terminal (cmd)
      - 2.2.3.2 `ffprobe.exe -i speech_live_portrait_final.mp4 -show_entries format=duration`
      - 2.2.3.3 Check if the "Stream #0 Video" and "Stream #1 Audio" have similar duration

README - Linux
--------------

How to run Speech Live Portrait app with the default source image and driving audio:
```
run_speechliveportrait.sh
```

You can pass in your own source image and driving audio by appending `--in_src=file` and `--in_drv=file` to the launching script. You might also need to change the ffmpeg command (for audio/video sync) manually inside the script accordingly. Speech Live Portrait only supports audio format of 32 bit floating PCM, mono channel and 16K sample rate. For better animation quality, suggestions of the source image:

How to mux the output video with input audio in post process(SDK only generates output video without audio):
1. The build_samples.sh would have already installed ffmpeg for you. If not, please install it using `apt install ffmpeg`
2. use the `ffmpeg` command in `run_speechliveportrait.sh` as reference

Required Features
-----------------
This app requires the following features to be installed. Make sure to install them using *install_features.ps1* (Windows) or *install_features.sh* (Linux) in your AR SDK features directory before building it.
- nvARFaceExpressions
- nvARLandmarkDetection
- nvARFaceBoxDetection
- nvARFace3DReconstruction

Command-Line Arguments for the Speech Live Portrait Sample Application
----------------------------------------------------------------------

| Argument                                 | Description |
|------------------------------------------|-------------|
| `--mode=<mode>`                          | Specifies the Speech Live Portrait working mode.<br><br>- `1`: Native facebox mode.<br>- `2`: Registration blending mode.<br>- `3`: Inset blending mode. |
| `--model_path=<path>`                    | Specifies the path to the models. |
| `--model_sel=<value>`                    | Specifies the model selection.<br><br>- `0`: Use model optimized for performance<br>- `1`: Use model optimized for quality |
| `--capture_outputs[={true\|false}]`      | Specifies whether to save the output video to the file system. |
| `--codec=<fourcc>`                       | FourCC code for the desired codec (default `H264`). |
| `--in_src=<file>`                        | Specifies the source portrait image. |
| `--in_drv=<file>`                        | Specifies the audio input. |
| `--out=<file>`                           | Specifies the output video file. |
| `--blink_duration=<number>`              | Specifies (in frames) the duration of eye blinks. Default is `6`. |
| `--blink_frequency=<number>`             | Specifies the frequency of blinks per minute. Default is `15`. |
| `--head_pose_mode=<mode>`                | Specifies the mode for head animation.<br><br>- `1`: No head animation.<br>- `2`: Use predefined head animation (default). A predefined animation is applied internally.<br>- `3`: Use user-provided head animation. User can provide their own head pose animation. |
| `--enable_look_away=<mode>`              | Specifies whether to enable random lookaway to avoid staring. Default is `0` (turned off). |
| `--look_away_offset_max=<number>`        | Specifies the maximum value (in degrees) of gaze offset when lookaway is enabled. Default is `20`. |
| `--look_away_interval_min=<seconds>`     | Specifies the minimum interval in integer seconds for triggering the lookaway event. Default is `8`. |
| `--look_away_interval_range=<seconds>`   | Specifies the range of the interval in integer seconds for triggering the lookaway event. Default is `3`. |
| `--mouth_expression_multiplier=<number>` | Specifies the degree of exaggeration for mouth movements. Higher values result in more exaggerated mouth motions. Default: `1.4f`. Range: [`1.0f`, `1.6f`]. |
| `--mouth_expression_base=<number>`       | Defines the base openness of the mouth when idle (that is, zero audio input). Higher values lead to a more open mouth appearance during the idle state. Default: `0.3f`. Range: [`0.0f`, `1.0f`]. |
| `--head_pose_multiplier=<number>`        | Specifies a multiplier to dampen the head pose animation. This is applicable only for `HeadPoseMode=2`. Default is `1.0f`. |
| `--log=<file>`                           | Log SDK errors to a file, "stderr" (default), or "". |
| `--log_level=<n>`                        | Specify the desired log level: `0` (fatal), `1` (error; default), `2` (warning), or `3` (info). |
