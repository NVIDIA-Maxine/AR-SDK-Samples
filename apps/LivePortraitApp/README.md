LivePortraitApp
===============

LivePortraitApp is a sample application that demonstrates the live portrait animation feature of the NVIDIA AR SDK. The application requires a video feed from a camera connected to the computer running the application, or from a video file, as specified with command-line arguments enumerated by executing: `LivePortraitApp.exe --help` (on Windows) or `./LivePortraitApp --help` (on Linux).

Notes:
- For best quality, webcam is recommended to be mounted at the top central position of your screen. And when starting the application, maintain neutral head pose, straight gaze and neutral expression in the beginning of the driving video.
- Good source image:
  - Neutral facial expression and head pose with mouth gently closed
  - Looking at the camera
  - Solid color or simple background if RGB input
  - 720P or 1080p is recommended
- User specific source image and driving video can be passed in by `--in_src=$PATH_TO_SRC_IMG` (both offline and webcam mode) and `--in_drv=$PATH_TO_DRIVING_VIDEO` (offline mode). In offline mode you can also specify where to save the generated video file by `--out=$PATH_TO_GENERATED`.
- The live portrait app support 3 modes. Mode 1 is the default if not specified. However user can pass in `--mode=[1|2|3]` to the batch script.
  - mode `1`: native face mode. The generated video contains whole head and part of the neck. Fixed output size as 512x512(with performance model) or 1024x1024(with quality model which is the default).
  - mode `2`: registration blending mode. The generated video is the same resolution as the source image being animated.
  - mode `3`: inset blending mode. The generated video is the same resolution as the source image being animated, but light-weighted compared to mode `2`
- Two sets of models are included. User can pass in either `--model_sel=0` for performance, or `--model_sel=1` for quality. Quality model is used by default.
- The live portrait app supports background replacement in the output video. In order to do that, you need to pass in RGBA format source image using `--in_src`, and need to pass in the background image using `--bg_img`. The alpha channel of the source image should contain valid background segmentation mask.
- frame-selection sub-module is enabled by default during startup. It helps picking up a good neutral frame from the driving video so as to get the best live portrait quality. However this module can be disable by `--frame_selection=0`.
- if you want to use external webcam, you can specific `--camera CAMERA_ID` in the webcam mode

For a full list of command line arguments, see *Command-Line Arguments for the Video Live Portrait Sample Application* below.

For a full list of keyboard controls, see *Keyboard Controls for the Video Live Portrait Sample Application* below.

Limitations:
1. abrupt head movement might not be supported in mode `2` or mode `3`.
2. occlusion in the driving video is not supported.

README - Windows
----------------

How to run Live Portrait app:

1. run offline (use default source image and driving video file)
  `run_LivePortraitAppOffline.bat`

2. run webcam (use default source image and user webcam as driving video)
  `run_LivePortraitAppWebcam.bat`

In the sample app offline mode, you can ignore the error message "Could not open codec 'libopenh264': Unspecified error" if you are trying to save the output video using openCV with h264 codec on Windows. Windows will fall back to its own h.264 codec. 

README - Linux
----------------

How to run Live Portrait app:

1. run offline (use default source image and driving video file)
  `run_liveportrait_offline.sh`

2. run webcam (use default source image and user webcam as driving video)
  `run_liveportrait_webcam.sh`

Required Features
-----------------
This app requires the following features to be installed. Make sure to install them using *install_features.ps1* (Windows) or *install_features.sh* (Linux) in your AR SDK features directory before building it.
- nvARLivePortrait
- nvARLandmarkDetection
- nvARFaceBoxDetection
- nvARFrameSelection
- nvARFaceExpressions

Command-Line Arguments for the Video Live Portrait Sample Application
---------------------------------------------------------------------

| Argument                            | Description |
|-------------------------------------|-------------|
| `--mode=<mode>`                     | Specifies the Video Live Portrait working mode.<br><br>- `1`: Native facebox mode.<br>- `2`: Registration blending mode.<br>- `3`: Inset blending mode. |
| `--model_path=<path>`               | Specifies the path to the models. |
| `--model_sel=<value>`               | Specifies the model selection.<br><br>- `0`: Use model optimized for performance<br>- `1`: Use model optimized for quality |
| `--offline_mode[={true\|false}]`    | Specifies whether to use offline video or an online camera video as the input.<br><br>- `true`: Use offline video as the input.<br>- `false`: Use an online camera as the input. |
| `--capture_outputs[={true\|false}]` | If `--offline_mode=true`, specifies whether to save the output video to the file system. |
| `--cam_res=[<width>x]<height>`      | If `--offline_mode=false`, specifies the camera resolution. The <width> parameter is optional. If omitted, <width> is computed from <height> to give an aspect ratio of 4:3. For example:<br><br>`--cam_res=640x480` or `--cam_res=480`<br><br>If `--offline_mode=true`, this argument is ignored. |
| `--camera=<camera_id>`              | If `--offline_mode=false`, specifies the camera ID. Default is `0`.<br><br>If `--offline_mode=true`, this argument is ignored. |
| `--codec=<fourcc>`                  | FourCC code for the desired codec (default `H264`). |
| `--in_src=<file>`                   | Specifies the source portrait image. |
| `--in_drv=<file>`                   | - If `--offline_mode=true`, specifies the driving video.<br>- If `--offline_mode=false`, this argument is ignored |
| `--bg_img=<file>`                   | Specifies the image to use as background in the output. Valid only if the source portrait image is a 4-channel RGBA image with valid segmentation mask. |
| `--out=<file>`                      | - If `--offline_mode=true`, specifies the output video file.<br>- If `--offline_mode=false`, this argument is ignored. |
| `--show_drive[={true\|false}]`      | Specifies whether to show the driving video left to the output video side-by-side. |
| `--show_bbox[={true\|false}]`       | If `--show_drive=true`, show the face bounding box in the driving video.<br><br>If `--show_drive=false`, this argument is ignored. |
| `--frame_selection=<policy>`        | Specifies the policy to use when runing the frame selection algorithm on the driving video to capture good neutral driving images.<br><br>- `0`: Disable frame selection.<br>- `1`: Run frame selection after a good neutral driving frame has been captured. No frame selection will run against the subsequent driving frames.<br>- `2`: Continue running frame selection against each driving frame until `FrameSelection` returns `ActiveDurationExpired` status. |
| `--ignore_alpha[={true\|false}]`    | Specifies whether to ignore the alpha channel when the input source image is RGBA format. |
| `--log=<file>`                      | Log SDK errors to a file, "stderr" (default), or "". |
| `--log_level=<n>`                   | Specify the desired log level: `0` (fatal), `1` (error; default), `2` (warning), or `3` (info). |

Keyboard Controls for the Video Live Portrait Sample Application
----------------------------------------------------------------

The Video Live Portrait sample application provides the following keyboard controls to change the runtime behavior of the application.

| Key | Function |
|-----|----------|
| `F` | Toggles the frame rate display. |
| `D` | Toggles showing driving video on and off. |
| `B` | Toggles showing bounding box on the driving video on and off.<br><br>This control is enabled only if `--offline_mode=false`. |
