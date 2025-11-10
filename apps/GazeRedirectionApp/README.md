GazeRedirectionApp
==================

GazeRedirectionApp is a sample application that demonstrates the eye contact feature of the NVIDIA AR SDK. The application can be run with a live webcam feed or offline with an input video, as specified with command-line arguments (enumerated by executing: `./GazeRedirectionApp --help`).

README - Windows
----------------

How to run GazeRedirectionApp:
  - The app can be run with a live webcam feed using `run_gazeredirectionapp_webcam.bat`
  - The app can be run offline with an input video using `run_gazeredirectionapp_offline.bat <input_file_path>` 

Note: The provided sample asset does not have an audio and is not expected to have an audio in the output. 

Note: If using the openh264 decoder, the `--split_screen_view` argument is only supported for videos resulting in a resolution
lower than 4K width. The input video needs to have a width lower than 2K pixels.

In the sample app offline mode, you can ignore the error message "Could not open codec 'libopenh264': Unspecified error" if you are trying to save the output video using openCV with h264 codec on Windows. Windows will fall back to its own h.264 codec. 

README - Linux
--------------

How to run GazeRedirectionApp:
  - The app can be run with a live webcam feed using `run_gazeredirectionapp_webcam.sh`
  - The app can be run offline with an input video using `run_gazeredirectionapp_offline.sh <input_file_path>` 

How to mux the output video with input audio (SDK generates output video without audio) in case the input video contains an audio track:
1. The `build_samples.sh` would have already installed ffmpeg for you. If not, please install it using "apt install ffmpeg"
2. Run `run_gazeredirectionapp_offline.sh`.

Note: The provided sample asset does not have an audio and is not expected to have an audio in the output. 


Required Features
-----------------
This app requires the following features to be installed. Make sure to install them using *install_features.ps1* (Windows) or *install_features.sh* (Linux) in your AR SDK features directory before building it.
- nvARGazeRedirection
- nvARLandmarkDetection
- nvARFaceBoxDetection

Command-Line Arguments for the Eye Contact Sample Application
-------------------------------------------------------------

| Argument                                | Description |
|-----------------------------------------|-------------|
| `--model_path=<path>`                   | Specifies the path to the models. |
| `--landmarks_126[={true\|false}]`       | Specifies whether to set the number of landmark points to 126 or 68:<br><br>- `true`: set the number of landmarks to 126.<br>- `false`: set the number of landmarks to 68. |
| `--temporal[={true\|false}]`            | When set to `true`, the landmark computation for eye contact is temporally optimized. |
| `--offline_mode[={true\|false}]`        | Specifies whether to use offline video or an online camera video as the input.<br><br>- `true`: Use offline video as the input.<br>- `false`: Use an online camera as the input. |
| `--redirect_gaze[={true\|false}]`       | Specifies whether to redirect the gaze.<br><br>- `true`: Gaze angles are estimated and redirected to make the person look frontal within a permissible range of angles.<br>- `false`: Perform only gaze estimation; do not redirect the gaze. |
| `--split_screen_view[={true\|false}]`   | This argument is applicable when redirection is enabled. It specifies whether to show the original video in addition to the output video. If `--offline_mode=false`, split screen mode can be toggled on and off by pressing the O key.<br><br>- `true`: Show the original video and gaze redirected output videos side by side. The visualizations are displayed on the original video.<br>- `false`: Show only the gaze redirected output video. |
| `--draw_visualizations[={true\|false}]` | When set to `true`, visualizations for the head pose and gaze direction are displayed. In addition, the head translation (x, y, z) and gaze angles (pitch, yaw) are displayed on the original video. The head pose visualization follows the color coding for red, green, and blue as x, y, and z. If gaze redirection is enabled, the split screen view should be enabled to draw the visualization. If `--offline_mode=false`, visualization mode can be toggled on and off by pressing the W key. |
| `--capture_outputs[={true\|false}]`     | If `--offline_mode=false`, video capture can be toggled on and off by pressing the C key.<br><br>A result file that contains the output video is written at the time of capture.<br><br>If `--offline_mode=true`, this argument is ignored. |
| `--cam_res=[<width>x]<height>`          | If `--offline_mode=false`, specifies the camera resolution. The <width> parameter is optional. If omitted, <width> is computed from <height> to give an aspect ratio of 4:3. For example:<br><br>`--cam_res=640x480` or `--cam_res=480`<br><br>If `--offline_mode=true`, this argument is ignored. |
| `--in=<file>`                           | - If `--offline_mode=true`, specifies the input video file.<br>- If `--offline_mode=false`, this argument is ignored. |
| `--out=<file>`                          | - If `--offline_mode=true`, specifies the output video file.<br>- If `--offline_mode=false`, this argument is ignored. |
| `--eyesize_sensitivity=<integer>`       | This argument correlates with the size of the eye region that is used to redirect the eyes. Valid values are 1, 2, 3, 4, and 5. A larger value means the eye region will be large. The default value is 3. |
| `--use_cuda_graph[={true\|false}]`      | Uses [CUDA Graphs](https://developer.nvidia.com/blog/cuda-graphs/) to improve performance. CUDA Graphs reduce the overhead of the GPU operation submission of Eye Contact networks. The default is false. |
| `--enable_look_away[={true\|false}]`    | If set to true, the eyes are redirected to look away for a random period occasionally to avoid staring. The default value is false. |
| `--look_away_offset_max=<angle>`        | If `--enable_look_away=true`, specifies the maximum integer value (0–10) of the gaze offset angle (in degrees). The default is 10. |
| `--look_away_interval_min=<seconds>`    | If `--enable_look_away=true`, specifies the minimum integer value (0–10) in seconds of the lookaway interval (in seconds). The default is 3. |
| `--look_away_interval_range=<seconds>`  | If `--enable_look_away=true`, specifies the range in integer seconds (0–10) for choosing the lookaway interval. The default is 8. |
| `--gaze_pitch_threshold_low=<float>`    | Pitch of the estimated gaze (in degrees from 10.0 to 35.0) at which redirection begins to transition away from camera and toward the estimated gaze. The default is 20.0. |
| `--gaze_yaw_threshold_low=<float>`      | Yaw of the estimated gaze (in degrees from 10.0 to 35.0) at which redirection begins to transition away from camera and toward the estimated gaze. The default is 20.0. |
| `--head_pitch_threshold_low=<float>`    | Pitch of the estimated head pose (in degrees from 10.0 to 35.0) at which redirection begins to transition away from camera and toward the estimated gaze. The default is 15.0 |
| `--head_yaw_threshold_low=<float>`      | Yaw of the estimated head pose (in degrees from 10.0 to 35.0) at which redirection begins to transition away from camera and toward the estimated gaze. The default is 25.0 |
| `--gaze_pitch_threshold_high=<float>`   | Pitch of the estimated gaze (in degrees from 10.0 to 35.0) at which redirection begins to transition away from camera and toward the estimated gaze. The default is 30.0 |
| `--gaze_yaw_threshold_high=<float>`     | Yaw of the estimated gaze (in degrees from 10.0 to 35.0) at which redirection equals the estimated gaze and redirection is turned off beyond this angle. The default is 30.0. |
| `--head_pitch_threshold_high=<float>`   | Pitch of the estimated head pose (in degrees from 10.0 to 35.0) at which redirection equals the estimated gaze and redirection is turned off beyond this angle. The default is 25.0. |
| `--head_yaw_threshold_high=<float>`     | Yaw of the estimated head pose (in degrees from 10.0 to 35.0) at which redirection equals the estimated gaze and redirection is turned off beyond this angle. The default is 30.0. |
| `--log=<file>`                          | Log SDK errors to a file, "stderr" (default), or "". |
| `--log_level=<n>`                       | Specify the desired log level: 0 (fatal), 1 (error; default), 2 (warning), or 3 (info). |

Keyboard Controls for the Eye Contact Sample Application
--------------------------------------------------------

The GazeRedirectionApp sample application provides the following keyboard
controls to change the runtime behavior of the application:

| Key         | Function |
|-------------|----------|
| `F`         | Toggles the frame rate display. |
| `C`         | Toggles video saving on and off.<br><br>- When video saving is toggled off, a file is saved with the captured video with a result file that contains the detected face box and landmarks.<br><br>- This control is enabled only if `--offline_mode=false` and `--capture_outputs=true`. |
| `L`         | Toggles the display of landmarks.<br><br>When the display of landmarks is toggled on, facial landmarks are displayed in addition to head pose and gaze. This control is enabled only if `--offline_mode=false` and `--draw_visualization=true`. |
| `O`         | Toggles the split screen view.<br><br>When toggled on, both the original and gaze redirected frames are displayed side by side. If `--draw_visualization=true` or visualization is toggled on, the head pose and gaze visualizations are displayed on the original frame. Landmarks are also optionally displayed on the original frame.<br><br>This control is enabled only if `--offline_mode=false` and `--redirect_gaze=true`. |
| `W`         | Toggles the visualizations.<br><br>When the visualization is toggled on, head pose, gaze, and landmarks can be visualized on the original frame. When toggled off, the visualizations are not displayed. This control is enabled only if `--offline_mode=false`. When `--redirect_gaze=true`, visualizations are seen only when `--split_screen_mode` is also enabled or toggled on. |
| `A`         | Toggles the `enableLookAway` option.<br><br>When `enableLookAway` is toggled on, eye contact with the camera occasionally breaks away and the user's eyes move to a small degree at random time instances. The minimum time duration and range from which the lookaway is picked can be fine tuned by the user. |
| `9` and `0` | Increments and decrements, respectively, the `LookAwayOffsetMax` parameter in the range `0–10` degrees.<br><br>When `enableLookAway` is toggled on, `LookAwayOffsetMax` specifies the maximum lookaway offset value that is added to the redirected gaze to make the person lookaway. The actual value of offset is randomly chosen in the range from `0` to `LookAwayOffsetMax`. |
| `1` and `2` | Increments and decrements, respectively, the `LookAwayIntervalMin` parameter in the range `0–10` seconds.<br><br>When `enableLookAway` is toggled on, `LookAwayIntervalMin` specifies the minimum time interval in which the lookaway happens. This parameter is internally multiplied by the frame rate. |
| `3` and `4` | Increments and decrements the `LookAwayIntervalRange` parameter in the range `0–10` seconds.<br><br>When `enableLookAway` is toggled on, `LookAwayIntervalRange` specifies the range of time in which the lookaway happens. This parameter is internally multiplied by the frame rate. |
