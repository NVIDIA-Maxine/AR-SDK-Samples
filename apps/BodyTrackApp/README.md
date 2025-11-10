BodyTrackApp
============

BodyTrackApp is a sample application that demonstrates the person detection and 3D body pose estimation
features of the NVIDIA AR SDK. The application requires a video feed from a camera connected to the computer
running the application, or from a video file, as specified with command-line arguments (enumerated by 
executing: `./BodyTrackApp --help`). 

The sample application can run in 2 modes which can be toggled through the `1` and `2` keys on 
the keyboard
- `1` - Person detection
- `2` - 3D Body Pose Estimation

The sample application supports both full body and upper body pose estimation which can be controlled 
by passing the following values to argument `--fullbody_pose_estimation`:

- `1` - Full body only estimation (default)
- `0` - Full and upper body estimation

It is recommended to use full body only mode when the input video contains only full body view. 
However, if the input captures upper body view or transitions between full and upper body views, using
full and upper body mode is recommended.

Postprocessing of joint angles is available in the sample application. It helps to stablize the joints that
are undetected. It can be toggled by passing the following values to argument `--postprocess_joint_angle`:

- `1` - Enable postprocess
- `0` - Disable postprocess (default)

It is recommended to enable joint angle postprocess when input captures upper body view (ie. lower body invisible)
and if no other customized postprocessing will be applied. Otherwise, it can be disabled.

The sample application also supports multi-object tracking which can be controlled by passing the following 
values to argument `--enable_people_tracking`:

- `1` - Enable multi-object tracking
- `0` - Disable multi-object tracking (default)

In the sample app offline mode, you can ignore the error message `"Could not open codec 'libopenh264': Unspecified error"` if you are trying to save the output video using openCV with h264 codec on Windows. Windows will fall back to its own h.264 codec. 

Required Features
-----------------
This app requires the following features to be installed. Make sure to install them using *install_features.ps1* (Windows) or *install_features.sh* (Linux) in your AR SDK features directory before building it.
- nvARBodyPoseEstimation
- nvARBodyDetection

Command-Line Arguments for the BodyTrack Sample Application
-----------------------------------------------------------

| Argument                                     | Description |
|----------------------------------------------|-------------|
| `--model_path=<path>`                        | Specifies the path to the models. |
| `--mode=<mode>`                              | Specifies whether to select high-performance: mode or high-quality mode.<br><br>- 0: Set mode to high quality. Supported for both `fullbody_pose_estimation=0` and `fullbody_pose_estimation=1`.<br>- 1: Set mode to high performance. Supported for `fullbody_pose_estimation=1` only. |
| `--fullbody_pose_estimation=<value>`         | Specifies whether to select full-body pose estimation or full-body and upper-body pose estimation mode.<br><br>- 0: Set to full-body and upper-body pose estimation. Supports only high-quality mode.<br>- 1: Set to full-body pose estimation. Supports both high-quality and high-performance modes. |
| `--confidence_threshold=<float>`             | Specifies the threshold on the confidence value of output keypoint locations. This is used for displaying joints that are predicted with a confidence value higher than this threshold. Set this between 0.0 and 1.0. Used only when `fullbody_pose_estimation=0`. |
| `--postprocess_joint_angle[={true\|false}]` | Specifies whether to enable or disable the postprocessing steps for joint angles corresponding to the joints predicted with low confidence. Used only when `fullbody_pose_estimation=0`. We recommend that you set this to true when the input is an upper body image or video. |
| `--app_mode=<mode>`                          | Specifies whether to select body detection or body-pose detection.<br><br>- 0: Set mode to body detection.<br>- 1: Set mode to body-pose detection. |
| `--temporal[={true\|false}]`                 | Optimizes the results for temporal input frames. If the input is a video, set this value to true. |
| `--use_cuda_graph[={true\|false}]`           | Uses CUDA Graphs to improve performance. CUDA Graph reduces the overhead of GPU operation submission of 3D body tracking. |
| `--offline_mode[={true\|false}]`             | Specifies whether to use offline video or an online camera video as the input.<br><br>- `true`: Use offline video as the input.<br>- `false`: Use an online camera as the input. |
| `--capture_outputs[={true\|false}]`          | If `--offline_mode=false`, specifies whether to enable the following features:<br><br>- Toggling video capture on and off by pressing the C key.<br>- Saving an image frame by pressing the S key.<br><br>Additionally, a result file that contains the detected landmarks and face boxes is written at the time of capture.<br><br>If `--offline_mode=true`, this argument is ignored. |
| `--cam_res=[<width>x]<height>`               | If `--offline_mode=false`, specifies the camera resolution. The <width> parameter is optional. If omitted, `<width>` is computed from `<height>` to give an aspect ratio of 4:3. For example:<br><br>`--cam_res=640x480` or `--cam_res=480`<br><br>If `--offline_mode=true`, this argument is ignored. |
| `--in_file=<file>` or `--in=<file>`          | - If `--offline_mode=true`, specifies the input video file.<br>- If `--offline_mode=false`, this argument is ignored. |
| `--out_file=<file>` or `--out=<file>`        | - If `--offline_mode=true`, specifies the output video file.<br>- If `--offline_mode=false`, this argument is ignored. |
| `--enable_people_tracking[={true\|false}]`   | Enables Multi-Person Tracking. This is supported only for `AppMode=1`. |
| `--shadow_tracking_age=<unsigned int>`       | This argument sets the Shadow Tracking Age for Multi-Person Tracking. The default value is 90. |
| `--probation_age=<unsigned int>`             | This argument sets the Probation Age for Multi-Person Tracking. The default value is 10. |
| `--max_targets_tracked=<unsigned int>`       | This argument sets the Maximum Targets Tracked. The default value is 30, and the minimum value is 1. |
| `--log=<file>`                               | Log SDK errors to a file, "stderr" (default), or "". |
| `--log_level=<n>`                            | Specify the desired log level: 0 (fatal), 1 (error; default), 2 (warning), or 3 (info). |


Keyboard Controls for the BodyTrack Sample Application
------------------------------------------------------

The BodyTrack sample application provides the following keyboard
controls to change the runtime behavior of the application:

| Key | Function |
|-----|----------|
| 1   | Selects the *body tracking only* mode and shows only the bounding boxes. |
| 2   | Selects the *body and body pose* tracking mode and shows the bounding boxes and body pose keypoints. |
| W   | Toggles the selected visualization mode on and off. |
| F   | Toggles the frame rate display. |
| C   | Toggles video saving on and off.<br><br>- When video saving is toggled off, a file is saved with the captured video with a result file that contains the detected face box and landmarks.<br>- This control is enabled only when `--offline_mode=false` and `--capture_outputs=true`. |
| S   | Saves an image and a result file.<br><br>This control is enabled only when `--offline_mode=false` and `--capture_outputs=true`. |
