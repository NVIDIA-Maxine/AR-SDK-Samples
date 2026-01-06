FaceTrackApp
============

FaceTrackApp is a sample application that demonstrates the face tracking and landmark tracking of the NVIDIA AR SDK. The application requires a video feed
from a camera connected to the computer running the application, or from a video file, as specified
with command-line arguments (enumerated by executing: `./FaceTrackApp --help`). 

The sample application can run in 3 modes which can be toggled through the `1` and `2` keys on the keyboard
- `1` - Face tracking
- `2` - Facial landmark tracking

In the sample app offline mode, you can ignore the error message "Could not open codec 'libopenh264': Unspecified error" if you are trying to save the output video using openCV with h264 codec on Windows. Windows will fall back to its own h.264 codec. 

Required Features
-----------------
This app requires the following features to be installed. Make sure to install them using *install_features.ps1* (Windows) or *install_features.sh* (Linux) in your AR SDK features directory before building it.
- nvARLandmarkDetection
- nvARFaceBoxDetection

Command-Line Arguments for the FaceTrack Sample Application
-----------------------------------------------------------

| Argument                             | Description |
|--------------------------------------|-------------|
| `--app_mode = <mode>`                | Specifies the mode in which the app runs at startup:<br><br>- `0`: Face Detection and Tracking<br>- `1`: Facial Landmark Detection. |
| `--model_path=<path>`                | Specifies the path to the models. |
| `--landmarks_126[={true\|false}]`    | Specifies whether to set the number of landmark points to 126 or 68:<br><br>- `true`: Set the number of landmarks to 126.<br>- `false`: Set the number of landmarks to 68. |
| `--landmark_mode=<mode>`             | Specifies whether to set the high-quality landmark model or high-performance model:<br><br>- `0`: Use a high-performance landmark model (default).<br>- `1`: Use a high-quality landmark model. |
| `--temporal[={true\|false}]`         | Optimizes the results for temporal input frames. If the input is a video, set this value to true. |
| `--offline_mode[={true\|false}]`     | Specifies whether to use offline video or an online camera video as the input:<br><br>- `true`: Use offline video as the input.<br>- `false`: Use an online camera as the input. |
| `--capture_outputs[={true\|false}]`  | If `--offline_mode=false`, specifies whether to enable the following features:<br><br>- Toggling video capture on and off by pressing the C key.<br>- Saving an image frame by pressing the S key.<br><br>Additionally, a result file that contains the detected landmarks and face boxes is written at the time of capture.<br><br>If `--offline_mode=true`, this argument is ignored. |
| `--cam_res=[<width>x]<height>`       | If `--offline_mode=false`, specifies the camera resolution. The width is optional. If you omit a value for the width, the value is computed from the height for an aspect ratio of 4:3. For example:<br><br>`--cam_res=640x480` or `--cam_res=480`.<br><br>If `--offline_mode=true`, this argument is ignored. |
| `--in=<file>`                        | - If `--offline_mode=true`, specifies the input video file.<br>- If `--offline_mode=false`, this argument is ignored. |
| `--out=<file>`                       | - If `--offline_mode=true`, specifies the output video file.<br>- If `--offline_mode=false`, this argument is ignored. |
| `--log=<file>`                       | Log SDK errors to a file, "stderr" (default), or "". |
| `--log_level=<n>`                    | Specify the desired log level: 0 (fatal), 1 (error; default), 2 (warning), or 3 (info). |

Keyboard Controls for the FaceTrackApp Sample Application
------------------------------------------------------

The FaceTrackApp sample application provides the following keyboard
controls to change the runtime behavior of the application:

| Key | Function |
|-----|----------|
| 1   | Selects the face-tracking-only mode and shows only the bounding boxes. |
| 2   | Selects the face and landmark tracking mode and shows only landmarks. |
| W   | Toggles the selected visualization mode on and off. |
| F   | Toggles the frame rate display. |
| C   | Toggles video saving on and off.<br><br>- When video saving is toggled off, a file is saved with the captured video with a result file that contains the detected face box and landmarks.<br><br>- This control is enabled only if `--offline_mode=false` and `--capture_outputs=true`. |
| S   | Saves an image and a result file.<br><br>This control is enabled only if `--offline_mode=false` and `--capture_outputs=true`. |
