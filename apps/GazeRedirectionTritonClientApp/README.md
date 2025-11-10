GazeRedirectionTritonClientApp
==============================

The GazeRedirectionTritonClientApp is a sample app only for the Triton enabled AR SDK, which can be used to run the Eye Contact effect on the server. 

It can concurrently process multiple input video files with Eye Contact. 

Its usage is: 
```
GazeRedirectionTritonClientApp [flags ...] inFile1 [ inFileN ...]
```

The inFile1, ... , inFileN are of the input video files of the same resolution. The input files are not included with the sample app in the SDK.

Required Features
-----------------
This app requires the following features to be installed. Make sure to install them using *install_features.ps1* (Windows) or *install_features.sh* (Linux) in your AR SDK features directory before building it.
- nvARGazeRedirection

Run the Triton Client Application
---------------------------------

First make sure you have the Triton server application running. See the base README.md for information on this.

The following sets up the AR SDK library path and then runs gaze redirection to produce three output video files.

```
source setup_env.sh

./GazeRedirectionTritonClientApp video1.mp4 video2.mp4 video3.mp4
```

Command-Line Arguments for the Gaze Redirection Triton Client Application
-------------------------------------------------------------------------

| Argument                       | Description |
|--------------------------------|-------------|
| `--url=<URL>`                  | URL to the Triton server |
| `--grpc[={true\|false}]`       | Use gRPC for data transfer to the Triton server instead of CUDA shared memory. |
| `--output_name_tag=<string>`   | A string appended to each inFile to create the corresponding output file name |
| `--log=<file>`                 | Log SDK errors to a file, "stderr" or "" (default stderr) |
| `--log_level=<N>`              | The desired log level: {`0`, `1`, `2`} = {FATAL, ERROR, WARNING}, respectively (default `1`) |
| `--eyesize_sensitivity`        | Set the eye size sensitivity parameter, an integer value between `2` and `6` (default `3`) |
| `--enable_look_away`           | Enables random look away to avoid staring (default 0), non-zero value to enable |
| `--look_away_offset_max`       | Maximum integer value of gaze offset angle (degrees) when lookaway is enabled (default `5`) |
| `--look_away_interval_min`     | Minimum value in seconds (integer value) for the lookaway interval in seconds (default `3`) |
| `--look_away_interval_range`   | Range in seconds (integer value) for the lookaway interval in seconds (default `8`) |
| `--gaze_pitch_threshold_low`   | Pitch of estimated gaze in degrees (float value) at which redirection starts transitioning away from camera and towards estimated gaze (default `25.0`) |
| `--gaze_yaw_threshold_low`     | Yaw of estimated gaze in degrees (float value) at which redirection starts transitioning away from camera and towards estimated gaze (default `20.0`) |
| `--head_pitch_threshold_low`   | Pitch of estimated head pose in degrees (float value) at which redirection starts transitioning away from camera and towards estimated gaze (default `20.0`) |
| `--head_yaw_threshold_low`     | Yaw of estimated head pose in degrees (float value) at which redirection starts transitioning away from camera and towards estimated gaze (default `25.0`) |
| `--gaze_pitch_threshold_high`  | Pitch of estimated gaze in degrees (float value) at which redirection starts transitioning away from camera and towards estimated gaze (default `30.0`) |
| `--gaze_yaw_threshold_high`    | Yaw of estimated gaze in degrees (float value) at which redirection starts transitioning away from camera and towards estimated gaze (default `30.0`) |
| `--head_pitch_threshold_high`  | Pitch of estimated head pose in degrees (float value) at which redirection starts transitioning away from camera and towards estimated gaze (default `25.0`) |
| `--head_yaw_threshold_high`    | Yaw of estimated head pose in degrees (float value) at which redirection starts transitioning away from camera and towards estimated gaze (default `30.0`) |
