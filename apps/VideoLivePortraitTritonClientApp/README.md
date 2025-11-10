VideoLivePortraitTritonClientApp
================================

The VideoLivePortraitTritonClientApp is a sample app only for the Triton enabled AR SDK, which can be used to run the Video Live Portrait effect on the server. 

It can concurrently process multiple input video files with Video Live Portrait. 

Its usage is: 

```
VideoLivePortraitTritonClientApp [flags ...] inFile1 [ inFileN ...]
```

The inFile1, ... , inFileN are of the input video files of the same resolution. The input files are not included with the sample app in the SDK.

Required Features
-----------------
This app requires the following features to be installed. Make sure to install them using *install_features.ps1* (Windows) or *install_features.sh* (Linux) in your AR SDK features directory before building it.
- nvARLivePortrait

Run the Triton Client Application
---------------------------------

First make sure you have the Triton server application running. See the base README.md for information on this.

The following sets up the AR SDK library path and then runs video live portrait to produce an output video file.

Note that the Video Live Portrait feature also needs a corresponding source
image for each input video stream.

```
source setup_env.sh

./VideoLivePortraitTritonClient --src_images=img1,img2,img3
video1.mp4 video2.mp4 video3.mp4
```

Command-Line Arguments for the Video Live Portrait Triton Client Application
------------------------------------------------------------------------------

| Argument                     | Description |
|------------------------------|-------------|
| `--url=<URL>`                | URL to the Triton server |
| `--grpc[={true\|false}]`     | Use gRPC for data transfer to the Triton server instead of CUDA shared memory. |
| `--output_name_tag=<string>` | A string appended to each inFile to create the corresponding output file name |
| `--log=<file>`               | Log SDK errors to a file, "stderr" or "" (default stderr) |
| `--log_level=<N>`            | The desired log level: {`0`, `1`, `2`} = {FATAL, ERROR, WARNING}, respectively (default `1`) |
| `--live_portrait_mode`       | Live Portrait Mode `1`: Crop (Default), `2`: Blend `3`: Inset |
| `--src_images`               | Comma separated list of identically sized source images |
| `--lp_model_sel`             | Live Portrait Model. `0`: Performance (Default), `1`: Quality |
| `--use_frame_selection`      | Use Frame Selection for Live Portrait (default `true`) |
| `--show_bounding_boxes`      | Show face bounding boxes for Live Portrait (default `false`) |
