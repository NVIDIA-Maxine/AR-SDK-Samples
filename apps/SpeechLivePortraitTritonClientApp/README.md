SpeechLivePortraitTritonClientApp
=================================

The SpeechLivePortraitTritonClientApp is a sample app, only for the Triton enabled AR SDK, which can be used to run the Speech Live Portrait effect on the server.

It can concurrently process multiple input files with Speech Live Portrait.

Its usage is: 

```
SpeechLivePortraitTritonClientApp [flags ...] inAudioFile1 [ inAudioFileN ...]
```

The inAudioFile1, ... , inAudioFileN are of the input audio files with format of 32 bit floating PCM, mono channel and 16K sample rate. The input files are not included with the sample app in the SDK.
SpeechLivePortrait also requires source images using `--src_images` argument. All source images should be the same resolution, and the number of source image should be equal to the number of input audio files.
Each source image will be animated by the corresponding input audio file, producing video outputs. Be noted that the input audio is not muxed into the generated video. Please refer to the ffmpeg command in the `run_speechliveportraitapp.sh` script (this script is for launching native speech live portrait feature, not over Triton server) for AV mux.

Required Features
-----------------
This app requires the following features to be installed. Make sure to install them using *install_features.ps1* (Windows) or *install_features.sh* (Linux) in your AR SDK features directory before building it.
- nvARSpeechLivePortrait

Run the Triton Client Application
---------------------------------

First make sure you have the Triton server application running. See the base README.md for information on this.

The following sets up the AR SDK library path and then runs speech live portrait to produce an output video file.

Note that the Speech Live Portrait feature also needs a corresponding source
image and input audio for each input video stream.

```
source setup_env.sh

./SpeechLivePortraitTritonClientApp --src_images=img1,img2,img3
audio1.wav audio2.wav audio3.wav
```

Command-Line Arguments for the Speech Live Portrait Triton Client Application
-------------------------------------------------------------------------------

| Argument                     | Description |
|------------------------------|-------------|
| `--verbose[={true\|false}]`  | Print interesting information (default `false`). |
| `--url=<URL>`                | URL to the Triton server |
| `--grpc[={true\|false}]`     | Use gRPC for data transfer to the Triton server instead of CUDA shared memory. |
| `--output_name_tag=<string>` | A string appended to each inFile to create the corresponding output file name |
| `--log=<file>`               | Log SDK errors to a file, "stderr" or "" (default stderr) |
| `--log_level=<N>`            | The desired log level: {`0`, `1`, `2`} = {FATAL, ERROR, WARNING}, respectively (default `1`) |
| `--mode`                     | Live Portrait Mode `1`: Crop (Default), `2`: Blend `3`: Inset |
| `--src_images=<src1[, ...]>` | Comma separated list of identically sized source images |
| `--model_sel`                | Speech Live Portrait Model. `0`: Performance (Default), `1`: Quality |
| `--show_bounding_boxes`      | Show face bounding boxes for Speech Live Portrait (default `false`) |
| `--ignore_alpha`             | Ignore the alpha channel of a RBGA input source image (default `false`) |
| `--help`                     | Print out this message |
