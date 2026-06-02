LipSyncTritonClientApp
======================

The LipSyncTritonClientApp is a sample app, only for the Triton enabled AR SDK, which can be used to run AR SDK Lip Sync feature on a Triton server.

It can concurrently process multiple input files.

Required Features
-----------------
This app requires the following features to be installed. Make sure to install them using *install_feature.ps1* (Windows) or *install_feature.sh* (Linux) in your AR SDK features directory before building it.
- nvARLipSync

Additional Dependencies
-----------------------

In order for the sample app and scripts to work correctly, FFmpeg must be installed.

```bash
apt install ffmpeg
```

Usage
-----

First make sure you have the Triton server application running. See the base README.md for information on this.

The following sets up the AR SDK library path and then runs the app to produce output video files.

```bash
# Single video and audio file
./run_lipsynctritonclientapp_offline.sh --src_videos=vid1.mp4 --src_audios=audio1.wav
# Multiple videos and audio files
./run_lipsynctritonclientapp_offline.sh --src_videos=vid1.mp4,vid2.mp4,vid3.mp4 --src_audios=audio1.wav,audio2.wav,audio3.wav
```

All source videos should be the same resolution and the same frame-rate.
The source audio files should be 32 bit floating-point PCM, mono channel with 16 kHz sample rate.
The number of source video files should be equal to the number of source audio files.
Each source video will be lip synced using the corresponding source audio file, producing video outputs. 
The script `run_lipsynctritonclientapp_offline.sh` will mux the source audio into the generated video after processing is complete.

Command-Line Arguments for the Lip Sync Triton Client Application
--------------------------------------------------------------------

| Argument                     | Description |
|------------------------------|-------------|
| `--verbose[={true\|false}]`  | Print verbose information (default `false`). |
| `--url=<URL>`                | URL to the Triton server |
| `--grpc[={true\|false}]`     | Use gRPC for data transfer to the Triton server instead of CUDA shared memory |
| `--log=<file>`               | Log SDK errors to a file, "stderr" or "" (default "stderr") |
| `--log_level=<N>`            | The desired log level: {`0`, `1`, `2`} = {FATAL, ERROR, WARNING}, respectively (default `1`) |
| `--src_videos=<src1[, ...]>` | Comma separated list of identically sized source video files |
| `--src_audios=<src1[, ...]>` | Comma separated list of source audio files |
| `--output_name_tag=<string>` | A string appended to each inFile to create the corresponding output file name (default `"output"`) |
| `--output_codec=<fourcc>`    | FourCC code for the desired codec (default `"avc1"` -- H264) |
| `--output_format=<format>`   | Format of the output video (default `"mp4"`) |
| `--head_movement_speed=<N>`  | Specifies the expected speed of head motion in the input video. The default value is 0.<br><br>- `0`: slow<br>- `1`: fast
| `--language=<N>`             | Specify whether to use a language-specific model for processing, or the generic multi-language model: 0=Generic, 1=German, 2=French, 3=Spanish (default: 0) |
