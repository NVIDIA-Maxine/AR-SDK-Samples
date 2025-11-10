LipSyncTritonClientApp
======================

The LipSyncTritonClientApp is a sample app, only for the Triton enabled AR SDK, which can be used to run AR SDK features on the server. 

It can concurrently process multiple input files.

Its usage is: 
```
LipSyncTritonClientApp [flags ...] --src_videos=inVideoFile1[, ...] --src_audios=inAudioFile1[, ...]
 ```

The inAudioFile1, ... , inAudioFileN are of the input audio files with format of 32 bit floating PCM, mono channel and 16K sample rate. The input files are not included with the sample app in the SDK.
Lip sync also requires source videos using `--src_videos` argument. All source videos should be the same resolution and should have 30fps, and the number of source video files should be equal to the number of input audio files.
Each source video will be lip synced by the corresponding input audio file, producing video outputs. Be noted that the input audio is not muxed into the generated video. Please refer to the ffmpeg command in the run_lipsynctritonclientapp_offline.sh script for AV mux.

Required Features
-----------------
This app requires the following features to be installed. Make sure to install them using *install_features.ps1* (Windows) or *install_features.sh* (Linux) in your AR SDK features directory before building it.
- nvARLipSync

Run the Triton Client Application
---------------------------------

First make sure you have the Triton server application running. See the base README.md for information on this.

The following sets up the AR SDK library path and then runs speech live portrait to produce an output video file.

```
source setup_env.sh

./LipsyncTritonClientApp --src_videos=vid1.mp4,vid2.mp4,vid3.mp4 --src_audios=audio1.wav,audio2.wav,audio3.wav
```

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
