FaceTrackTritonClientApp
========================

The FaceTrackTritonClientApp is a sample app, only for the Triton enabled AR SDK, which can be used to run Face Detection and Landmark Detection on the server. 

Its usage is: 

```
run_facetracktritonclientapp_offline.sh --effect=effect [flags ...] inFile1 [ inFileN ...]
```
 
The inFile1, ... , inFileN are of the input video files of the same resolution. The input files are not included with the sample app in the SDK.

Required Features
-----------------
This app requires the following features to be installed. Make sure to install them using *install_features.ps1* (Windows) or *install_features.sh* (Linux) in your AR SDK features directory before building it.
- nvARFaceBoxDetection
- nvARLandmarkDetection
- nvARFace3DReconstruction
- nvARFaceExpressions

Run the Triton Client Application
---------------------------------

First make sure you have the Triton server application running. See the base README.md for information on this.

The following sets up the AR SDK library path and then runs speech live portrait to produce an output video file.

```
source setup_env.sh

./FaceTrackTritonClientApp --effect=FaceBoxDetection video1.mp4 video2.mp4
video3.mp4

./FaceTrackTritonClientApp --effect=LandmarkDetection video1.mp4 video2.mp4
video3.mp4
```

Command-Line Arguments for the FaceTrackTritonClientApp Sample Application
--------------------------------------------------------------------------

| Argument                         | Description |
|----------------------------------|-------------|
| `--effect=<effect>`              | the effect to apply (supported: FaceBoxDetection, LandmarkDetection) |
| `--url=<URL>`                    | URL to the Triton server |
| `--grpc[=(true\|false)]`         | use gRPC for data transfer to the Triton server instead of CUDA shared memory |
| `--output_name_tag=<string>`     | a string appended to each inFile to create the corresponding output file name |
| `--log=<file>`                   | Log SDK errors to a file, `"stderr"` or `""` (default stderr) |
| `--log_level=<N>`                | the desired log level: {`0`, `1`, `2`} = {FATAL, ERROR, WARNING}, respectively (default `1`) |
| `--temporal`                     | temporal flag (default `0xFFFFFFFF`) |
| `--landmarks_126[=(true\|false)]`| set the number of facial landmark points to `126`, otherwise default to `68` |
| `--landmark_mode`                | select Landmark Detection Model. `0`: Performance (Default),  `1`: Quality |
