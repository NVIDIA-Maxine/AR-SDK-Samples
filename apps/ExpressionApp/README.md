ExpressionApp
=============

ExpressionApp is a sample application using the AR SDK to extract face expression signals from video. These signals are
used to control the expressions, pose and gaze of a 3D morphable face model. The application can either process
real-time video from a webcam or offline videos from files. It illustrates the facial keypoints that are tracked, plots
the expression signals that are derived, and renders an animated 3D avatar mesh.

The application runs either the Face3DReconstruction, or FaceExpression feature, depending on which expression mode is
used. The expression mode is toggled using the `1` and `2` keys on the keyboard

- `1` - Face3DReconstruction expression estimation
- `2` - FaceExpression expression estimation (default, and recommended for avatar animation)

The FaceExpression mode is preferred for avatar animation. Note that Face3DReconstruction is demonstrated for its
ability to track the face over time for AR effects. This feature enables identity face shape estimation on top of
expression estimation and is better demonstrated in the FaceTrackApp sample application. The resulting expression weights
from FaceExpression is more accurate than from Face3DReconstruction.

For details on command line arguments, execute `ExpressionApp --help`.
For more controls and configurations of the sample app, including expression definition and conversion to ARKit
blendshapes, please read the SDK programming guide. It also contains information about how to control the GUI which can
be enabled by running the application with the `--show_ui` argument (this requires the samples to be built with the `-DENABLE_UI=ON` CMake option).

In the sample app offline mode, you can ignore the error message "Could not open codec 'libopenh264': Unspecified error" if you are trying to save the output video using openCV with h264 codec on Windows. Windows will fall back to its own h.264 codec.


Required Features
-----------------
This app requires the following features to be installed. Make sure to install them using *install_features.ps1* (Windows) or *install_features.sh* (Linux) in your AR SDK features directory before building it.
- nvARFaceExpressions
- nvARLandmarkDetection
- nvARFaceBoxDetection
- nvARFace3DReconstruction

Command-Line Arguments for the ExpressionApp Sample Application
---------------------------------------------------------------

| Argument                           | Description |
|------------------------------------|-------------|
| `--cam_res=[<width>x]<height>`     | Specifies the resolution as the height or the width and height. |
| `--codec=<fourcc>`                 | FourCC code for the desired codec (default `avc1`). |
| `--debug[={true\|false}]`          | Reports debugging information (default false). |
| `--expr_mode=<number>`             | The SDK feature used for generation expressions.<br><br>- `1`: `Face3DReconstruction`<br><br>- `2`: Facial Expression Estimation (default) |
| `--pose_mode=<number>`             | Pose mode used for the FaceExpressions feature only. The default value is 0.<br><br>- `0`: `3DOF`<br><br>- `1`: `6DOF` |
| `--face_model=<file>`              | Specifies the face model to be used for fitting (default `face_model2.nvf`). |
| `--filter=<bitfield>`             | Here are the values:<br><br>- `1`: face box<br>- `2`: landmarks<br>- `4`: pose<br>- `16`: expressions<br>- `32`: gaze<br>- `256`: eye and mouth closure<br><br>The default value is 55, which means face box, landmarks, pose, expressions, gaze, and no closure. |
| `--gaze=<number>`                  | Specifies the gaze estimation mode:<br><br>- `0`: Implicit (default)<br><br>- `1`: Explicit |
| `--cheekpuff[={1\|0}]`             | (Experimental) Enable cheek puff blendshapes. The default value is 0. |
| `--fov=<degrees>`                  | The field of view in degrees. The default value is 0, which implies orthography. |
| `--help`                           | Prints help information. |
| `--in=<file>`                      | Specifies the input file. The default value is 0 (webcam). |
| `--loop[={true\|false}]`           | Plays the same video repeatedly. |
| `--model_dir=<path>`               | Specifies the directory that contains the TRT models. |
| `--model_path=<path>`              | Specifies the directory that contains the TRT models. |
| `--out=<file>`                     | Specifies the output file. |
| `--render_model=<file>`            | Specifies the face model that will be used for rendering. The default is `face_model2.nvf`. For a more comprehensive visualization model using partitions, use `face_model3.nvf`. |
| `--show[={true\|false}]`           | Shows the results. The default value is false, unless `--out` is empty. |
| `--show_ui[={true\|false}]`        | Shows the expression calibration UI. The default value is false. |
| `--temporal=<bitfield>`           | Applies the temporal filter. For more information, refer to `--filter`. |
| `--view_mode=<bitfield>`          | Here are the values:<br><br>- `1`: mesh<br>- `2`: image<br>- `4`: plot<br>- `8`: landmarks<br><br>The default value is 15, which means that all filters will be applied. |
| `--verbose[={true\|false}]`        | Reports additional information. The default value is false (disabled). |
| `--log=<file>`                     | Log SDK errors to a file, "stderr" (default), or "". |
| `--log_level=<n>`                  | Specify the desired log level: 0 (fatal), 1 (error; default), 2 (warning), or 3 (info). |

Keyboard Controls for the ExpressionApp Sample Application
----------------------------------------------------------

The ExpressionApp sample application provides the following keyboard
controls to change the runtime behavior of the application:

| Key           | Function |
|---------------|----------|
| q             | Escapes. |
| Q             | Exits the app. |
| m             | Toggles the mesh display. |
| n             | Calibrates the expression weights. |
| i             | Toggles the image display. |
| p             | Toggles the plot display. |
| l             | Toggles the landmark display (lowercase L). |
| f             | Toggles the frame rate display. |
| L or Ctrl + L | Toggles landmark filtering. |
| N or Ctrl + N | Un-calibrates expression weights. |
| P or Ctrl + P | Toggles pose filtering. |
| E or Ctrl + E | Toggles expression filtering. |
| G or Ctrl + G | Toggles gaze filtering. |
| C or Ctrl + C | Toggles closure enhancement. |
| M or Ctrl + M | Toggles pose mode. |
| 1             | Uses expressions from mesh fitting (`Face3DReconstruction`). |
| 2             | Uses expressions from DNN (Facial Expression Estimation). |

> [!NOTE]
> On Linux, the Shift keystrokes are necessary because the control
> keystrokes do not behave as expected.

Application GUI
---------------

Expression coefficients and their
fine-tuning controls can be displayed by selecting the various
**Expression Graph Options** checkboxes.
To set other display options, in **Expression Mode**, enter 1 or 2.
You can save or load settings by clicking **SaveSettings** or
**LoadSettings**.

> [!NOTE]
> The calibration process might require some fine tuning, which can
> take time.

Expression Coefficient Transfer Function
----------------------------------------

The ExpressionApp estimates the expression coefficients for each frame
that serves as the input. To extract the maximum expressivity in the
final model, use each expression coefficient. To achieve additional
expressivity and responsiveness, you can optionally tune additional
parameters through calibration. For more information, refer to the section on
Calibration.

The parameters that are tuned during calibration are applied to a
transfer function, where the coefficients range from 0 through 1:

-  0: This expression blendshape *is not* activated.
-  1: This expression blendshape *is* fully activated.

To fine-tune the expressivity, a transfer function is applied to the
expression coefficients before the coefficients are sent to render the
model, which introduces scaling, offsets, and non-linearities. The
transfer function looks like the following:

``y = 1 - (pow(1 - (max(x + a, 0) * b), c))``

In this function, *x* is the input coefficient (0 ≤ *x* ≤ 1), and *y*
 is the output coefficient. The parameters are as follows:

-  *a* = offset
-  *b* = scale
-  *c* = exponent

The parameters are tunable from the application's GUI for each
expression mode.

By playing around with the parameters, you can see the changes in
the effects, specifically for larger expression shapes such as ``jawOpen``,
or brow expressions. Instead of randomly playing with parameters,
another option is semi-automatic calibration.

Calibration
-----------

The calibration process comprises automatic and manual portions. For
each step, we will determine the *a, b,* and *c* parameters.

> [!NOTE]
> Calibration is not required, but it aids in the responsiveness and
> the accuracy of shape coefficients, especially to properly close the
> lips and add responsiveness to speech.

Here is the calibration process:

1. Calibrate the neutral expression.

   This is the automatic part of the calibration process where you sit
   straight in front of the web camera, keep a neutral face while
   looking straight into the camera, and click **Calibrate**. This
   function computes the offset parameters for each expression based on
   your neutral face and estimates the scaling parameters based on the
   offsets. The bar graph now keeps most expression coefficients close
   to zero when you maintain a neutral face.

2. Scale the expressions.

   Although you might want to isolate individual expressions, it is
   rarely possible, especially when the expression basis consists of 53
   individual expression blendshapes. Some expressions can be isolated
   better than others, and in the process of tuning the scaling, you can
   focus on expressions that can be more or less isolated from others.

   For example, focus on the on the following expressions:

   -  ``browInnerUp_R``

   -  ``browInnerUp_L``

   -  ``browOuterUp_R``

   -  ``browOuterUp_L``

   Most people can fully raise their eyebrows, which means that all four
   of these expression coefficients are set to 1.

   a. In the Expression Graph Options section in the GUI, select the
      **Brow** checkbox.

   b. A new window displays the brow-related expressions.

   c. To tune the scaling, perform the full eyebrow raise expression, and
      move the sliders to correspond to the scale parameter until all four
      expression coefficients are close to 1. If you scale the expressions
      too much, it will lead to over exposure.

   d. Repeat step c for ``browDown_R`` and ``browDown_L`` while lowering
      the eyebrows.

      Not all expressions can be isolated like brow shapes, but some
      additional expression scalings can be applied.

      The following subset of expressions can be scaled relatively
      isolated:

      - ``browDown_L``
      - ``browDown_R``
      - ``browInnerUp_L``
      - ``browInnerUp_R``
      - ``browOuterUp_L``
      - ``browOuterUp_R``
      - ``cheekPuff_L``
      - ``cheekPuff_R``
      - ``eyeBlink_L``
      - ``eyeBlink_R``
      - ``jawForward``
      - ``jawLeft``
      - ``jawOpen``
      - ``jawRight``
      - ``mouthClose``
      - ``mouthLeft``
      - ``mouthRight``
      - ``mouthSmile_L``
      - ``mouthSmile_R``

      You can isolate individual shapes and apply scaling to max out these
      expressions.

> [!NOTE]
> `mouthClose` is a special shape and should not be individually tuned.
> It works as a corrective shape for the `jawOpen` shape. This means
> that `mouthClose` should have the same scaling and the same exponent
> as the `jawOpen` shape. If the values are not the same, there might
> be undesired intersections, such as the upper and lower lips intersecting
> past each other.

3. Determine the expression exponents.

   This step is the least scientific one, but it provides additional
   expressivity and responsiveness, especially for mouth shapes that can
   be slightly muted during speech. A high exponent value makes the
   expression more responsive in the low range of values, and a low
   exponent mutes the expression. Another approach to increase the
   responsiveness while talking is to increase the exponents of all
   mouth-related blendshapes to a specific value, for example between
   1.5 and 2. The values depend on your needs which is why each
   calibration is an individual process.

   If the exponent of the `jawOpen` or `mouthClose` shape is
   changed, change the value of the other shape to the same value
   because the shapes work with each other.

   To reset the calibration settings, click `UnCalibrate`.

> [!NOTE]
> Complete calibration, including neutral expression and offset
> calibration might not always be possible. In some cases, just setting
> parameters, such as the exponents for mouth shapes helps enhance
> expressiveness.

Saving and Loading Calibration Files
------------------------------------

After you complete the calibration, to save it, click **SaveSettings**.
To load the previously saved settings, which is in the application
folder and named `ExpressionAppSettings.json`, click **LoadSettings**. To
load a specific file from a previous calibration setting, click
**LoadSettingsFromFile**.

> [!NOTE]
> A calibration file typically corresponds to a specific person in a
> specific setting, where the lighting conditions and the camera are
> mostly the same. If the capture setting is different, start a new
> calibration session.

GUI Window
----------

Windows are resizable, but if all expressions do not appear in the
calibration window, you can group them by using check boxes. Keyboard
shortcuts only work when the ExpressionApp main window is in focus.

Calibration Between Modes
-------------------------

When switching between expression mode 1 and expression mode 2, the
calibration settings should not be the same. The two modes estimate
expression shapes differently, so the set of coefficients will vary
significantly, and you need to start a new calibration session.
