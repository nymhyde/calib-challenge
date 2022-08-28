calibration-challenge
---------------------

### video meta data
Height : 874.0
Width  : 1164.0
FPS    : 20.0   (however OpenCV plays it back at 25.0.. weird)
Total Frame Count : 1200 frames


## Understanding the problem

### Transformation : Vehicle Frame <--> Camera Frame
- To understand position and orientation of vehicles in the environment, we are interested in the co-ordinate systems
  that are aligned with the road plane and road direction.
- However, the camera is user-installed and is suffice to believe that it is arbitrarily mounted in the ego-vehicle at a
  certain height and angle, which usually is not parallel to the road plane.

### Pinhole Camera Model
- The *pinhole camera model* projects a point in 3D in a straight line that passes through a focal point on to the focal
  plane of the camera. This projection is called **perspective or rectilinear projection**. 
- A point with 3D co-ordinated x_c = R * x_v + t is mapped to a 2D co-ordinates (u,v) via the transformation s * u = K *
  x_c
- For a fixed lens whose focal length does not change, the intrinsic matrix K only depends on the camer's model and only
  has to be calibrated once.
- In contrast, R and t are referred to as the the camera's extrinsic parameters.

### Extrinsic Calibration
- While the intrinsic calibration can be done once, the extrinsic parameters depend on the position and orientation of
  the camera with respect to the ego-vehicle, which vary between users and even for the same user on different
  installations.
- So, the calibration problem that has to performed for each video refers to only the extrinsic parameters, R and t.

#### Assumptions
1. The camera is centered in the car's horizontal axis, the translation vector *t* only depends on the camera height
   *h*.

Under this assumption, the calibration of extrinsic parameters translates to estimating the camera angles (α,β,γ) and
the camera height *h*.

## Camera Calibration
- Many datasets that are available for developing perception algorithms assume that calibration values for the camera
  suite is known. This assumption is not surprising, because cameras can be carefully placed at specific positions and
  oreintations by researchers and car manufacturers.
- However, commodity dashcams or smartphone cameras are often placed by drivers or car owners themselves to record the
  environments around them.
- The camera's orientation can change easily in this case by mere touching.
- To address this issue, camera calibration is a crucial step for giving accurate localization related to other objects
  in the scene.
- The estimation of camera's extrinsic parameters can be done in two steps : Rotation Matrix and Camera Height.

#### Assumption
- Flat Ground Assumption

