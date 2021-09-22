# Traffic density estimation for traffic monitoring cameras
## Usage
### Camera calibration
#### Data organization
```
Camera Calibration  
└───Camera Code
    └───images
    │   │   vehicle_01.jpg
    │   │   vehicle_02.jpg
    │   │   ...
    │ 
    └───labels
    │   │   vehicle_01.json
    │   │   vehicle_02.json
    │   │   ...
    │ 
    └───objects
        │   vehicle_3dkeypoints.json
```
#### Step 1: Pick up vehicle images
+ Collect vehicle image data from traffic surveillance cameras.
+ Examples are shown below.

![vehicle_images](assets/vehicle_images.png)

#### Step 2: Label vehicles
+ Label the key points of each vehicle through [labelme](https://github.com/wkentaro/labelme).
+ Examples are shown below.
  
![vehicle_labels](assets/vehicle_label.png)
+ Annotated vehicle images (recommended for more than 5 images).
+ Annotated vehicle key point data.
+ 3D vehicle key point data (recommended for more than 5 vehicle types)

#### Run the script
Please run the script camera_calibration/main.py.

### Vehicle detection
