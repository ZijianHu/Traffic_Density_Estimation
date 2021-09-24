# Traffic density estimation for traffic monitoring cameras
## Usage
### Camera calibration

#### Step 1: Pick up vehicle images
+ Collect vehicle image data from traffic surveillance cameras.
+ Examples are shown below.
+ Recommended for more than 5 vehicles.

![vehicle_images](assets/vehicle_images.png)

#### Step 2: Label vehicles
+ Label the key points of each vehicle through [labelme](https://github.com/wkentaro/labelme).
+ Examples are shown below.
+ Recommended for more than 6 key points
![vehicle_labels](assets/vehicle_label.png)

#### Step 3: Prepare your vehicle model data
+ You can make your own vehicle model data.
+ The key points can be the location of the *headlight, taillight, license plate, wiper, etc*.
+ An example of the data structure of the vehicle model data is listed below

```json
{
  "Toyota_Corolla": [
    [0.27, -4.28, 0.74],
    [1.51, -4.28, 0.74],
    [0.89, -4.65, 0.45],
    [0.89, -3.53, 1.02],
    [-0.05, -3.06, 1.04],
    [1.83, -3.06, 1.04],
    [1.78, -0.2, 0.87],
    [0, 0.2, 0.87]
  ]
}, 
  "Toyota_Prius": ...
```

#### Step 4: Manage your data
+ You can store your data into the below fashion

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

#### Step 5: Run the calibration
Please run the script camera_calibration/main.py. 

There are also some hyperparameters to set. You can look into the camera_calibration/main.py for detailed information.


### Vehicle detection
