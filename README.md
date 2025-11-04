# Foundation Pose
Use Foundation Pose for object pose estimation and YOLO for first frame mask.
## Setup
We have to use 2 seperated conda environments to avoid conflicts.
### YOLO
For YOLO training and masking environment, run
```bash
conda env create -f environment.yml
```
### Foundation Pose
Different from the official instructions, following steps are tied for CUDA 12.+ and ROS2 integration

Create conda env and install eigen
```bash
# create conda environment
conda create -n foundationpose python=3.10

# activate conda environment
conda activate foundationpose

# Install Eigen3 3.4.0 under conda environment
conda install conda-forge::eigen=3.4.0
export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:/eigen/path/under/conda"
```
Use the modified `requirements.txt` to install dependencies
```bash
pip install -r requirements.txt
```
Install NVDiffrast
```bash
python -m pip install --quiet --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git
```
Use system compiler
```bash
export CXX=/usr/bin/g++
export CC=/usr/bin/gcc
```
Use modified `build_all_conda.sh` and `bundlesdf/mycuda/setup.py`
```bash
CMAKE_PREFIX_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/pybind11/share/cmake/pybind11 bash build_all_conda.sh
```
If `eigen` errors occur, please put your `eigen` path in your conda environment into `include_dir`

Install `pytorch3d`
```bash
conda install -c conda-forge pytorch3d
```
Solve `numpy` and `scipy`
```bash
conda install numpy=1.25 scipy=1.11
```
## Usage
### YOLO Training
1. Use [Roboflow](app.roboflow.com) to annotate images and create dataset
2. Download the dataset.
3. Use `Ultralytics` to train the model.

To see settings
```bash
yolo settings
```
Set your own run directory to save yolo models
```bash
yolo settings datasets_dir="/home/cosmosmount/Desktop/FoundationPose/yolo/datasets" runs_dir="/home/cosmosmount/Desktop/FoundationPose/yolo"
```
Start training
```bash
yolo task=detect mode=train model=yolov8n.pt data="/pathto/data.yaml" epochs=50 imgsz=640
```

### Realsense Camera
We use RGB-D camera for detection.
```bash
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true
```