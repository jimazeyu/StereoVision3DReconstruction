# StereoVision3DReconstruction
This project accomplishes the task of 3D reconstruction based on binocular stereo vision. It realizes the entire process from camera calibration, image matching, to point cloud visualization.

#### Introduction

Major assignment for machine vision, sparse mapping and dense mapping with stereo cameras.

#### Instructions

1. Use `chmod 777 ./build.sh`
2. Use `./build.sh` to compile the program. The compiled executable files are in the bin folder.
3. There are 3 executable files generated: stereoDense for dense mapping, stereoSparse for sparse mapping, and xxx.
4. Use the following commands to run the files:
   - `./bin/stereoSparse pics/img0.png pics/img1.png` for sparse mapping.
   - `./bin/stereoDense pics/img0.png pics/img1.png` for dense mapping.

#### Framework

First, camera calibration is performed, including monocular camera calibration, distortion correction, and stereo camera alignment. The Zhang's chessboard calibration method is used to obtain the parameters of the monocular camera, and then the stereo camera images are aligned and preprocessed.
Image matching has two methods: The first is sparse matching based on feature points, using Harris or Orb to detect feature points, and then using Brief to describe and match these points; the second is dense matching using the SGBM algorithm.
After matching, the camera parameters are combined to calculate disparity and depth. Finally, the three-dimensional coordinates are recorded and visualized using the pangolin library, completing the 3D reconstruction.

![image](https://github.com/jimazeyu/StereoVision3DReconstruction/assets/69748976/b5e9d5b8-916b-48af-a212-96e5fa3fd72e)


#### Results

1. Calibration

![image](https://github.com/jimazeyu/StereoVision3DReconstruction/assets/69748976/2cddafa2-000d-49c6-88c0-76d12b1540c2)


![image](https://github.com/jimazeyu/StereoVision3DReconstruction/assets/69748976/d45a2b81-4f04-42b3-9201-dd01eaaeb181)


2. Harris feature detection

![image](https://github.com/jimazeyu/StereoVision3DReconstruction/assets/69748976/fd6eb10d-cce7-47dc-b09e-fae0ce82df84)


3. Fast feature detection

![image](https://github.com/jimazeyu/StereoVision3DReconstruction/assets/69748976/72daed11-1134-4bef-9c02-0e98387cd70d)


4. Sparse matching

![image](https://github.com/jimazeyu/StereoVision3DReconstruction/assets/69748976/c26e6f52-7d4c-4040-899b-6cb30cbbac38)

![image](https://github.com/jimazeyu/StereoVision3DReconstruction/assets/69748976/890ef98b-7f22-4981-aed9-17cc16f2a21a)


5. Dense matching
   
![image](https://github.com/jimazeyu/StereoVision3DReconstruction/assets/69748976/e79f5ce6-ef0d-41f4-9a21-ab4d92f9bfd5)

![image](https://github.com/jimazeyu/StereoVision3DReconstruction/assets/69748976/9c3bf38b-7779-4144-bffe-877ffa1c8e3c)

