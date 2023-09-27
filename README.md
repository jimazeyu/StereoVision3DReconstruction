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

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20230927002709913.png" alt="image-20230927002709913" style="zoom:30%;" />

#### Results

1. Calibration

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20230927003058966.png" alt="image-20230927003058966" style="zoom:70%;" />

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20230927003044332.png" alt="image-20230927003044332" style="zoom:50%;" />

2. Harris feature detection

   <img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20230927003144957.png" alt="image-20230927003144957" style="zoom:50%;" />

3. Orb feature detection

   <img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20230927003152899.png" alt="image-20230927003152899" style="zoom:50%;" />

4. Sparse matching

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20230927003320062.png" alt="image-20230927003320062" style="zoom:70%;" />

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20230927003537459.png" alt="image-20230927003537459" style="zoom:70%;" />

5. Dense matching

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20230927003623082.png" alt="image-20230927003623082" style="zoom:70%;" />

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20230927003632941.png" alt="image-20230927003632941" style="zoom:70%;" />
