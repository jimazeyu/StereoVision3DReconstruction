/*
 * @Author: Chenhaopeng
 * @Date: 2022-06-18 21:49:53
 * @LastEditors: Chenhaopeng
 * @LastEditTime: 2022-06-18 21:49:53
 * @FilePath: /Stereo_Bighomework/include/camera_calibration.hpp
 * @Description: ��Ŀ����궨��˫Ŀ����궨��ͼ�����
 * @Version: 1.0
 */
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv; 
using namespace std;

//���궨��ĺ���
void drawChessBoard(int blocks_per_row = 11, int blocks_per_col = 8, int block_size = 75);// 11  8  75  
//ͼƬ����
const int imageWidth = 640;
const int imageHeight = 480;
//����Ľǵ���Ŀ
const int points_per_row = 10;
//����Ľǵ���Ŀ
const int points_per_col = 7;
//�ܵĽǵ���Ŀ
const int boardCorner = points_per_row * points_per_col;//�궨��ÿ��ÿ�нǵ��������10*7���ǵ�
//����궨ʱ��Ҫ���õ�ͼ��֡��
const int frameNumber = 8;
//�궨��ڰ׸��ӵĴ�С ��λ��mm
const int squareSize = 75;
//�궨������ڽǵ�
const Size corner_size = Size(points_per_row, points_per_col);
Size image_size = Size(imageWidth, imageHeight);//ͼƬ�ߴ�

Mat R, T, E, F;
//R��תʸ�� Tƽ��ʸ�� E�������� F��������
vector<Mat> rvecs; //R
vector<Mat> tvecs; //T
//��������������Ƭ�ǵ�����꼯�ϣ���Ȼ������ͬһ�������
vector<vector<Point2f>> imagePointL;
//�ұ������������Ƭ�ǵ�����꼯��
vector<vector<Point2f>> imagePointR;
//��ͼ��Ľǵ��ʵ�ʵ��������꼯��
vector<vector<Point3f>> objRealPoint;
//��������ĳһ��Ƭ�ǵ����꼯��
vector<Point2f> cornerL;
//�ұ������ĳһ��Ƭ�ǵ����꼯��
vector<Point2f> cornerR;
//Ϊ����ͼƬ����RGB�ͻҶ�ͼ�ľ���
Mat rgbImageL, grayImageL;
Mat rgbImageR, grayImageR;
//У����ת����R��ͶӰ����P����ͶӰ����Q
Mat Rl, Rr, Pl, Pr, Q;
//ӳ���
Mat mapLx, mapLy, mapRx, mapRy;
Rect validROIL, validROIR;//�������λ�ͼ����

/**
 * @description: ����궨����ģ���ʵ����������
 * @param {vector<vector<Point3f>>} & obj
 * @param {int} points_per_row
 * @param {int} points_per_col
 * @param {int} imgNumber
 * @param {int} squareSize
 * @return {}
 * @author: Chenhaopeng
 */
void RealPoint(vector<vector<Point3f>>& obj, int points_per_row, int points_per_col, int imgNumber, int squareSize)
{
    vector<Point3f> imgpoint;
    for (int rowIndex = 0; rowIndex < points_per_col; rowIndex++)
    {
        for (int colIndex = 0; colIndex < points_per_row; colIndex++)
        {
            imgpoint.push_back(Point3f(rowIndex * squareSize, colIndex * squareSize, 0));
        }
    }
    for (int imgIndex = 0; imgIndex < imgNumber; imgIndex++)
    {
        obj.push_back(imgpoint);
    }
}


int main(int argc, char** argv)
{
    //drawChessBoard();//���궨��
    //�Ƚ��е�Ŀ����ı궨
    ifstream fin("file_images.txt"); //��ȡ�궨ͼƬ��·������cpp������ͬһ·����
    if (!fin)                              //����Ƿ��ȡ���ļ�
    {
        cerr << "û���ҵ��ļ�" << endl;
        return -1;
    }
    ofstream fout("calibration_result.txt"); //�����������ڴ��ı��ļ���
    //���ζ�ȡÿһ��ͼƬ��������ȡ�ǵ�
    cout << "��ʼ��ȡ�ǵ㡭��" << endl; 
    int image_nums = 0;  //ͼƬ����
    vector<Point2f> points_per_image;               //����ÿ��ͼ��⵽�Ľǵ�
    vector<vector<Point2f>> points_all_images;      //��һ����ά���鱣���⵽�����нǵ�
    string image_file_name;                         //����һ���ļ������ַ���
    while (getline(fin, image_file_name))           //���ж�ȡ�����ж����ַ���
    {
        image_nums++;
        Mat image_raw = imread(image_file_name);        //����ͼƬ
        if (image_nums == 1)//ȷ���±궨ͼƬ�ĳߴ�
        {
            image_size.width = image_raw.cols;  //ͼ��Ŀ��Ӧ������
            image_size.height = image_raw.rows; //ͼ��ĸ߶�Ӧ������
            cout << "image_size.width = " << image_size.width << endl;
            cout << "image_size.height = " << image_size.height << endl;
        }
        //�ǵ��ⲿ��
        Mat image_gray;                               //�洢�Ҷ�ͼ�ľ���
        cvtColor(image_raw, image_gray, CV_RGB2GRAY); //��RBGͼת��Ϊ�Ҷ�ͼ
        // ��ȡ�ǵ�
        bool success = findChessboardCorners(image_gray, corner_size, points_per_image);
        if (!success)
        {
            cout << "can not find the corners " << endl;
            exit(1);
        }
        else
        {
            //�����ؾ�ȷ�������ַ�����
            find4QuadCornerSubpix(image_gray, points_per_image, Size(5, 5));  //�����ؽǵ�            
            points_all_images.push_back(points_per_image); //���������ؽǵ�
            //��ͼ�л����ǵ�λ��
            //�ǵ���ӻ�
            drawChessboardCorners(image_raw, corner_size, points_per_image, success); //���ǵ�����
            //�����ò鿴�ǵ�����Ч��
            //imshow("Camera calibration", image_raw);
            //waitKey(0); //�ȴ���������
        }
    }
    destroyAllWindows();//��ս���
    //���ͼ����Ŀ
    int image_sum_nums = points_all_images.size();
    cout << "image_sum_nums = " << image_sum_nums << endl;
    //��ʼ����궨
    Size block_size(21, 21);                            //ÿ��С����ʵ�ʴ�С, ֻ��Ӱ���������ƽ������t
    Mat camera_K(3, 3, CV_32FC1,Scalar::all(0));        //�ڲξ���3*3
    Mat distCoeffs(1, 5, CV_32FC1,Scalar::all(0));      //�������1*5
    vector<Mat> rotationMat;                            //��ת����
    vector<Mat> translationMat;                         //ƽ�ƾ���
    //��ʼ���ǵ���ά����,������,���ϵ���!!!
    vector<Point3f> points3D_per_image;
    for (int i = 0; i < corner_size.height; i++)
    {
        for (int j = 0; j < corner_size.width; j++)
        {
            points3D_per_image.push_back(Point3f(block_size.width * j, block_size.height * i, 0));
        }
    }
    vector<vector<Point3f>> points3D_all_images(image_nums, points3D_per_image);        //��������ͼ��ǵ����ά����, z=0

    int point_counts = corner_size.area(); //ÿ��ͼƬ�Ͻǵ����
    /* calibrateCamera����ʹ��˵����
     * points3D_all_images: ��ʵ��ά����
     * points_all_images: ��ȡ�Ľǵ�
     * image_size: ͼ��ߴ�
     * camera_K : �ڲξ���K
     * distCoeffs: �������
     * rotationMat: ÿ��ͼƬ����ת����
     * translationMat: ÿ��ͼƬ��ƽ������
     */
     //����궨
    calibrateCamera(points3D_all_images, points_all_images, image_size, camera_K, distCoeffs, rotationMat, translationMat, 0);
    //�Ա궨�����������
    double total_err = 0.0;               //����ͼ��ƽ������ܺ�
    double err = 0.0;                     //ÿ��ͼ���ƽ�����
    vector<Point2f> points_reproject;     //��ͶӰ��
    cout << "\n\tÿ��ͼ��ı궨���:\n";
    fout << "ÿ��ͼ��ı궨��\n";
    for (int i = 0; i < image_nums; i++)
    {
        vector<Point3f> points3D_per_image = points3D_all_images[i];
        //ͨ��֮ǰ�궨�õ����������Σ�����ά�������ͶӰ
        projectPoints(points3D_per_image, rotationMat[i], translationMat[i], camera_K, distCoeffs, points_reproject);
        //��������֮������
        vector<Point2f> detect_points = points_all_images[i];  //��ȡ����ͼ��ǵ�
        Mat detect_points_Mat = Mat(1, detect_points.size(), CV_32FC2); //��Ϊ����
        Mat points_reproject_Mat = Mat(1, points_reproject.size(), CV_32FC2);  //2ͨ������ͶӰ�ǵ����������
        for (int j = 0; j < detect_points.size(); j++)//����ȡ�Ľǵ���Ϣһһ�������
        {
            detect_points_Mat.at<Vec2f>(0, j) = Vec2f(detect_points[j].x, detect_points[j].y);
            points_reproject_Mat.at<Vec2f>(0, j) = Vec2f(points_reproject[j].x, points_reproject[j].y);
        }
        err = norm(points_reproject_Mat, detect_points_Mat, NormTypes::NORM_L2);//�������
        total_err += err /= point_counts;//�����
        cout << "��" << i + 1 << "��ͼ���ƽ�����Ϊ�� " << err << "����" << endl;
        fout << "��" << i + 1 << "��ͼ���ƽ�����Ϊ�� " << err << "����" << endl;
    }
    cout << "����ƽ�����Ϊ�� " << total_err / image_nums << "����" << endl;
    fout << "����ƽ�����Ϊ�� " << total_err / image_nums << "����" << endl;

    //���궨���д��txt�ļ������ڲ鿴
    Mat rotate_Mat = Mat(3, 3, CV_32FC1, Scalar::all(0)); //������ת����
    cout << "\n����ڲ�������:" << endl;
    cout << camera_K << endl << endl;
    fout << "\n����ڲ�������:" << endl;
    fout << camera_K << endl << endl;
    cout << "����ϵ����\n";
    cout << distCoeffs << endl << endl << endl;
    fout << "����ϵ����\n";
    fout << distCoeffs << endl << endl << endl;
    for (int i = 0; i < image_nums; i++)
    {
        Rodrigues(rotationMat[i], rotate_Mat); //����ת����ͨ���޵����˹��ʽת��Ϊ��ת����
        fout << "��" << i + 1 << "��ͼ�����ת����Ϊ��" << endl;
        fout << rotate_Mat << endl;
        fout << "��" << i + 1 << "��ͼ���ƽ������Ϊ��" << endl;
        fout << translationMat[i] << endl
            << endl;
    }
    fout << endl;
    fout.close();
    //ͼ��У��֮�󣬻��ͼ����вü�
    //��������ڲξ���
    Mat cameraMatrixL = camera_K;
    //��õĻ������
    Mat distCoeffL = distCoeffs;
    //��������ڲξ���
    Mat cameraMatrixR = camera_K;//û��˫Ŀ����ͷ��ֻ����һ������ͷ����
    Mat distCoeffR = distCoeffs;
    Mat img;
    int count = 0;
    while (count < frameNumber)
    {
        char filename[100];
        /*��ȡ��ߵ�ͼ��*/
        rgbImageL = imread("photo//9.png", IMREAD_COLOR);//��������������˫Ŀ�궨��ͼƬ
        imshow("chessboardL", rgbImageL);
        cvtColor(rgbImageL, grayImageL, CV_BGR2GRAY);//ת���ɻҶ�ͼ
        /*��ȡ�ұߵ�ͼ��*/
        rgbImageR = imread("photo//8.png", IMREAD_COLOR);
        cvtColor(rgbImageR, grayImageR, CV_BGR2GRAY);
        //�ٴνǵ���
        bool FindL, FindR;
        FindL = findChessboardCorners(rgbImageL, corner_size, cornerL);
        FindR = findChessboardCorners(rgbImageR, corner_size, cornerR);
        if (FindL == true && FindR == true)
        {
            cornerSubPix(grayImageL, cornerL, Size(5, 5), Size(-1, 1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 20, 0.1));//�����ؽǵ���
            drawChessboardCorners(rgbImageL, corner_size, cornerL, FindL);
            imshow("chessboardL", rgbImageL);//��ʾ��ǵ����ߺ��ͼƬ
            imagePointL.push_back(cornerL);

            cornerSubPix(grayImageR, cornerR, Size(5, 5), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 20, 0.1));
            drawChessboardCorners(rgbImageR, corner_size, cornerR, FindR);
            imshow("chessboardR", rgbImageR);//��ʾ�ҽǵ����ߺ��ͼƬ
            imagePointR.push_back(cornerR);
            count++;
        }
        else
        {
            cout << "the image is bad please try again" << endl;
        }
        if (waitKey(1) == 'q')
        {
            break;
        }
    }

    //����ʵ�ʵ�У�������ά���꣬����ʵ�ʱ궨���ӵĴ�С������
    RealPoint(objRealPoint, points_per_row, points_per_col, frameNumber, squareSize);
    cout << "�궨�ɹ�" << endl;

    //�궨˫Ŀ����ͷ
    /*stereoCalibrate����˵����
    * objectPoints- vector<point3f> �͵����ݽṹ���洢�궨�ǵ�����������ϵ�е�λ��
    * imagePoints1- vector<vector<point2f>> �͵����ݽṹ���洢�궨�ǵ��ڵ�һ��������µ�ͶӰ�������������
    * imagePoints2- vector<vector<point2f>> �͵����ݽṹ���洢�궨�ǵ��ڵڶ���������µ�ͶӰ�������������
    * cameraMatrix1-����/����͵ĵ�һ�������������������CV_CALIB_USE_INTRINSIC_GUESS , CV_CALIB_FIX_ASPECT_RATIO ,CV_CALIB_FIX_INTRINSIC , or CV_CALIB_FIX_FOCAL_LENGTH���е�һ��������־�����ã�������������һЩ��ȫ��������Ҫ����ʼ��
    * distCoeffs1-��һ�������������/����ͻ������������ݽ���ģ�͵Ĳ�ͬ��������������ɱ�־����
    * cameraMatrix2-����/����͵ĵڶ����������������󡣲�������ͬ��һ�������������
    * distCoeffs2-��һ�������������/����ͻ������������ݽ���ģ�͵Ĳ�ͬ��������������ɱ�־����
    * imageSize-ͼ��Ĵ�С
    * R-����ͣ���һ�͵ڶ��������֮�����ת����
    * T-����ͣ���һ�͵ڶ��������֮���ƽ�ƾ���
    * E-����ͣ���������
    * F-����ͣ���������
    * term_crit-�����Ż�����ֹ����
    * flag-
    *  CV_CALIB_FIX_INTRINSIC ����ñ�־�����ã���ô�ͻ�̶������cameraMatrix��distCoeffs���䣬ֻ���R,T,E,F
    *  CV_CALIB_USE_INTRINSIC_GUESS �����û��ṩ��cameraMatrix��distCoeffsΪ��ʼֵ��ʼ����
    *  CV_CALIB_FIX_PRINCIPAL_POINT ���������в���ı������λ��
    *  CV_CALIB_FIX_FOCAL_LENGTH ���������в���ı佹��
    *  CV_CALIB_SAME_FOCAL_LENGTH ǿ�Ʊ�������������Ľ�����ͬ
    *  CV_CALIB_ZERO_TANGENT_DIST ������䱣��Ϊ��
    *  CV_CALIB_FIX_K1,...,CV_CALIB_FIX_K6 ���������в��ı���Ӧ��ֵ����������� CV_CALIB_USE_INTRINSIC_GUESS ����ʹ���û��ṩ�ĳ�ʼֵ����������Ϊ��
    *  CV_CALIB_RATIONAL_MODEL ����ģ�͵�ѡ����������˸ò���������ʹ�ø���ȷ�Ļ���ģ�ͣ�distCoeffs�ĳ��Ⱦͻ���8
    */
    double rms = stereoCalibrate(objRealPoint, imagePointL, imagePointR,
        cameraMatrixL, distCoeffL,
        cameraMatrixR, distCoeffR,
        image_size, R, T, E, F, CALIB_USE_INTRINSIC_GUESS,
        TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, 1e-5));

    cout << "����궨���RMSΪ " << rms << endl;//����궨���
    //������������������������
    /*stereoRectify����˵����
    * cameraMatrix1-��һ������������������
    * distCoeffs1-��һ��������Ļ�������
    * cameraMatrix2-�ڶ�������������������
    * distCoeffs1-�ڶ���������Ļ�������
    * imageSize-ͼ���С
    * R- stereoCalibrate() ��õ�R����
    * T- stereoCalibrate() ��õ�T����
    * R1-������󣬵�һ���������У���任������ת�任��
    * R2-������󣬵ڶ����������У���任������ת����
    * P1-������󣬵�һ���������������ϵ�µ�ͶӰ����
    * P2-������󣬵ڶ����������������ϵ�µ�ͶӰ����
    * Q-4*4����Ȳ���ӳ�����
    * flags-��ѡ�ı�־����������� CV_CALIB_ZERO_DISPARITY ,������� CV_CALIB_ZERO_DISPARITY �Ļ����ú�����������У�����ͼ�����������ͬ���������ꡣ����ú�����ˮƽ��ֱ���ƶ�ͼ����ʹ�������õķ�Χ���
    * alpha-����������������Ϊ������ԣ������������졣�������Ϊ0����ôУ����ͼ��ֻ����Ч�Ĳ��ֻᱻ��ʾ��û�к�ɫ�Ĳ��֣����������Ϊ1����ô�ͻ���ʾ����ͼ������Ϊ0~1֮���ĳ��ֵ����Ч��Ҳ��������֮�䡣
    * newImageSize-У�����ͼ��ֱ��ʣ�Ĭ��Ϊԭ�ֱ��ʴ�С��
    * validPixROI1-��ѡ�����������Rect�����ݡ����ڲ����������ض���Ч
    * validPixROI2-��ѡ�����������Rect�����ݡ����ڲ����������ض���Ч
    */
    stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, image_size, R, T, Rl,
        Rr, Pl, Pr, Q, CALIB_ZERO_DISPARITY, -1, image_size, &validROIL, &validROIR);

    /*����ȥ����
    * initUndistortRectifyMap��������˵����
    * cameraMatrix�������������ͷ�ڲ�������3X3����
    * distCoeffs�������������ͷ����ϵ������5X1����
    * newCameraMatrix���������У�����3X3���������
    * size��������ͷ�ɼ�����ʧ��ͼ��ߴ�
    * m1type����map1���������ͣ�������CV_32FC1��CV_16SC2
    * map1���������X������ӳ�����
    * map2���������Y������ӳ�����
    */
    initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pl, image_size, CV_32FC1, mapLx, mapLy);
    initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, image_size, CV_32FC1, mapRx, mapRy);
    //������Ҫ������ͼƬ
    Mat photoleft = imread("photo//4.png");
    Mat photoright = imread("photo//3.png");
    imshow("Recitify Before", photoleft);
    Mat rectifyImageL, rectifyImageR;
    /* remap����˵����һ��ͼ����ĳλ�õ����ط��õ���һ��ͼƬָ��λ�á�
    * ����˵����
    * src��������ͼ�񣬼�ԭͼ����Ҫ��ͨ��8λ���߸������͵�ͼ��
    * dst(c++)�������ͼ�񣬼�Ŀ��ͼ�����ԭͼ��һ���ĳߴ������
    * map1�����������ֿ��ܱ�ʾ�Ķ��󣺣�1����ʾ�㣨x,y���ĵ�һ��ӳ�䣻��2����ʾCV_16SC2��CV_32FC1��
    * map2���������ֿ��ܱ�ʾ�Ķ��󣺣�1����map1��ʾ�㣨x,y��ʱ����������������κ�ֵ����2����ʾ CV_16UC1��CV_32FC1���͵�Yֵ
    * intermap2polation������ֵ��ʽ�������в�ֵ��ʽ��
    * ��1��INTER_NEAREST��������ڲ�ֵ
    * ��2��INTER_LINEAR����˫���Բ�ֵ��Ĭ�ϣ�
    * ��3��INTER_CUBIC����˫��������ֵ��Ĭ�ϣ�
    * ��4��INTER_LANCZOS4����lanczos��ֵ��Ĭ�ϣ�
    * intborderMode�����߽�ģʽ��Ĭ��BORDER_CONSTANT
    * borderValue�����߽���ɫ��Ĭ��Scalar()��ɫ
    */
    remap(photoleft, rectifyImageL, mapLx, mapLy, INTER_LINEAR);//��ӳ��
    remap(photoright, rectifyImageR, mapRx, mapRy, INTER_LINEAR);
    //����remap֮�����������ͼ���Ѿ����沢���ж�׼��
    imshow("rectifyImageL", rectifyImageL);//��ʾ������ͼ��
    imshow("rectifyImageR", rectifyImageR);
    imwrite("rectifyImageL.png", rectifyImageL);//���������ͼ��
    imwrite("rectifyImageR.png", rectifyImageR);
    //���漰�������
    FileStorage fs("intrisics.txt", FileStorage::WRITE);//�ڲ�����
    if (fs.isOpened())
    {
        fs << "cameraMatrixL" << cameraMatrixL << "cameraDistcoeffL" << distCoeffL << "cameraMatrixR" << cameraMatrixR << "cameraDistcoeffR" << distCoeffR;
        fs.release();
        cout << "������ڲξ���" << cameraMatrixL << endl << "���������ϵ����" << distCoeffL << endl << "������ڲξ���" << cameraMatrixR << endl << "���������ϵ����" << distCoeffR << endl;
    }
    else
    {
        cout << "Error: can not save the intrinsics!!!!" << endl;
    }
    fs.open("extrinsics.txt", FileStorage::WRITE);//�������
    if (fs.isOpened())
    {
        fs << "R" << R << "T" << T << "Rl" << Rl << "Rr" << Rr << "Pl" << Pl << "Pr" << Pr << "Q" << Q;
        cout << "R=" << R << endl << "T=" << T << endl << "Rl=" << Rl << endl << "Rr" << Rr << endl << "Pl" << Pl << endl << "Pr" << Pr << endl << "Q" << Q << endl;
        fs.release();
    }
    else
    {
        cout << "Error: can not save the extrinsic parameters\n";
    }
    //��ʾУ�����
    Mat canvas;
    double sf;
    int w, h;
    sf = 600. / MAX(image_size.width, image_size.height);
    w = cvRound(image_size.width * sf);
    h = cvRound(image_size.height * sf);
    canvas.create(h, w * 2, CV_8UC3);

    //��ͼ�񻭵�������
    Mat canvasPart = canvas(Rect(0, 0, w, h));
    resize(rectifyImageL, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);
    Rect vroiL(cvRound(validROIL.x * sf), cvRound(validROIL.y * sf),
        cvRound(validROIL.width * sf), cvRound(validROIL.height * sf));
    rectangle(canvasPart, vroiL, Scalar(0, 0, 255), 3, 8);

    cout << "Painted ImageL" << endl;

    //��ͼ�񻭵�������
    canvasPart = canvas(Rect(w, 0, w, h));
    resize(rectifyImageR, canvasPart, canvasPart.size(), 0, 0, INTER_LINEAR);
    Rect vroiR(cvRound(validROIR.x * sf), cvRound(validROIR.y * sf),
        cvRound(validROIR.width * sf), cvRound(validROIR.height * sf));
    rectangle(canvasPart, vroiR, Scalar(0, 255, 0), 3, 8);

    cout << "Painted ImageR" << endl;

    //���϶�Ӧ������
    for (int i = 0; i < canvas.rows; i += 16)
        line(canvas, Point(0, i), Point(canvas.cols, i), Scalar(0, 255, 0), 1, 8);

    imshow("rectified", canvas);//��ʾ����ͼ��
    imwrite("rectified.png", canvas);//�������ͼ��
    cout << "wait key" << endl;
    waitKey(0);
    return 0;
}



////�ڵ����ϻ��궨��
////��������,Ĭ��ÿ��11��block, ÿ��8��block, block��СΪ75������. Ҳ����10*7���ڵ�
//void drawChessBoard(int blocks_per_row, int blocks_per_col, int block_size)
//{
//    //blocks_per_row=11 //ÿ��11������,Ҳ����10����
//    //blocks_per_col=8  //ÿ��8������,Ҳ����7����
//    //block_size=75     //ÿ�����ӵ����ش�С
//    Size board_size = Size(block_size * blocks_per_row, block_size * blocks_per_col);
//    Mat chessboard = Mat(board_size, CV_8UC1);
//    unsigned char color = 0;
//    for (int i = 0; i < blocks_per_row; i++)
//    {
//        color = ~color;
//        for (int j = 0; j < blocks_per_col; j++)
//        {
//            chessboard(Rect(i * block_size, j * block_size, block_size, block_size)).setTo(color);
//            color = ~color;
//        }
//    }
//    Mat chess_board = Mat(board_size.height + 100, board_size.width + 100, CV_8UC1, Scalar::all(256)); //������������50�����ؿհ�
//    chessboard.copyTo(chess_board.rowRange(50, 50 + board_size.height).colRange(50, 50 + board_size.width));
//    imshow("chess_board", chess_board);
//    imwrite("chess_board.png", chess_board);
//    waitKey(-1);
//    destroyAllWindows();
//}


