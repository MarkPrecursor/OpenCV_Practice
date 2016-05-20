/*OpenCV基本矩阵操作及图像访问练习*/
# include <iostream>
# include <opencv2/opencv.hpp>
# include <opencv2/core.hpp>

using namespace std;
using namespace cv;

void Img_ReadandShow();
void Basic_Calculation();
void Fast_pixy_Access_Color();
void Fast_pixy_Access_Gray();


int main()
{
    // Fast_pixy_Access_Color();
    Fast_pixy_Access_Gray();
    return 0;
}


void Basic_Calculation()
{
    Mat E = Mat::eye(10, 10, CV_64F);
    Mat O = Mat::ones(10, 10, CV_64F) * 2;
    add(E, O, O);               //矩阵加法
    scaleAdd(E, -3, O, O);      //加权相加，可以用作相减(参数为-1时)
    absdiff(E, O, O);           //I=|I1-I2|
    subtract(E, O, O);
    cout << O << endl;
    E = E * O;                  // 乘法
    E += O;                     //加法
    E = E.t();                  //转置
    cout << E << endl;
    double e = determinant(E);  //行列式
	cout << e <<endl;
    Mat I = E.inv();
    cout << I * E << endl;
}


void Img_ReadandShow()
{
    Mat img;
    img = imread("2.JPG", IMREAD_COLOR);
    namedWindow("Result");
    imshow("Result", img);
    waitKey(0); 
}


void Fast_pixy_Access_Color()
/*彩色图像的像素快速访问*/
{
    Mat img = imread("2.JPG", IMREAD_COLOR);
    namedWindow("Test");
    imshow("Test", img);
    waitKey(200);
    double t = (double)getTickCount();
    // do something ...
    cout << img.rows * img.cols << endl;
    // cout << img.cols << endl;
    cout << "Start" << endl;
    for (int i=0; i<img.rows; i++)  
    {  
        const Vec3b* Mpoint=img.ptr <Vec3b>(i);//读取三个通道
        // cout << Mpoint << endl;
        for (int j=0;j<img.cols;j++)  
        {   
            // cout << &(*(Mpoint+j)) << endl;//这个是地址
            Vec3b intensity= *(Mpoint + j); //这个是获得的三元组，这里只能作为读取，而不能作为修改
            // cout << intensity << endl; 
            if(intensity[2] < 128) img.at<Vec3b>(i, j)[2] = 0;//修改要用这种方式
            if(intensity[1] < 128) img.at<Vec3b>(i, j)[1] = 0;
            if(intensity[0] < 128) img.at<Vec3b>(i, j)[0] = 0;
        }  
    }

    t = ((double)getTickCount() - t)/getTickFrequency();
    cout << "Times passed in seconds: " << t << endl;   
    imshow("Test", img);
    waitKey(0);
}



void Fast_pixy_Access_Gray()
{
    Mat img = imread("2.JPG", IMREAD_GRAYSCALE);
    namedWindow("Test");
    imshow("Test", img);
    waitKey(200);

    int col = img.cols;
    int row = img.rows;
    double t = (double)getTickCount();
    for(int i = 0; i < row; i++)
    {
        for(int j = 0; j < col; j++)
        {
            if(img.at<uchar>(i, j) < 128) img.at<uchar>(i, j) = 0;
        }   
    }
    t = ((double)getTickCount() - t)/getTickFrequency();
    cout << "Times passed in seconds: " << t << endl; 
    imshow("Test", img);
    waitKey(0);
}