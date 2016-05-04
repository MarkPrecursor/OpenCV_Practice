#include <iostream>
#include "opencv2/highgui.hpp"
#include "opencv2/stitching.hpp"

using namespace cv;
using namespace std;

int main()
{
    string srcFile[6]={"1.JPG","2.JPG","3.JPG","4.JPG","5.JPG","6.JPG"};
    string dstFile="result.jpg";
    vector<Mat> imgs;
    for(int i=0;i<6;++i)
    {
         Mat img=imread(srcFile[i]);
         if (img.empty())
         {
             cout<<"Can't read image '"<<srcFile[i]<<"'\n";
             system("pause");
             return -1;
         }
         imgs.push_back(img);
    }
    cout<<"Please wait..."<<endl;
    Mat pano;
    Stitcher stitcher = Stitcher::createDefault(false);
    Stitcher::Status status = stitcher.stitch(imgs, pano);
    if (status != Stitcher::OK)
    {
        cout<<"Can't stitch images, error code=" <<int(status)<<endl;
        system("pause");
        return -1;
    }
    imwrite(dstFile, pano);
//    namedWindow("Result");
//    imshow("Result",pano);

    waitKey(0);

//    destroyWindow("Result");
    system("pause");
    return 0;
}
