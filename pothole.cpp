#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "highgui.h"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;
  
Mat src, dilation_dst, dilation_dst1;

const int w = 500;
int levels = 3;
//vector of points and vector for levels of contours
vector<vector<Point> > contours;
vector<Vec4i> hierarchy;

// all the dialation properties are defined
int dilation_elem = 2;
int dilation_size = 2;
int const max_elem = 2;
int const max_kernel_size = 21;

void Dilation( int, void* );

int main(int argc, char* argv[])
{
 src = imread(argv[1], CV_LOAD_IMAGE_COLOR);
 if( !src.data )
  { 
   return -1; 
  }
resize(src,src, Size(128, 128));
namedWindow( "input", CV_WINDOW_AUTOSIZE);
imshow("input",src);

imwrite("/home/aditya/images/rep1.jpg", src);
  

  createTrackbar( "Element:\n 0: Rect \n 1: Cross \n 2: Ellipse", "Dilation Demo",
                  &dilation_elem, max_elem,
                  Dilation );

  createTrackbar( "Kernel size:\n 2n +1", "Dilation Demo",
                  &dilation_size, max_kernel_size,
                  Dilation );

  Dilation( 0, 0 );
  waitKey(0);
  return 0;

}

void Dilation( int, void* )
{
  //namedWindow( "final output", CV_WINDOW_NORMAL);
 
  int dilation_type;
  if( dilation_elem == 0 ){ dilation_type = MORPH_RECT; }
  else if( dilation_elem == 1 ){ dilation_type = MORPH_CROSS; }
  else if( dilation_elem == 2) { dilation_type = MORPH_ELLIPSE; }
  //the structuring element for dialation is defined
  Mat element1 = getStructuringElement( MORPH_RECT,
                                       Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                       Point( dilation_size, dilation_size ) );
    
 
   
  dilate(src, dilation_dst, element1 );
  //imshow("dilated",dilation_dst);
  
  //the structuring element for erosion is defined
  Mat element2 = getStructuringElement( MORPH_RECT,
                                       Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                      Point( dilation_size, dilation_size ) );

  erode(dilation_dst, dilation_dst, element2);
  //imshow("eroded",dilation_dst);
  //imwrite("/home/aditya/images/rep2.jpg", dilation_dst);
  
 
    Mat p = Mat::zeros(dilation_dst.cols*dilation_dst.rows, 5, CV_32F);
    Mat bestLabels, centers, clustered;
    vector<Mat> bgr;
    cv::split(dilation_dst, bgr);
    // i think there is a better way to split pixel bgr color
    for(int i=0; i<dilation_dst.cols*dilation_dst.rows; i++) {
        p.at<float>(i,0) = (i/dilation_dst.cols) / dilation_dst.rows;
        p.at<float>(i,1) = (i%dilation_dst.cols) / dilation_dst.cols;
        p.at<float>(i,2) = bgr[0].data[i] / 255.0;
        p.at<float>(i,3) = bgr[1].data[i] / 255.0;
        p.at<float>(i,4) = bgr[2].data[i] / 255.0;
    }

    int K = 8;
    cv::kmeans(p, K, bestLabels,
            TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0),
            3, KMEANS_PP_CENTERS, centers);

    int colors[K];
    for(int i=0; i<K; i++) {
        colors[i] = 255/(i+1);
    }
    // i think there is a better way to do this mayebe some Mat::reshape?
    clustered = Mat(dilation_dst.rows, dilation_dst.cols, CV_32F);
    for(int i=0; i<dilation_dst.cols*dilation_dst.rows; i++) {
        clustered.at<float>(i/dilation_dst.cols, i%dilation_dst.cols) = (float)(colors[bestLabels.at<int>(0,i)]);
//      cout << bestLabels.at<int>(0,i) << " " << 
//              colors[bestLabels.at<int>(0,i)] << " " << 
//              clustered.at<float>(i/src.cols, i%src.cols) << " " <<
//              endl;
    }

    clustered.convertTo(clustered, CV_8UC3);
    imshow("clustered", clustered);

  cvtColor(dilation_dst,dilation_dst,CV_BGR2GRAY);
   
  
//applying thresholding
 cv::threshold(dilation_dst,dilation_dst, 178, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

 imshow( "final output1", dilation_dst );
 
  int l,m;
  Mat draw1 = src.clone();
  
 
  int flag = 0;

   
}


//https://isic-archive.com/#images
//https://machinelearningmastery.com/implement-random-forest-scratch-python/
 //pothole-4fc96 

//IJTRA

















