//#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
//#include "opencv2/videoio.hpp"
#include <iostream>

#include <opencv2/features2d.hpp>

using namespace cv;
using namespace std;

void drawText(Mat & image);

// Timer function
double CLOCK() {
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC,  &t);
	return (t.tv_sec * 1000)+(t.tv_nsec*1e-6);
}


int main( int argc, const char** argv ) {

    int TILE_SIZE = 32;
    double start_cpu, finish_cpu;
    double start_gpu, finish_gpu;
    
    // Check inputs
    if (argc != 3){
        cout << "Incorrect number of inputs" << endl;
        cout << argv[0] << " <input file>" << endl;
        return -1;
    }
    
    // Read input image from argument
    Mat input_image1 = imread(argv[1], IMREAD_COLOR);
    Mat input_image2 = imread(argv[2], IMREAD_COLOR);
    
    if (input_image1.empty()){
        cout << "Image cannot be loaded..!!" << endl;
        return -1;
    }
    if (input_image2.empty()){
        cout << "Image cannot be loaded..!!" << endl;
        return -1;
    }

    // Convert the color image to grayscale image
    cvtColor(input_image1, input_image1, COLOR_BGR2GRAY); 
    cvtColor(input_image2, input_image2, COLOR_BGR2GRAY); 
    
    unsigned int height = input_image1.rows;
    unsigned int  width = input_image1.cols;
    cout << "Image1 size: " << height << "x" << width << endl;
    height = input_image2.rows;
    width = input_image2.cols;
    cout << "Image2 size: " << height << "x" << width << endl;
    
    cv::Ptr<cv::SIFT> siftPtr = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypoints1;
    siftPtr->detect(input_image1, keypoints1);
    std::vector<cv::KeyPoint> keypoints2;
    siftPtr->detect(input_image2, keypoints2);

    DescriptorExtractor* extractor;
    extractor = cv::SiftDescriptorExtractor::create();
    
    Mat descriptors1, descriptors2;
    siftPtr->compute(input_image1, keypoints1, descriptors1);
    siftPtr->compute(input_image2, keypoints2, descriptors2);
  
    // Add results to image and save.
    cv::Mat output1, output2;
    cv::drawKeypoints(input_image1, keypoints1, output1);
    cv::imwrite("sift_result1.jpg", output1);
    cv::drawKeypoints(input_image2, keypoints2, output2);
    cv::imwrite("sift_result2.jpg", output2);

    // Match descriptors using FLANN
    FlannBasedMatcher matcher;
    std::vector< std::vector<DMatch> > knn_matches;
    matcher.knnMatch( descriptors1, descriptors2, knn_matches, 2 );

    std::cout<< knn_matches.size() << endl;
    
    return 0;
}


void drawText(Mat & image)
{
    putText(image, "Hello OpenCV",
            Point(20, 50),
            FONT_HERSHEY_COMPLEX, 1, // font face and scale
            Scalar(255, 255, 255), // white
            1, LINE_AA); // line thickness and type
}
