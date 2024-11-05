#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


using namespace std;
using namespace cv;

double hbVectorsDistance(const double huMoments[7], double contourClass[7]);
int main() {
    string filename = "./images/t3.jpg";

    Mat src = imread(filename, IMREAD_GRAYSCALE);
    Mat thedImage, blurredImage;

    if (!src.data) { 
        cout << "\nUnable to read input image";
        return -1;
    }

    namedWindow("src", WINDOW_NORMAL);
    imshow("src", src);
    
    medianBlur(src, blurredImage, 3);
    threshold(blurredImage, thedImage, 110, 255, THRESH_BINARY);
    imwrite("./images/blurredImage.png",  blurredImage);
    namedWindow("thedImage", WINDOW_NORMAL);
    imshow("thedImage", thedImage);
    imwrite("./images/thedImage.png", thedImage);
    //thedImage=blurredImage;

    vector<vector<double>> contourClasses = {
        {0.49, 2.00, 1.96, 3.79, 6.82, 4.95, -6.80},
        {0.22, 0.49, 2.51, 3.05, 5.92, 3.59, -6.05},
        {0.55, 3.08, 4.90, 5.35, -10.64, 6.90, -10.61},
        {0.44, 1.55, 2.52, 3.76, 7.21, 4.96, 6.95},
        {0.42, 1.40, 2.72, 4.13, 7.55, 4.84, 9.07},
        {0.46, 1.74, 1.89, 2.59, -4.88, -3.49, -5.15}
    };

    vector<vector<Point>> contours;
    findContours(thedImage, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
    cout << "\n\n" << contours.size() << " contours have been drawn\n";

    RNG rng(12345);
    Mat drawing = Mat::zeros(thedImage.size(), CV_8UC3);
    for (int i = 0; i < contours.size(); i++) {
        //Random color
        Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
        //Drawing contours
        drawContours(drawing, contours, i, color, 1);
        //Writting contour index
        putText(drawing, to_string(i),  Point(contours[i][0].x-5, contours[i][0].y-5), FONT_HERSHEY_SIMPLEX,  0.2, color, 1);
    }
    //Save drawing image
    namedWindow("Contours", WINDOW_NORMAL);
    imshow("Contours", drawing);
    imwrite("./images/drawing.png", drawing);

    //Loop for computing Hu  moments
    vector<vector<double>> huMoments(contours.size(), vector<double>(7));
    for (int k = 0; k < contours.size(); k++) {
        Moments momx = moments(contours[k]);
        double hu[7];
        HuMoments(momx, hu);
        cout<<"\nContour "<<k;
        for (int i = 0; i < 7; i++) {
            //Scaling Hu moments
            huMoments[k][i] = -1 * copysign(1.0, hu[i]) * log10(abs(hu[i]));
            printf("\nHu moment %d: %8.21f", i, huMoments[k][i]);
        }
    }


    vector<double> euclideanDistancesDiff(contourClasses.size());
    //Loop for  computing Euclidean distance to each class and get the clolsest one
    for (int i = 0; i < contours.size(); i++) {
        double minDistance = DBL_MAX;
        int closestClass = 1;
        
        for (int k = 0; k < contourClasses.size(); k++) {
            double distance = hbVectorsDistance(huMoments[i].data(), contourClasses[k].data());
            if (distance < minDistance) {
                minDistance = distance;
                closestClass = k+1; //k+1 because classes goes from 1 to 6
            }
        }
        cout << "\nContour " << i << " is closest to class " << closestClass << " with distance " << minDistance << "\n";
    }

    waitKey(0);
    return 0;
}

//Distance between vectors function
double hbVectorsDistance(const double huMoments[7], double contourClass[7]){
    double sum=0;
    for(int i=0; i<7; i++){
        //cout<<"\n"<<huMoments[i]<<"-"<<contourClass[i]<<endl;
        double diff=huMoments[i]-contourClass[i];
        sum+=diff*diff;
    }
    double res=sqrt(sum);
    //cout<<"sum="<<sum<<"\nResult="<<res<<endl;
    return res;
}