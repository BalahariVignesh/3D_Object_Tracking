# SFND 3D Object Tracking

Welcome to the final project of the camera course. By completing all the lessons, you now have a solid understanding of keypoint detectors, descriptors, and methods to match them between successive images. Also, you know how to detect objects in an image using the YOLO deep-learning framework. And finally, you know how to associate regions in a camera image with Lidar points in 3D space. Let's take a look at our program schematic to see what we already have accomplished and what's still missing.

<img src="images/course_code_structure.png" width="779" height="414" />

In this final project, you will implement the missing parts in the schematic. To do this, you will complete four major tasks: 
1. First, you will develop a way to match 3D objects over time by using keypoint correspondences. 
2. Second, you will compute the TTC based on Lidar measurements. 
3. You will then proceed to do the same using the camera, which requires to first associate keypoint matches to regions of interest and then to compute the TTC based on those matches. 
4. And lastly, you will conduct various tests with the framework. Your goal is to identify the most suitable detector/descriptor combination for TTC estimation and also to search for problems that can lead to faulty measurements by the camera or Lidar sensor. In the last course of this Nanodegree, you will learn about the Kalman filter, which is a great way to combine the two independent TTC measurements into an improved version which is much more reliable than a single sensor alone can be. But before we think about such things, let us focus on your final project in the camera course. 

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level project directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./3D_object_tracking`.
---
# FP1-Match 3D Objects
1. Here first I loop through the matches to add all the keypoints belonging in the bounding boxes to a list.
2. Then calculate a score matrix(like a map) between bounding boxes of previous and current frame.
3. Finally, calculate for each bounding box in the current frame, which is the best match from the previous frame using the score matrix i calculated in the previous step.
These steps are achieved using the below code.
```C++
void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    // ...
    //getting the number of bounding boxes from prev and curr datafrate
    int prevImageTotalBoundingBoxes = prevFrame.boundingBoxes.size();
    int currImageTotalBoundingBoxes = currFrame.boundingBoxes.size();

    //creating the map matrix
    int scoreMatrix[prevImageTotalBoundingBoxes][currImageTotalBoundingBoxes]={};

    //iterating through the matches
    for(auto it = matches.begin(); it!=matches.end();it++)
    {
        cv::KeyPoint kptsprevImage = prevFrame.keypoints[it->queryIdx];
        cv::Point pt_prevImage = cv::Point(kptsprevImage.pt.x,kptsprevImage.pt.y);
        std::vector<int> prevImageBoxList;
        for(int i=0;i<prevImageTotalBoundingBoxes;i++)
        {
            if(prevFrame.boundingBoxes[i].roi.contains(pt_prevImage))
                prevImageBoxList.push_back(i);
        }

        cv::KeyPoint kptscurrImage = currFrame.keypoints[it->trainIdx];
        cv::Point pt_currImage = cv::Point(kptscurrImage.pt.x,kptscurrImage.pt.y);
        std::vector<int> currImageBoxList;
        for(int i=0;i<currImageTotalBoundingBoxes;i++)
        {
            if(currFrame.boundingBoxes[i].roi.contains(pt_currImage))
                currImageBoxList.push_back(i);
        }

        //Calculating the box score matrix
        for(auto prev:prevImageBoxList)
        {
            for(auto curr:currImageBoxList)
            scoreMatrix[prev][curr] +=1;
        }

    }
    for(int i=0;i<prevImageTotalBoundingBoxes;i++)
    {
        int maxMatchCount = 0;
        int bestMatch = 0;
        for(int j=0;j<currImageTotalBoundingBoxes;j++)
        {
            if(scoreMatrix[i][j]>maxMatchCount)
            {
                maxMatchCount=scoreMatrix[i][j];
                bestMatch=j;
            }
        }
        bbBestMatches[i]=bestMatch;
    }
}

```
---
# FP2-Compute Lidar-based TTC
The code for computing the Time-To-Collision based on Lidar measurements.
Here I took the lowest X values of Lidar measurement for each of the previous and current frame to calculate the TTC.
Also I removed the outliers by using the laneWidth parameter.
PS:Instead of min X values, I also tried to compute the average of X values to calculate TTC, which I have commented out in the code present in camFusion_Student.cpp file.
```C++
void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // ...
    double dT = 1.0/frameRate;// or use 0.1 - the time between measurements in seconds
    double laneWidth = 4.0;

    //find closest distance to Lidar Points within ego lane
    double minXPrev = 1e9,minXCurr=1e9;
    for(auto it = lidarPointsPrev.begin(); it!=lidarPointsPrev.end();it++)
    {
        if(abs(it->y)<=laneWidth/2.0)
            minXPrev = minXPrev > it->x?it->x:minXPrev;
    }
    for(auto it = lidarPointsCurr.begin(); it!=lidarPointsCurr.end();it++)
    {
        if(abs(it->y)<=laneWidth/2.0)
            minXCurr = minXCurr > it->x?it->x:minXCurr;
    }
    TTC = minXCurr * dT / (minXPrev-minXCurr);
 
}
```
---
# FP3-Associate Keypoint Correspondences with Bounding Boxes
Here I find and cluster all the keypoint matches that belong to each 3D Object.
1. Looping through the KeyPoint Matches to find eucledian distance between the matching keypoints of previous frame and current frame, and store them in a distances vector.
2. Calculate the Eucledian Distance mean value using the distances stored in the vector from previous step.
3. Using a distance threshold, now the outliers are filtered, only valid keypoints and matches are added to the bounding box.
```C++
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // ...
    //First Looping through kptMatches to accumulate distances and find their average.
    std::vector<double> distances;
    for(auto it=kptMatches.begin();it!=kptMatches.end();it++)
    {
        cv::KeyPoint kpt1 = kptsPrev[it->queryIdx];
        cv::KeyPoint kpt2 = kptsCurr[it->trainIdx];

        if (boundingBox.roi.contains(kpt2.pt))
            distances.push_back(cv::norm(kpt2.pt-kpt1.pt));
    }
    double eucledianDistanceMean = accumulate(distances.begin(),distances.end(),0.0)/distances.size();
    double distanceThreshold = 1.5*eucledianDistanceMean;

    //now looping through the matches again to filter the outliers.
    for(auto it = kptMatches.begin();it!= kptMatches.end(); it++)
    {
        cv::KeyPoint kpt1 = kptsPrev[it->queryIdx];
        cv::KeyPoint kpt2 = kptsCurr[it->trainIdx];

        if(boundingBox.roi.contains(kpt2.pt))
        {
            double dist = cv::norm(kpt2.pt-kpt1.pt);
            if(dist<distanceThreshold)
            {
                boundingBox.keypoints.push_back(kpt2);
                boundingBox.kptMatches.push_back(*it);
            }
        }
    }
}
```
---
# FP4-Compute Camera-Based TTC
Here the TTC is computed using those keypoint correspondences that have been matched with bounding boxes of previous and current frame in the last task.

```C++
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // ...
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr and prev frame
    for(auto it1 = kptMatches.begin();it1!=kptMatches.end();++it1)
    {

        //get current keypoint and its matched partner in the previous frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for(auto it2 = kptMatches.begin()+1;it2!=kptMatches.end();++it2)
        {
            double minDist = 100.0;
            //get next keypoint and its matched partner in previous frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            //compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if(distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            {//avoid division by zerp

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);

            }
        }
    }
    //only ccontinue if list of distance ratios is not empty
    if(distRatios.size()==0)
    {
        TTC=NAN;
        return;
    }

    //MeanDistRatio
    std::sort(distRatios.begin(),distRatios.end());
    long medIndex = floor(distRatios.size()/2.0);
    double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex-1]+distRatios[medIndex])/2.0:distRatios[medIndex];
    double dT = 1/frameRate;
    TTC=-dT/(1-medDistRatio);
}
```

---
# FP5-Performance Evaluation 1

The images read were from 0 to 18 - 19 frames in total. Out of which 18 TTC are calculated.
Out of all the 18 TTC, 3 of them seem to deviate wildly because of outliers leading to incorrect calculations.
They are in the tabulation below:
Frame Number | TTC value
-------------|----------
7|34.340420s
12|-10.853745s
17|-9.994236s

These deviations might be due to the outliers and reflective values taken as Xmin to calculate the Time To Collision.

### Frame 7
![Lidar Frame 7](https://github.com/BalahariVignesh/3D_Object_Tracking/blob/main/TTC%20Lidar%20new/7.png)

### Frame 12
![Lidar Frame 12](https://github.com/BalahariVignesh/3D_Object_Tracking/blob/main/TTC%20Lidar%20new/14.png)

### Frame 17
![Lidar Frame 17](https://github.com/BalahariVignesh/3D_Object_Tracking/blob/main/TTC%20Lidar%20new/19.png)

---
# FP6- Performance Evaluation 2
This last exercise is about running the different detector / descriptor combinations and looking at the differences in TTC estimation. Find out which methods perform best and also include several examples where camera-based TTC estimation is way off. As with Lidar, describe your observations again and also look into potential reasons. This is the last task in the final project.

The task is complete once all detector / descriptor combinations implemented in previous chapters have been compared with regard to the TTC estimate on a frame-by-frame basis. To facilitate the comparison, a spreadsheet and graph should be used to represent the different TTCs.
