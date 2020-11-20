#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType;
        if(descriptorType.compare("SIFT")==0)
        {
            normType=cv::NORM_L2;
        }
        else
        {
            //normType=cv::NORM_HAMMING;
            normType=cv::NORM_L2;
        }
        
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        // ...
        if (descSource.type() != CV_32F||descRef.type()!=CV_32F)
        { // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }

        //... TODO : implement FLANN matching
        cout << "FLANN matching";
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)
        double t = (double)cv::getTickCount();
        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << " (NN) with n=" << matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;

        //matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)

        // k nearest neighbors (k=2)
        std::vector<std::vector<cv::DMatch>> knn_matches;
        // TODO : implement k-nearest-neighbor matching
        double t = (double)cv::getTickCount();
        matcher->knnMatch(descSource, descRef, knn_matches,2); // Finds the best match for each descriptor in desc1
        
        

        // TODO : filter matches using descriptor distance ratio test
        const float ratio_thresh = 0.8f;
      
        for(size_t it =0;it<knn_matches.size();it++)
        {
            if(knn_matches[it][0].distance<ratio_thresh*knn_matches[it][1].distance)
            {
                matches.push_back(knn_matches[it][0]);
            }
        }
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << " (KNN) with n=" << knn_matches.size()-matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("BRIEF") == 0)//BRIEF, ORB, FREAK, AKAZE, SIFT
    {

        //...
        int bytes = 32; //legth of the descriptor in bytes, valid values are: 16, 32 (default) or 64 .
        bool use_orientation = false; //sample patterns using keypoints orientation, disabled by default.
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create(bytes,use_orientation);
        
    }
    else if (descriptorType.compare("ORB") == 0)
    {

        //...
        int nfeatures = 500; //The maximum number of features to retain.
        float scaleFactor = 1.2f; //Pyramid decimation ratio, greater than 1. scaleFactor==2 means the classical pyramid, where each next level has 4x less pixels than the previous, but such a big scale factor will degrade feature matching scores dramatically. On the other hand, too close to 1 scale factor will mean that to cover certain scale range you will need more pyramid levels and so the speed will suffer.
        int nlevels = 8; //The number of pyramid levels. The smallest level will have linear size equal to input_image_linear_size/pow(scaleFactor, nlevels - firstLevel).
        int edgeThreshold = 31;//This is size of the border where the features are not detected. It should roughly match the patchSize parameter.
        int firstLevel = 0;//The level of pyramid to put source image to. Previous layers are filled with upscaled source image.
        int WTA_K = 2; //The number of points that produce each element of the oriented BRIEF descriptor. The default value 2 means the BRIEF where we take a random point pair and compare their brightnesses, so we get 0/1 response. Other possible values are 3 and 4. For example, 3 means that we take 3 random points (of course, those point coordinates are random, but they are generated from the pre-defined seed, so each element of BRIEF descriptor is computed deterministically from the pixel rectangle), find point of maximum brightness and output index of the winner (0, 1 or 2). Such output will occupy 2 bits, and therefore it will need a special variant of Hamming distance, denoted as NORM_HAMMING2 (2 bits per bin). When WTA_K=4, we take 4 random points to compute each bin (that will also occupy 2 bits with possible values 0, 1, 2 or 3).
        cv::ORB::ScoreType scoreType = cv::ORB::HARRIS_SCORE; //if FAST_SCORE is used it produces slightly unstable keypoints, but it is a little faster to compute
        int patchSize = 31;//size of the patch used by the oriented BRIEF descriptor. Of course, on smaller pyramid layers the perceived image area covered by a feature will be larger.
        int fastThreshold = 20;//fast threshold
        extractor = cv::ORB::create(nfeatures,scaleFactor,nlevels,edgeThreshold,firstLevel,WTA_K,scoreType,patchSize,fastThreshold);
    }
    else if (descriptorType.compare("FREAK") == 0)
    {

        //...
        bool orientationNormalized = true;//Enable orientation normalization.
        bool scaleNormalized = true;//Enable scale normalization.
        float patternScale = 22.0f;//Scaling of the description pattern.
        int nOctaves = 4;//Number of octaves covered by the detected keypoints.
        const std::vector< int > & 	selectedPairs = std::vector< int >(); //optional
        
        extractor = cv::xfeatures2d::FREAK::create(orientationNormalized,scaleNormalized,patternScale,nOctaves);
    }
    else if (descriptorType.compare("AKAZE") == 0)
    {

        //...
        cv::AKAZE::DescriptorType descriptor_type = cv::AKAZE::DESCRIPTOR_MLDB;//Type of the extracted descriptor: DESCRIPTOR_KAZE, DESCRIPTOR_KAZE_UPRIGHT, DESCRIPTOR_MLDB or DESCRIPTOR_MLDB_UPRIGHT.
        int descriptor_size = 0;//Size of the descriptor in bits. 0 -> Full size
        int descriptor_channels = 3;//	Number of channels in the descriptor (1, 2, 3)
        float threshold = 0.001f;//Detector response threshold to accept point
        int nOctaves = 4;//Maximum octave evolution of the image
        int nOctaveLayers = 4;//Default number of sublevels per scale level
        cv::KAZE::DiffusivityType diffusivity = cv::KAZE::DIFF_PM_G2;//Diffusivity type. DIFF_PM_G1, DIFF_PM_G2, DIFF_WEICKERT or DIFF_CHARBONNIER
        extractor = cv::AKAZE::create(descriptor_type,descriptor_size,descriptor_channels,threshold,nOctaves,nOctaveLayers,diffusivity);

    }
    else if (descriptorType.compare("SIFT") == 0)
    {

        //...
        int nfeatures = 0;//	The number of best features to retain. The features are ranked by their scores (measured in SIFT algorithm as the local contrast)
        int nOctaveLayers = 3;// 	The number of layers in each octave. 3 is the value used in D. Lowe paper. The number of octaves is computed automatically from the image resolution.
        double contrastThreshold = 0.04;//The contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions. The larger the threshold, the less features are produced by the detector.
        double 	edgeThreshold = 10;//	The threshold used to filter out edge-like features. Note that the its meaning is different from the contrastThreshold, i.e. the larger the edgeThreshold, the less features are filtered out (more features are retained).
        double 	sigma = 1.6;//	The sigma of the Gaussian applied to the input image at the octave #0. If your image is captured with a weak camera with soft lenses, you might want to reduce the number.
        //extractor = cv::xfeatures2d::SIFT::create(nfeatures,nOctaveLayers,contrastThreshold,edgeThreshold,sigma);
        extractor = cv::xfeatures2d::SIFT::create();
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}
void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)

    // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    // add corners to result vector
    for (int row=0; row<dst_norm.rows; row++) {
        for (int col=0; col<dst_norm.cols; col++) {
            
            if ((int)dst_norm.at<float>(row,col) > minResponse) {
                cv::KeyPoint temp_KeyPoint;
                temp_KeyPoint.pt = cv::Point2f(col, row);
                temp_KeyPoint.response = dst_norm.at<float>(row, col);
                temp_KeyPoint.size = 2*apertureSize;
                keypoints.push_back(temp_KeyPoint);
            }
        }
    }

    double t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Harris detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }


}
void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{
    if (detectorType.compare("FAST") == 0)
    {
        int threshold = 10;
        bool nonmaxSuppression = true;
        cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16;
        cv::Ptr<cv::FeatureDetector> detector = cv::FastFeatureDetector::create(threshold,nonmaxSuppression,type);
        double t = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << detectorType << " detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    }
    else if (detectorType.compare("BRISK")==0)
    {
        int threshold = 30; //AGAST detection threshold score.
        int octaves = 3; //detection octaves. Use 0 to do single scale.
        float patternScale = 1.0f; //apply this scale to the pattern used for sampling the neighbourhood of a keypoint.
        cv::Ptr<cv::FeatureDetector> detector = cv::BRISK::create(threshold,octaves,patternScale);
        double t = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << detectorType << " detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    }
    else if (detectorType.compare("ORB")==0)
    {
        int nfeatures = 500; //The maximum number of features to retain.
        float scaleFactor = 1.2f; //Pyramid decimation ratio, greater than 1. scaleFactor==2 means the classical pyramid, where each next level has 4x less pixels than the previous, but such a big scale factor will degrade feature matching scores dramatically. On the other hand, too close to 1 scale factor will mean that to cover certain scale range you will need more pyramid levels and so the speed will suffer.
        int nlevels = 8; //The number of pyramid levels. The smallest level will have linear size equal to input_image_linear_size/pow(scaleFactor, nlevels - firstLevel).
        int edgeThreshold = 31;//This is size of the border where the features are not detected. It should roughly match the patchSize parameter.
        int firstLevel = 0;//The level of pyramid to put source image to. Previous layers are filled with upscaled source image.
        int WTA_K = 2; //The number of points that produce each element of the oriented BRIEF descriptor. The default value 2 means the BRIEF where we take a random point pair and compare their brightnesses, so we get 0/1 response. Other possible values are 3 and 4. For example, 3 means that we take 3 random points (of course, those point coordinates are random, but they are generated from the pre-defined seed, so each element of BRIEF descriptor is computed deterministically from the pixel rectangle), find point of maximum brightness and output index of the winner (0, 1 or 2). Such output will occupy 2 bits, and therefore it will need a special variant of Hamming distance, denoted as NORM_HAMMING2 (2 bits per bin). When WTA_K=4, we take 4 random points to compute each bin (that will also occupy 2 bits with possible values 0, 1, 2 or 3).
        cv::ORB::ScoreType scoreType = cv::ORB::HARRIS_SCORE; //if FAST_SCORE is used it produces slightly unstable keypoints, but it is a little faster to compute
        int patchSize = 31;//size of the patch used by the oriented BRIEF descriptor. Of course, on smaller pyramid layers the perceived image area covered by a feature will be larger.
        int fastThreshold = 20;//fast threshold
        
        cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create(nfeatures,scaleFactor,nlevels,edgeThreshold,firstLevel,WTA_K,scoreType,patchSize,fastThreshold);
        double t = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << detectorType << " detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    }
    else if (detectorType.compare("AKAZE")==0)
    {
        cv::AKAZE::DescriptorType descriptor_type = cv::AKAZE::DESCRIPTOR_MLDB;//Type of the extracted descriptor: DESCRIPTOR_KAZE, DESCRIPTOR_KAZE_UPRIGHT, DESCRIPTOR_MLDB or DESCRIPTOR_MLDB_UPRIGHT.
        int descriptor_size = 0;//Size of the descriptor in bits. 0 -> Full size
        int descriptor_channels = 3;//	Number of channels in the descriptor (1, 2, 3)
        float threshold = 0.001f;//Detector response threshold to accept point
        int nOctaves = 4;//Maximum octave evolution of the image
        int nOctaveLayers = 4;//Default number of sublevels per scale level
        cv::KAZE::DiffusivityType diffusivity = cv::KAZE::DIFF_PM_G2;//Diffusivity type. DIFF_PM_G1, DIFF_PM_G2, DIFF_WEICKERT or DIFF_CHARBONNIER
        cv::Ptr<cv::FeatureDetector> detector = cv::AKAZE::create(descriptor_type,descriptor_size,descriptor_channels,threshold,nOctaves,nOctaveLayers,diffusivity);
        double t = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << detectorType << " detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    }
    else if (detectorType.compare("SIFT")==0)
    {
        int nfeatures = 0;//	The number of best features to retain. The features are ranked by their scores (measured in SIFT algorithm as the local contrast)
        int nOctaveLayers = 3;// 	The number of layers in each octave. 3 is the value used in D. Lowe paper. The number of octaves is computed automatically from the image resolution.
        double contrastThreshold = 0.04;//The contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions. The larger the threshold, the less features are produced by the detector.
        double 	edgeThreshold = 10;//	The threshold used to filter out edge-like features. Note that the its meaning is different from the contrastThreshold, i.e. the larger the edgeThreshold, the less features are filtered out (more features are retained).
        double 	sigma = 1.6;//	The sigma of the Gaussian applied to the input image at the octave #0. If your image is captured with a weak camera with soft lenses, you might want to reduce the number.
        //cv::Ptr<cv::FeatureDetector> detector = cv::xfeatures2d::SIFT::create(nfeatures,nOctaveLayers,contrastThreshold,edgeThreshold,sigma);
        cv::Ptr<cv::FeatureDetector> detector = cv::xfeatures2d::SIFT::create();
        double t = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << detectorType << " detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl; 
    }

     if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Results of ";
        windowName.append(detectorType);
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }

}