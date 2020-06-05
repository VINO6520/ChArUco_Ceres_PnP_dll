#include"pch.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <io.h>
#include <string>
#include <iostream>  
#include <fstream> 
#include <csignal>
#include <windows.h>
#include <vector>
#include <list>
#include <algorithm>
#include <iterator>
#include <cstdio>
#include <dirent.h>
#include <cmath>
#include <limits>
#include <thread>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\calib3d\calib3d.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <ceres\ceres.h>
#include <sys/timeb.h>
#include <sys/timeb.h>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;


#if !defined MATCHER
#define MATCHER




#define NOCHECK      0
#define CROSSCHECK   1
#define RATIOCHECK   2
#define BOTHCHECK    3

class RobustMatcher {

private:

	// pointer to the feature point detector object
	cv::Ptr<cv::FeatureDetector> detector;
	// pointer to the feature descriptor extractor object
	cv::Ptr<cv::DescriptorExtractor> descriptor;
	int normType;
	float ratio; // max ratio between 1st and 2nd NN
	bool refineF; // if true will refine the F matrix
	bool refineM; // if true will refine the matches (will refine F also)
	double distance; // min distance to epipolar
	double confidence; // confidence level (probability)

public:

	RobustMatcher(const cv::Ptr<cv::FeatureDetector> &detector,
		const cv::Ptr<cv::DescriptorExtractor> &descriptor = cv::Ptr<cv::DescriptorExtractor>())
		: detector(detector), descriptor(descriptor), normType(cv::NORM_L2),
		ratio(0.5f), refineF(true), refineM(true), confidence(0.98), distance(1.0) {

		// in this case use the associated descriptor
		if (!this->descriptor) {
			this->descriptor = this->detector;
		}
	}

	// Set the feature detector
	void setFeatureDetector(const cv::Ptr<cv::FeatureDetector>& detect) {

		this->detector = detect;
	}

	// Set descriptor extractor
	void setDescriptorExtractor(const cv::Ptr<cv::DescriptorExtractor>& desc) {

		this->descriptor = desc;
	}

	// Set the norm to be used for matching
	void setNormType(int norm) {

		normType = norm;
	}

	// Set the minimum distance to epipolar in RANSAC
	void setMinDistanceToEpipolar(double d) {

		distance = d;
	}

	// Set confidence level in RANSAC
	void setConfidenceLevel(double c) {

		confidence = c;
	}

	// Set the NN ratio
	void setRatio(float r) {

		ratio = r;
	}

	// if you want the F matrix to be recalculated
	void refineFundamental(bool flag) {

		refineF = flag;
	}

	// if you want the matches to be refined using F
	void refineMatches(bool flag) {

		refineM = flag;
	}

	// Clear matches for which NN ratio is > than threshold
	// return the number of removed points 
	// (corresponding entries being cleared, i.e. size will be 0)
	int ratioTest(const std::vector<std::vector<cv::DMatch> >& inputMatches,
		std::vector<cv::DMatch>& outputMatches) {

		int removed = 0;

		// for all matches
		for (std::vector<std::vector<cv::DMatch> >::const_iterator matchIterator = inputMatches.begin();
			matchIterator != inputMatches.end(); ++matchIterator) {

			//   first best match/second best match
			if ((matchIterator->size() > 1) && // if 2 NN has been identified 
				(*matchIterator)[0].distance / (*matchIterator)[1].distance < ratio) {

				// it is an acceptable match
				outputMatches.push_back((*matchIterator)[0]);

			}
			else {

				removed++;
			}
		}

		return removed;
	}

	// Insert symmetrical matches in symMatches vector
	void symmetryTest(const std::vector<cv::DMatch>& matches1,
		const std::vector<cv::DMatch>& matches2,
		std::vector<cv::DMatch>& symMatches) {

		// for all matches image 1 -> image 2
		for (std::vector<cv::DMatch>::const_iterator matchIterator1 = matches1.begin();
			matchIterator1 != matches1.end(); ++matchIterator1) {

			// for all matches image 2 -> image 1
			for (std::vector<cv::DMatch>::const_iterator matchIterator2 = matches2.begin();
				matchIterator2 != matches2.end(); ++matchIterator2) {

				// Match symmetry test
				if (matchIterator1->queryIdx == matchIterator2->trainIdx  &&
					matchIterator2->queryIdx == matchIterator1->trainIdx) {

					// add symmetrical match
					symMatches.push_back(*matchIterator1);
					break; // next match in image 1 -> image 2
				}
			}
		}
	}

	// Apply both ratio and symmetry test
	// (often an over-kill)
	void ratioAndSymmetryTest(const std::vector<std::vector<cv::DMatch> >& matches1,
		const std::vector<std::vector<cv::DMatch> >& matches2,
		std::vector<cv::DMatch>& outputMatches) {

		// Remove matches for which NN ratio is > than threshold

		// clean image 1 -> image 2 matches
		std::vector<cv::DMatch> ratioMatches1;
		int removed = ratioTest(matches1, ratioMatches1);
		//std::cout << "Number of matched points 1->2 (ratio test) : " << ratioMatches1.size() << std::endl;
		// clean image 2 -> image 1 matches
		std::vector<cv::DMatch> ratioMatches2;
		removed = ratioTest(matches2, ratioMatches2);
		//std::cout << "Number of matched points 1->2 (ratio test) : " << ratioMatches2.size() << std::endl;

		// Remove non-symmetrical matches
		symmetryTest(ratioMatches1, ratioMatches2, outputMatches);

		//std::cout << "Number of matched points (symmetry test): " << outputMatches.size() << std::endl;
	}

	// Identify good matches using RANSAC
	// Return fundamental matrix and output matches
	cv::Mat ransacTest(const std::vector<cv::DMatch>& matches,
		std::vector<cv::KeyPoint>& keypoints1,
		std::vector<cv::KeyPoint>& keypoints2,
		std::vector<cv::DMatch>& outMatches) {

		// Convert keypoints into Point2f	
		std::vector<cv::Point2f> points1, points2;

		for (std::vector<cv::DMatch>::const_iterator it = matches.begin();
			it != matches.end(); ++it) {

			// Get the position of left keypoints
			points1.push_back(keypoints1[it->queryIdx].pt);
			// Get the position of right keypoints
			points2.push_back(keypoints2[it->trainIdx].pt);
		}
		cv::Mat fundamental;
		if (points1.size() > 30) {
			// Compute F matrix using RANSAC
			std::vector<uchar> inliers(points1.size(), 0);
			fundamental = cv::findFundamentalMat(
				points1, points2, // matching points
				inliers,         // match status (inlier or outlier)  
				cv::FM_RANSAC,   // RANSAC method
				distance,        // distance to epipolar line
				confidence);     // confidence probability

								 // extract the surviving (inliers) matches
			std::vector<uchar>::const_iterator itIn = inliers.begin();
			std::vector<cv::DMatch>::const_iterator itM = matches.begin();
			// for all matches
			for (; itIn != inliers.end(); ++itIn, ++itM) {

				if (*itIn) { // it is a valid match

					outMatches.push_back(*itM);
				}
			}

			if (refineF || refineM) {
				// The F matrix will be recomputed with all accepted matches

				// Convert keypoints into Point2f for final F computation	
				points1.clear();
				points2.clear();

				for (std::vector<cv::DMatch>::const_iterator it = outMatches.begin();
					it != outMatches.end(); ++it) {

					// Get the position of left keypoints
					points1.push_back(keypoints1[it->queryIdx].pt);
					// Get the position of right keypoints
					points2.push_back(keypoints2[it->trainIdx].pt);
				}

				// Compute 8-point F from all accepted matches
				fundamental = cv::findFundamentalMat(
					points1, points2, // matching points
					cv::FM_8POINT); // 8-point method

				//std::cout << "points1 的大小" << points1.size() << std::endl;

				if (refineM) {

					std::vector<cv::Point2f> newPoints1, newPoints2;
					// refine the matches
					correctMatches(fundamental,             // F matrix
						points1, points2,        // original position
						newPoints1, newPoints2); // new position
					for (int i = 0; i < points1.size(); i++) {

						/*std::cout << "(" << keypoints1[outMatches[i].queryIdx].pt.x
							<< "," << keypoints1[outMatches[i].queryIdx].pt.y
							<< ") -> ";
						std::cout << "(" << newPoints1[i].x
							<< "," << newPoints1[i].y << std::endl;
						std::cout << "(" << keypoints2[outMatches[i].trainIdx].pt.x
							<< "," << keypoints2[outMatches[i].trainIdx].pt.y
							<< ") -> ";
						std::cout << "(" << newPoints2[i].x
							<< "," << newPoints2[i].y << std::endl;*/

						keypoints1[outMatches[i].queryIdx].pt.x = newPoints1[i].x;
						keypoints1[outMatches[i].queryIdx].pt.y = newPoints1[i].y;
						keypoints2[outMatches[i].trainIdx].pt.x = newPoints2[i].x;
						keypoints2[outMatches[i].trainIdx].pt.y = newPoints2[i].y;
					}
				}
			}
		}

		return fundamental;
	}

	// Match feature points using RANSAC
	// returns fundamental matrix and output match set
	cv::Mat match(cv::Mat& image1, cv::Mat& image2, // input images 
		std::vector<cv::DMatch>& matches, // output matches and keypoints
		std::vector<cv::KeyPoint>& keypoints1, std::vector<cv::KeyPoint>& keypoints2,
		int check = BOTHCHECK) {  // check type (symmetry or ratio or none or both)

								   // 1. Detection of the feature points
		detector->detect(image1, keypoints1);
		detector->detect(image2, keypoints2);

		//std::cout << "Number of feature points (1): " << keypoints1.size() << std::endl;
		//std::cout << "Number of feature points (2): " << keypoints2.size() << std::endl;

		// 2. Extraction of the feature descriptors
		cv::Mat descriptors1, descriptors2;
		descriptor->compute(image1, keypoints1, descriptors1);
		descriptor->compute(image2, keypoints2, descriptors2);

		//std::cout << "descriptor matrix size: " << descriptors1.rows << " by " << descriptors1.cols << std::endl;

		// 3. Match the two image descriptors
		//    (optionaly apply some checking method)

		// Construction of the matcher with crosscheck 
		cv::BFMatcher matcher(normType,            //distance measure
			check == CROSSCHECK);  // crosscheck flag

								   // vectors of matches
		std::vector<std::vector<cv::DMatch> > matches1;
		std::vector<std::vector<cv::DMatch> > matches2;
		std::vector<cv::DMatch> outputMatches;

		// call knnMatch if ratio check is required
		if (check == RATIOCHECK || check == BOTHCHECK) {
			// from image 1 to image 2
			// based on k nearest neighbours (with k=2)
			matcher.knnMatch(descriptors1, descriptors2,
				matches1, // vector of matches (up to 2 per entry) 
				2);		  // return 2 nearest neighbours

			//std::cout << "Number of matched points 1->2: " << matches1.size() << std::endl;

			if (check == BOTHCHECK) {
				// from image 2 to image 1
				// based on k nearest neighbours (with k=2)
				matcher.knnMatch(descriptors2, descriptors1,
					matches2, // vector of matches (up to 2 per entry) 
					2);		  // return 2 nearest neighbours

				//std::cout << "Number of matched points 2->1: " << matches2.size() << std::endl;
			}

		}

		// select check method
		switch (check) {

		case CROSSCHECK:
			matcher.match(descriptors1, descriptors2, outputMatches);
			//std::cout << "Number of matched points 1->2 (after cross-check): " << outputMatches.size() << std::endl;
			break;
		case RATIOCHECK:
			ratioTest(matches1, outputMatches);
			//std::cout << "Number of matched points 1->2 (after ratio test): " << outputMatches.size() << std::endl;
			break;
		case BOTHCHECK:
			ratioAndSymmetryTest(matches1, matches2, outputMatches);
			//std::cout << "Number of matched points 1->2 (after ratio and cross-check): " << outputMatches.size() << std::endl;
			break;
		case NOCHECK:
		default:
			matcher.match(descriptors1, descriptors2, outputMatches);
			//std::cout << "Number of matched points 1->2: " << outputMatches.size() << std::endl;
			break;
		}

		// 4. Validate matches using RANSAC
		cv::Mat fundamental = ransacTest(outputMatches, keypoints1, keypoints2, matches);
		//std::cout << "Number of matched points (after RANSAC): " << matches.size() << std::endl;

		// return the found fundamental matrix
		return fundamental;
	}

	// Match feature points using RANSAC
	// returns fundamental matrix and output match set
	// this is the simplified version presented in the book
	cv::Mat matchBook(cv::Mat& image1, cv::Mat& image2, // input images 
		std::vector<cv::DMatch>& matches, // output matches and keypoints
		std::vector<cv::KeyPoint>& keypoints1, std::vector<cv::KeyPoint>& keypoints2) {

		// 1. Detection of the feature points
		detector->detect(image1, keypoints1);
		detector->detect(image2, keypoints2);

		// 2. Extraction of the feature descriptors
		cv::Mat descriptors1, descriptors2;
		descriptor->compute(image1, keypoints1, descriptors1);
		descriptor->compute(image2, keypoints2, descriptors2);

		// 3. Match the two image descriptors
		//    (optionnally apply some checking method)

		// Construction of the matcher with crosscheck 
		cv::BFMatcher matcher(normType,   //distance measure
			true);      // crosscheck flag

						// match descriptors
		std::vector<cv::DMatch> outputMatches;
		matcher.match(descriptors1, descriptors2, outputMatches);

		// 4. Validate matches using RANSAC
		cv::Mat fundamental = ransacTest(outputMatches, keypoints1, keypoints2, matches);

		// return the found fundemental matrix
		return fundamental;
	}

};

#endif





///////////////////////////////////////////////////////////////
//将空间点绕Z轴旋转
//输入参数 x y为空间点原始x y坐标
//thetaz为空间点绕Z轴旋转多少度，角度制范围在-180到180
//outx outy为旋转后的结果坐标
template<typename T>
inline void codeRotateByZ(T x, T y, T thetaz, T& outx, T& outy)
{
	T x1 = x;//将变量拷贝一次，保证&x == &outx这种情况下也能计算正确
	T y1 = y;
	T rz = thetaz * CV_PI / 180;
	outx = cos(rz) * x1 - sin(rz) * y1;
	outy = sin(rz) * x1 + cos(rz) * y1;
}

//将空间点绕Y轴旋转
//输入参数 x z为空间点原始x z坐标
//thetay为空间点绕Y轴旋转多少度，角度制范围在-180到180
//outx outz为旋转后的结果坐标
template<typename T>
inline void codeRotateByY(T x, T z, T thetay, T& outx, T& outz)
{
	T x1 = x;
	T z1 = z;
	T ry = thetay * CV_PI / 180;
	outx = cos(ry) * x1 + sin(ry) * z1;
	outz = cos(ry) * z1 - sin(ry) * x1;
}

//将空间点绕X轴旋转
//输入参数 y z为空间点原始y z坐标
//thetax为空间点绕X轴旋转多少度，角度制，范围在-180到180
//outy outz为旋转后的结果坐标
template<typename T>
inline void codeRotateByX(T y, T z, T thetax, T& outy, T& outz)
{
	T y1 = y;//将变量拷贝一次，保证&y == &y这种情况下也能计算正确
	T z1 = z;
	T rx = thetax * CV_PI / 180;
	outy = cos(rx) * y1 - sin(rx) * z1;
	outz = cos(rx) * z1 + sin(rx) * y1;
}


//点乘
template<typename T>
inline T DotProduct(const T x[3], const T y[3]) {
	return (x[0] * y[0] + x[1] * y[1] + x[2] * y[2]);
}
//叉乘
template<typename T>
inline void CrossProduct(const T x[3], const T y[3], T result[3]) {
	result[0] = x[1] * y[2] - x[2] * y[1];
	result[1] = x[2] * y[0] - x[0] * y[2];
	result[2] = x[0] * y[1] - x[1] * y[0];
}


template<typename T>
//矩阵相乘
void matrix_Mul(const T src1[16], const T src2[16], T dst[16], int PointNum, int two, int two_B)
{
	T sub = T(0.0);
	int i, j, k;
	for (i = 0; i < PointNum; i++)
	{
		for (j = 0; j < two_B; j++)
		{
			sub = T(0.0);
			for (k = 0; k < two; k++)
			{
				sub += src1[i*two + k] * src2[k*two_B + j];
			}
			dst[i*two_B + j] = sub;
		}
	}
}

//旋转矩阵到向量
template<typename T>
void Rotationmatrix2AngleAxis(const T src[9], T dst[3])
{
	const T R_trace = src[0] + src[4] + src[8];
	const T theta = acos((R_trace - T(1))*T(0.5));
	T Right[9];
	// 0 1 2   0 3 6
	// 3 4 5   1 4 7
	// 6 7 8   2 5 8
	Right[1] = (src[1] - src[3])*T(0.5) / sin(theta);
	Right[2] = (src[2] - src[6])*T(0.5) / sin(theta);
	Right[3] = (src[3] - src[1])*T(0.5) / sin(theta);
	Right[5] = (src[5] - src[7])*T(0.5) / sin(theta);
	Right[6] = (src[6] - src[2])*T(0.5) / sin(theta);
	Right[7] = (src[7] - src[5])*T(0.5) / sin(theta);

	dst[0] = (Right[7] - Right[5])*T(0.5)*theta;
	dst[1] = (Right[2] - Right[6])*T(0.5)*theta;
	dst[2] = (Right[3] - Right[1])*T(0.5)*theta;

}

//向量到矩阵
template<typename T>
void AngleAxis2Rotationmatrix(const T src[3], T dst[9])
{
	const T theta2 = DotProduct(src, src);

	if (theta2 > T(std::numeric_limits<double>::epsilon())) {

		const T theta = sqrt(theta2);
		const T c = cos(theta);
		const T c1 = 1. - c;
		const T s = sin(theta);
		const T theta_inverse = 1.0 / theta;

		const T w[3] = { src[0] * theta_inverse,
			src[1] * theta_inverse,
			src[2] * theta_inverse };

		dst[0] = c * T(1) + c1 * w[0] * w[0] + s * T(0);
		dst[1] = c * T(0) + c1 * w[0] * w[1] + s * T(-w[2]);
		dst[2] = c * T(0) + c1 * w[0] * w[2] + s * T(w[1]);
		dst[3] = c * T(0) + c1 * w[0] * w[1] + s * T(w[2]);
		dst[4] = c * T(1) + c1 * w[1] * w[1] + s * T(0);
		dst[5] = c * T(0) + c1 * w[1] * w[2] + s * -w[0];
		dst[6] = c * T(0) + c1 * w[0] * w[2] + s * -w[1];
		dst[7] = c * T(0) + c1 * w[1] * w[2] + s * w[0];
		dst[8] = c * T(1) + c1 * w[2] * w[2] + s * T(0);
	}
	else {//角度非常微小的情况
		dst[0] = T(1);
		dst[1] = T(0);
		dst[2] = T(0);
		dst[3] = T(0);
		dst[4] = T(1);
		dst[5] = T(0);
		dst[6] = T(0);
		dst[7] = T(0);
		dst[8] = T(1);
	}
}
template<typename T>
void AngleAxisRotatePoint(const T angle_axis[3], const T pt[3], T result[3])
{
	const T theta2 = DotProduct(angle_axis, angle_axis);//angle_axis[3]为旋转向量，DotProduct()点乘求解模长的平方，计算角度大小，

	if (theta2 > T(std::numeric_limits<double>::epsilon())) {//确保 theta2 跟0比足够大，否则开方后还是0，无法取倒数

		const T theta = sqrt(theta2);
		const T costheta = cos(theta);
		const T sintheta = sin(theta);
		const T theta_inverse = 1.0 / theta;

		const T w[3] = { angle_axis[0] * theta_inverse,
			angle_axis[1] * theta_inverse,
			angle_axis[2] * theta_inverse };

		T w_cross_pt[3];
		CrossProduct(w, pt, w_cross_pt);//叉乘

		const T tmp = DotProduct(w, pt) * (T(1.0) - costheta);

		result[0] = pt[0] * costheta + w_cross_pt[0] * sintheta + w[0] * tmp;
		result[1] = pt[1] * costheta + w_cross_pt[1] * sintheta + w[1] * tmp;
		result[2] = pt[2] * costheta + w_cross_pt[2] * sintheta + w[2] * tmp;
	}

	else {//角度非常微小的情况，与0接近,利用一阶泰勒近似

		T w_cross_pt[3];
		CrossProduct(angle_axis, pt, w_cross_pt);

		result[0] = pt[0] + w_cross_pt[0];
		result[1] = pt[1] + w_cross_pt[1];
		result[2] = pt[2] + w_cross_pt[2];
	}
}


struct cost_function_define
{
	cost_function_define(Point3f P0, Point3f P1, Point2f uv0, Point2f uv1, Mat KL, Mat KR, Mat DL, Mat DR, Mat T01)
		:_P0(P0), _P1(P1), _uv0(uv0), _uv1(uv1), _KL(KL), _KR(KR), _DL(DL), _DR(DR), _T01(T01) {}
	template<typename T>
	bool operator()(const T*  const cere_r, const T* const cere_t, T* residual)const
	{
		//第一步，根据 左相机位姿 和 外参 得到右相机

		//利用手写的罗德里格斯公式将旋转向量 cere_r 转成矩阵 Rc0
		T Rc0[9];
		AngleAxis2Rotationmatrix(cere_r, Rc0);

		T TL[16];

		//把左相机位姿拼接成4*4矩阵，一维数组表示；
		TL[0] = Rc0[0]; TL[1] = Rc0[1]; TL[2] = Rc0[2]; TL[3] = cere_t[0];
		TL[4] = Rc0[3]; TL[5] = Rc0[4]; TL[6] = Rc0[5]; TL[7] = cere_t[1];
		TL[8] = Rc0[6]; TL[9] = Rc0[7]; TL[10] = Rc0[8]; TL[11] = cere_t[2];
		TL[12] = T(0);  TL[13] = T(0);  TL[14] = T(0);   TL[15] = T(1);

		//准备外参矩阵
		T T0_1[16];
		T0_1[0] = T(_T01.at<double>(0, 0));
		T0_1[1] = T(_T01.at<double>(0, 1));
		T0_1[2] = T(_T01.at<double>(0, 2));
		T0_1[3] = T(_T01.at<double>(0, 3));

		T0_1[4] = T(_T01.at<double>(1, 0));
		T0_1[5] = T(_T01.at<double>(1, 1));
		T0_1[6] = T(_T01.at<double>(1, 2));
		T0_1[7] = T(_T01.at<double>(1, 3));

		T0_1[8] = T(_T01.at<double>(2, 0));
		T0_1[9] = T(_T01.at<double>(2, 1));
		T0_1[10] = T(_T01.at<double>(2, 2));
		T0_1[11] = T(_T01.at<double>(2, 3));

		T0_1[12] = T(_T01.at<double>(3, 0));
		T0_1[13] = T(_T01.at<double>(3, 1));
		T0_1[14] = T(_T01.at<double>(3, 2));
		T0_1[15] = T(_T01.at<double>(3, 3));

		//外参矩阵与左相机位姿矩阵相乘得到右相机位姿矩阵
		T TR[16];
		matrix_Mul(T0_1, TL, TR, 4, 4, 4);

		//右相机旋转矩阵，旋转向量，位移
		T Rc1[9]; T Rc1_v[3]; T tc1[3];
		Rc1[0] = TR[0]; Rc1[1] = TR[1]; Rc1[2] = TR[2]; tc1[0] = TR[3];
		Rc1[3] = TR[4]; Rc1[4] = TR[5]; Rc1[5] = TR[6]; tc1[1] = TR[7];
		Rc1[6] = TR[8]; Rc1[7] = TR[9]; Rc1[8] = TR[10]; tc1[2] = TR[11];

		//右相机旋转矩阵 转为 旋转向量
		Rotationmatrix2AngleAxis(Rc1, Rc1_v);

		//第二步 准备投影,做残差
		//左右相机三d点
		T p0_1[3], p1_1[3];//points of world
		T p0_2[3], p1_2[3];//point in camera  coordinate system 

		p0_1[0] = T(_P0.x);
		p0_1[1] = T(_P0.y);
		p0_1[2] = T(_P0.z);

		p1_1[0] = T(_P1.x);
		p1_1[1] = T(_P1.y);
		p1_1[2] = T(_P1.z);

		//cout << "point_3d: " << p_1[0] << " " << p_1[1] << "  " << p_1[2] << endl;
		// 将世界坐标系中的特征点转换到相机坐标系中
		AngleAxisRotatePoint(cere_r, p0_1, p0_2);
		AngleAxisRotatePoint(Rc1_v, p1_1, p1_2);

		p0_2[0] = p0_2[0] + cere_t[0];
		p0_2[1] = p0_2[1] + cere_t[1];
		p0_2[2] = p0_2[2] + cere_t[2];

		p1_2[0] = p1_2[0] + tc1[0];
		p1_2[1] = p1_2[1] + tc1[1];
		p1_2[2] = p1_2[2] + tc1[2];

		const T x0 = p0_2[0] / p0_2[2];
		const T y0 = p0_2[1] / p0_2[2];

		const T x1 = p1_2[0] / p1_2[2];
		const T y1 = p1_2[1] / p1_2[2];

		T DL1 = T(_DL.at<double>(0, 0));
		T Dl2 = T(_DL.at<double>(0, 1));

		T DR1 = T(_DR.at<double>(0, 0));
		T DR2 = T(_DR.at<double>(0, 1));

		T r2_L = x0 * x0 + y0 * y0;
		T r2_R = x1 * x1 + y1 * y1;

		T distortion_L = T(1.0) + r2_L * (DL1 + Dl2 * r2_L);
		T distortion_R = T(1.0) + r2_R * (DR1 + DR2 * r2_R);

		//三维点重投影计算的像素坐标
		const T u0 = x0 * distortion_L*_KL.at<double>(0, 0) + _KL.at<double>(0, 2);
		const T v0 = y0 * distortion_L*_KL.at<double>(1, 1) + _KL.at<double>(1, 2);

		const T u1 = x1 * distortion_L*_KR.at<double>(0, 0) + _KR.at<double>(0, 2);
		const T v1 = y1 * distortion_L*_KR.at<double>(1, 1) + _KR.at<double>(1, 2);

		//观测的在图像坐标下的值
		T uv0_u = T(_uv0.x);
		T uv0_v = T(_uv0.y);

		T uv1_u = T(_uv1.x);
		T uv1_v = T(_uv1.y);

		residual[0] = uv0_u - u0 + uv1_u - u1;
		residual[1] = uv0_v - v0 + uv1_v - v1;

		return true;
	}
	Point3f _P0, _P1;
	Point2f _uv0, _uv1;
	Mat _KL, _KR, _DL, _DR;
	Mat _T01;
};



struct mono_cost_function_define
{
	mono_cost_function_define(Point3f P0, Point2f uv0, Mat KL, Mat DL)
		:_P0(P0), _uv0(uv0), _KL(KL), _DL(DL) {}
	template<typename T>
	bool operator()(const T*  const cere_r, const T* const cere_t, T* residual)const
	{

		// 准备投影,做残差
		//相机三d点
		T p0_1[3];//points of world
		T p0_2[3];//point in camera  coordinate system 

		p0_1[0] = T(_P0.x);
		p0_1[1] = T(_P0.y);
		p0_1[2] = T(_P0.z);

		//cout << "point_3d: " << p_1[0] << " " << p_1[1] << "  " << p_1[2] << endl;
		// 将世界坐标系中的特征点转换到相机坐标系中
		AngleAxisRotatePoint(cere_r, p0_1, p0_2);

		p0_2[0] = p0_2[0] + cere_t[0];
		p0_2[1] = p0_2[1] + cere_t[1];
		p0_2[2] = p0_2[2] + cere_t[2];


		const T x0 = p0_2[0] / p0_2[2];
		const T y0 = p0_2[1] / p0_2[2];


		T DL1 = T(_DL.at<double>(0, 0));
		T Dl2 = T(_DL.at<double>(0, 1));

		T r2_L = x0 * x0 + y0 * y0;

		T distortion_L = T(1.0) + r2_L * (DL1 + Dl2 * r2_L);


		//三维点重投影计算的像素坐标
		const T u0 = x0 * distortion_L*_KL.at<double>(0, 0) + _KL.at<double>(0, 2);
		const T v0 = y0 * distortion_L*_KL.at<double>(1, 1) + _KL.at<double>(1, 2);


		//观测的在图像坐标下的值
		T uv0_u = T(_uv0.x);
		T uv0_v = T(_uv0.y);


		residual[0] = uv0_u - u0;
		residual[1] = uv0_v - v0;

		return true;
	}
	Point3f _P0;
	Point2f _uv0;
	Mat _KL, _DL;

};

#if !defined VRSLAM
#define VRSLAM

class vrslam {

public:

	int64_t getSystemTime()//时间戳函数，返回值最好是int64_t，long long也可以
	{
		struct timeb t;
		ftime(&t);
		return 1000 * t.time + t.millitm;
	}

	/***************** Mat转vector **********************/
	template<typename _Tp>
	vector<_Tp> convertMat2Vector(const Mat &mat, int m, int n)
	{
		return (vector<_Tp>)(mat.reshape(m, n));//通道数不变，按行转为一行 m=1,n=1
	}

	template<typename _Tp>
	cv::Mat convertVector2Mat(vector<_Tp> v, int channels, int rows)
	{
		cv::Mat mat = cv::Mat(v);//将vector变成单列的mat
		cv::Mat dest = mat.reshape(channels, rows).clone();//PS：必须clone()一份，否则返回出错
		return dest;
	}


	/************检测元素是否在向量Vector中****************/
	bool is_element_in_vector(vector<int> v, int element)
	{
		vector<int>::iterator it;
		it = find(v.begin(), v.end(), element);
		if (it != v.end()) {
			return true;
		}
		else {
			return false;
		}
	}

	void read_camera_param(string &path_camera_yaml, Mat &cameraMatrix, Mat &cameraDistCoeffs) {

		cv::FileStorage fs(path_camera_yaml, cv::FileStorage::READ);
		if (!fs.isOpened()) // failed
		{
			cout << "Open File Failed!" << endl;
		}

		else // succeed
		{
			cout << "succeed open camera.yml" << endl;
			char buf1[100];
			sprintf_s(buf1, "camera_matrix");
			char buf2[100];
			sprintf_s(buf2, "distortion_coefficients");
			fs[buf1] >> cameraMatrix;
			fs[buf2] >> cameraDistCoeffs;
			fs.release();
			cout << cameraMatrix << endl;
			cout << cameraDistCoeffs << endl;
		}
	}

	void saveImage(cv::Mat image, string &outPutPath, int index)// 保存spinview转换mat后的数据
	{
		//定义保存图像的名字
		string strSaveName;
		char buffer[256];
		sprintf_s(buffer, "D%04d", index);
		strSaveName = buffer;

		//定义保存图像的完整路径
		string strImgSavePath = outPutPath + "\\" + strSaveName;
		//定义保存图像的格式
		strImgSavePath += ".jpg";
		//strImgSavePath += ".png
		ostringstream imgname;
		//保存操作
		imwrite(strImgSavePath.c_str(), image);
	}




	bool getImages(vector<string>& files, string &path)
	{

		DIR *dp;
		struct dirent *dirp;
		if ((dp = opendir(path.c_str())) == NULL) {
			cout << "failed to get the ...images!" << endl;
			return false;;
		}
		while ((dirp = readdir(dp)) != NULL) {
			string name = string(dirp->d_name);
			//cout<<name<<endl;
			if (name.substr(0, 1) != "." && name != ".." && name.substr(name.size() - 3, name.size()) == "jpg")
				files.push_back(name);
		}
		closedir(dp);
		sort(files.begin(), files.end());
		cout << "Got the images." << endl;
		return true;
	}

	void read_camera_ex_param(string &path_camera_yaml, Mat  &R_CamLeft2CamRight, Mat &T_CamLeft2CamRight) {

		cv::FileStorage fs(path_camera_yaml, cv::FileStorage::READ);
		if (!fs.isOpened()) // failed
		{
			cout << "Open File Failed!" << endl;
		}
		else // succeed
		{
			cout << "succeed open ex_camera.yml" << endl;
			char buf1[100];
			sprintf_s(buf1, "R");
			char buf2[100];
			sprintf_s(buf2, "T");
			fs[buf1] >> R_CamLeft2CamRight;
			fs[buf2] >> T_CamLeft2CamRight;
			fs.release();
			cout << R_CamLeft2CamRight << endl;
			cout << T_CamLeft2CamRight << endl;
		}
	}



	void multimatch(string rawImagePath,Mat Limage, Mat Rimage, int num,
		vector<cv::Point2f> &pointsL, vector<cv::Point2f> &pointsR,
		vector < cv::Point3f > &objectPointsL, vector<cv::Point3f> &objectPointsR)
	{
		//Eigen::Matrix4f Ti2T0;
		//Ti2T0 << 0.9997, 0.0150, -0.0212, 3022.3921,
		//	-0.0152, 0.9999, -0.0065, -1.5706,
		//	0.0211, 0.0068, 0.9999, 6.3633,
		//	0, 0, 0, 1;
		//Eigen::Matrix4f Ti3T0;
		//Ti3T0 << -0.0274, 0.0029, 0.9996, 6010.4821,
		//	-0.0327, 0.9995, -0.00383, -224.7422,
		//	-0.9991, -0.0328, -0.0273, 6625.9614,
		//	0, 0, 0, 1;

		Eigen::Matrix4f Ti2T0;
		Ti2T0 << 0.9997, 0.0150, -0.0212, 1088.0612,
			-0.0152, 0.9999, -0.0065, -0.5654,
			0.0211, 0.0068, 0.9999, 2.2908,
			0, 0, 0, 1;
		Eigen::Matrix4f Ti3T0;
		Ti3T0 << -0.0274, 0.0029, 0.9996, 2163.7736,
			-0.0327, 0.9995, -0.00383, -80.9072,
			-0.9991, -0.0328, -0.0273, -2385.3461,
			0, 0, 0, 1;

		Eigen::Matrix4f rotate1, rotate3, rotate2;


		//第一步，绕z轴正向（逆时针，z轴指向纸面外）旋转90度
			//定义旋转矩阵
			//cos(x),sin(x),0
			//-sin(x),cos(x),0
			//	0,	0,	1,  

		rotate1 << cos(M_PI / 2), sin(M_PI / 2), 0, 0,
			-sin(M_PI / 2), cos(M_PI / 2), 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1;


		rotate2 << 1, 0, 0, 0,
			0, cos(M_PI), sin(M_PI), 0,
			0, -sin(M_PI), cos(M_PI), 0,
			0, 0, 0, 1;


		RobustMatcher rmatcher(cv::xfeatures2d::SIFT::create(1000));
		//mytools tools;
		//rawImagePath = "E:\\opencv_demo\\sift_PnP\\sift_PnP\\raw\\";
		string savematch = "E:\\opencv_demo\\sift_PnP\\sift_PnP\\match\\";
		vector<string> rawImageStr;

		if (!getImages(rawImageStr, rawImagePath))
		{
			cout << "Failed to get the  Limages!" << endl;
		}

		for (int i = 0; i < rawImageStr.size(); ++i)
		{
			string StrImg = rawImagePath + rawImageStr[i];
			Mat rawImage;
			rawImage = imread(StrImg);

			//cout << endl << endl;

			std::vector<cv::DMatch> matchesL, matchesR;
			std::vector<cv::KeyPoint> keypointsL, keypointsR, keypoints_raw_L, keypoints_raw_R;

			//cout << "第 " << num << "-" << i << " 张 L 图片匹配" << endl;

			cv::Mat fundamental_L = rmatcher.match(Limage, rawImage, matchesL,
				keypointsL, keypoints_raw_L);
			//cout << "matchesL 的大小是" << matchesL.size() << endl;
			Mat imageMatchesL, imageMatchesR;
			drawMatches(Limage, keypointsL, rawImage, keypoints_raw_L, matchesL, imageMatchesL);

			stringstream ssL;
			ssL << savematch << num << "-" << i << "-imageMatchesL.jpg";
			string savenameL = ssL.str();
			//imwrite(savenameL, imageMatchesL);

			//cout << "第 " << num << "-" << i << "张 R 图片匹配" << endl;
			cv::Mat fundamental_R = rmatcher.match(Rimage, rawImage, matchesR,
				keypointsR, keypoints_raw_R);

			drawMatches(Rimage, keypointsR, rawImage, keypoints_raw_R, matchesR, imageMatchesR);

			stringstream ssR;
			ssR << savematch << num << "-" << i << "-imageMatchesR.jpg";
			string savenameR = ssR.str();
			imwrite(savenameR, imageMatchesR);

			//cout << "matchesR1 的大小是" << matchesR.size() << endl;

			//把keypoints转为vector points
			//Get the position of left keypoints
			vector<cv::Point2f> points_RAW_L, points_RAW_R;

			//cout << "points_RAW_L 的大小是" << points_RAW_L.size() << endl;
			//cout << "points_RAW_R 的大小是" << points_RAW_R.size() << endl;


			for (std::vector<cv::DMatch>::const_iterator it = matchesL.begin();
				it != matchesL.end(); ++it)
			{
				pointsL.push_back(keypointsL[it->queryIdx].pt);
				points_RAW_L.push_back(keypoints_raw_L[it->trainIdx].pt);
			}

			//Get the position of right keypoints
			for (std::vector<cv::DMatch>::const_iterator it = matchesR.begin();
				it != matchesR.end(); ++it)
			{
				pointsR.push_back(keypointsR[it->queryIdx].pt);
				points_RAW_R.push_back(keypoints_raw_R[it->trainIdx].pt);
			}

			//cout << "pointsL 的大小是" << pointsL.size() << endl;
			//cout << "pointsR 的大小是" << pointsR.size() << endl;

			if (points_RAW_L.size() > 0 && i == 0)
			{
				for (int i = 0; i < points_RAW_L.size(); i++)
				{
					objectPointsL.push_back(cv::Point3f(points_RAW_L[i].y * 0.83, points_RAW_L[i].x * 0.83, 0));
				}
			}

			if (points_RAW_R.size() > 0 && i == 0)
			{
				for (int i = 0; i < points_RAW_R.size(); i++)
				{
					objectPointsR.push_back(cv::Point3f(points_RAW_R[i].y * 0.83, points_RAW_R[i].x * 0.83, 0));
				}
			}

			if (points_RAW_L.size() > 0 && i == 1)
			{
				for (int i = 0; i < points_RAW_L.size(); i++)
				{
					Eigen::Vector4f E_points2_2, E_points;
					E_points2_2 << points_RAW_L[i].x*0.83, points_RAW_L[i].y*0.83, 0, 1;
					E_points = rotate2 * rotate1*Ti2T0 * E_points2_2;

					objectPointsL.push_back(cv::Point3f(E_points(0), E_points(1), E_points(2)));
				}
			}
			if (points_RAW_R.size() > 0 && i == 1)
			{
				for (int i = 0; i < points_RAW_R.size(); i++)
				{
					Eigen::Vector4f E_points2, E_points;
					E_points2 << points_RAW_R[i].x*0.83, points_RAW_R[i].y*0.83, 0, 1;
					E_points = rotate2 * rotate1*Ti2T0 * E_points2;

					objectPointsR.push_back(cv::Point3f(E_points(0), E_points(1), E_points(2)));

				}

			}
			//cout << "points_RAW_L 的大小是" << points_RAW_L.size() << endl;
			if (points_RAW_L.size() > 0 && i == 2)
			{
				for (int i = 0; i < points_RAW_L.size(); i++)
				{
					Eigen::Vector4f E_points2_2, E_points;
					E_points2_2 << points_RAW_L[i].x*1.35, points_RAW_L[i].y*1.35, 0, 1;

					E_points = rotate2 * rotate1 * Ti3T0 * E_points2_2;
					objectPointsL.push_back(cv::Point3f(E_points(0), E_points(1), E_points(2)));
				}
			}
			//cout << "points_RAW_R 的大小是" << points_RAW_R.size() << endl;

			if (points_RAW_R.size() > 0 && i == 2)
			{
				for (int i = 0; i < points_RAW_R.size(); i++)
				{
					Eigen::Vector4f E_points2, E_points;
					E_points2 << points_RAW_R[i].x*1.35, points_RAW_R[i].y*1.35, 0, 1;
					E_points = rotate2 * rotate1 * Ti3T0 * E_points2;
					objectPointsR.push_back(cv::Point3f(E_points(0), E_points(1), E_points(2)));
				}
			}

			//cout << "objectPointsL 的大小是" << objectPointsL.size() << endl;
			//cout << "objectPointsR 的大小是" << objectPointsR.size() << endl;


			//for (int i = 0; i < pointsR.size(); i++)
			//{
			//	cout << "pointsR- " << i << "- is " << pointsR[i] << endl;
			//}

			//for (int i = 0; i < points_RAW_R.size(); i++)
			//{
			//	cout << "points_RAW_R- " << i << "- is " << points_RAW_R[i] << endl;
			//}

			//for (int i = 0; i < objectPointsR.size(); i++)
			//{
			//	cout << "objectPointsR- " << i << "- is " << objectPointsR[i] << endl;
			//}

			//cout << "输出测试" << endl;
		}

	}

	void CamPose0to1(Eigen::Quaterniond q0, Eigen::Quaterniond q1,
		/*相机的相对位姿*/		 Eigen::Vector3d t_vec0, Eigen::Vector3d t_vec1, Eigen::Matrix4d &T01)
	{
		Eigen::Quaterniond q01 = q0.inverse()*q1;
		q01.normalize();
		cout << "q01 is " << q01.coeffs() << endl;
		Eigen::Vector3d  t01 = q0.toRotationMatrix().inverse()*(t_vec1 - t_vec0);
		cout << "t01 is " << t01 << endl;

		Eigen::Matrix3d rotation_matrix;
		rotation_matrix = q01.toRotationMatrix();

		T01 << rotation_matrix(0), rotation_matrix(1), rotation_matrix(2), t01(0),
			rotation_matrix(3), rotation_matrix(4), rotation_matrix(5), t01(1),
			rotation_matrix(6), rotation_matrix(7), rotation_matrix(8), t01(2),
			0, 0, 0, 1;
	}

	void Cam_Pose(string rawImagePath,vector<string> LsImgs, vector<string> RsImgs, string leftpicture, string rightpicture,
		Mat camera0Matrix, Mat camera1Matrix, Mat camera0DistCoeffs, Mat camera1DistCoeffs, Mat transform_T,string poseSavePath)
	{
		Eigen::Matrix4d T0n;

		T0n << 1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1;

		Eigen::Matrix3d R_n_0;
		Eigen::Vector3d T_n_0;

		R_n_0 << 1, 0, 0, 0, 1, 0, 0, 0, 1;
		T_n_0 << 0, 0, 0;
		Eigen::Quaterniond q0 = Eigen::Quaterniond(R_n_0);
		cout << "q0 = " << q0.coeffs() << endl;
		Eigen::Matrix4d Tn;

		for (int i = 0; i < LsImgs.size(); ++i)
		{

			string LStrImg = leftpicture + LsImgs[i];
			string RStrImg = rightpicture + RsImgs[i];

			Mat Limage, Rimage;

			Limage = imread(LStrImg);
			Rimage = imread(RStrImg);

			//saveImage(Limage,,i);
			//namedWindow("Limage",CV_WINDOW_NORMAL);
			//imshow("Limage", Limage);
			//waitKey(500);

			std::vector<cv::Point2f> pointsL, pointsR, points3, points4;
			vector<Point3f>objectPointsL, objectPointsR;


			//cout << endl << endl;
			cout << "第-" << i << "-张图开始。。。。。。" << endl;


			multimatch(rawImagePath,Limage, Rimage, i, pointsL, pointsR, objectPointsL, objectPointsR);

			if (objectPointsL.size() > 15 || objectPointsR.size() > 15)
			{
				double cere_rot[3], cere_tranf[3];
				Mat rvec, tvec;
				if (objectPointsL.size() > 15 && objectPointsR.size() > 15) {

					solvePnPRansac(objectPointsL, pointsL,      // corresponding 3D/2D pts 
						camera0Matrix, camera0DistCoeffs,			  // calibration 
						rvec, tvec);

					cere_rot[0] = rvec.at<double>(0, 0);
					cere_rot[1] = rvec.at<double>(1, 0);
					cere_rot[2] = rvec.at<double>(2, 0);

					cere_tranf[0] = tvec.at<double>(0, 0);
					cere_tranf[1] = tvec.at<double>(1, 0);
					cere_tranf[2] = tvec.at<double>(2, 0);

					ceres::Problem problem;
					for (int i = 0; i < pointsL.size(); i++)
					{
						ceres::CostFunction* costfunction =
							new ceres::AutoDiffCostFunction<cost_function_define, 2, 3, 3>
							(new cost_function_define(objectPointsL[i], objectPointsR[i],
								pointsL[i], pointsR[i],
								camera0Matrix, camera1Matrix, camera0DistCoeffs, camera1DistCoeffs, transform_T));
						//(new cost_function_define(PointsOfWorld[i],idsCorner[i]));
						problem.AddResidualBlock(costfunction, NULL, cere_rot, cere_tranf);//注意，cere_rot不能为Mat类型 
					}
					ceres::Solver::Options options;
					options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
					options.minimizer_progress_to_stdout = true;
					ceres::Solver::Summary summary;
					ceres::Solve(options, &problem, &summary);
					//std::cout << summary.FullReport() << "\n";
				}
				else
				{
					if (objectPointsL.size() > 15) {

						solvePnPRansac(objectPointsL, pointsL,      // corresponding 3D/2D pts 
							camera0Matrix, camera0DistCoeffs,			  // calibration 
							rvec, tvec);

						cere_rot[0] = rvec.at<double>(0, 0);
						cere_rot[1] = rvec.at<double>(1, 0);
						cere_rot[2] = rvec.at<double>(2, 0);

						cere_tranf[0] = tvec.at<double>(0, 0);
						cere_tranf[1] = tvec.at<double>(1, 0);
						cere_tranf[2] = tvec.at<double>(2, 0);

						ceres::Problem problem;
						for (int i = 0; i < pointsL.size(); i++)
						{
							ceres::CostFunction* costfunction =
								new ceres::AutoDiffCostFunction<mono_cost_function_define, 2, 3, 3>
								(new mono_cost_function_define(objectPointsL[i],
									pointsL[i], camera0Matrix, camera0DistCoeffs));
							//(new cost_function_define(PointsOfWorld[i],idsCorner[i]));
							problem.AddResidualBlock(costfunction, NULL, cere_rot, cere_tranf);//注意，cere_rot不能为Mat类型 
						}
						ceres::Solver::Options options;
						options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
						options.minimizer_progress_to_stdout = true;
						ceres::Solver::Summary summary;
						ceres::Solve(options, &problem, &summary);
						//std::cout << summary.FullReport() << "\n";

					}
					else {

						if (objectPointsR.size() > 15) {

							solvePnPRansac(objectPointsR, pointsR,      // corresponding 3D/2D pts 
								camera1Matrix, camera1DistCoeffs,			  // calibration 
								rvec, tvec);

							//cout << "rvec is " << rvec << endl;
							cere_rot[0] = rvec.at<double>(0, 0);
							cere_rot[1] = rvec.at<double>(1, 0);
							cere_rot[2] = rvec.at<double>(2, 0);

							cere_tranf[0] = tvec.at<double>(0, 0);
							cere_tranf[1] = tvec.at<double>(1, 0);
							cere_tranf[2] = tvec.at<double>(2, 0);

							ceres::Problem problem;
							for (int i = 0; i < pointsR.size(); i++)
							{
								ceres::CostFunction* costfunction =
									new ceres::AutoDiffCostFunction<mono_cost_function_define, 2, 3, 3>
									(new mono_cost_function_define(objectPointsR[i],
										pointsR[i], camera1Matrix, camera1DistCoeffs));
								//(new cost_function_define(PointsOfWorld[i],idsCorner[i]));
								problem.AddResidualBlock(costfunction, NULL, cere_rot, cere_tranf);//注意，cere_rot不能为Mat类型 
							}
							ceres::Solver::Options options;
							options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
							options.minimizer_progress_to_stdout = true;
							ceres::Solver::Summary summary;
							ceres::Solve(options, &problem, &summary);
							//std::cout << summary.FullReport() << "\n";
						}
						else
						{
							continue;
						}
					}
				}
				Mat R_vec = (Mat_<double>(3, 1) << cere_rot[0], cere_rot[1], cere_rot[2]); //数组转cv向量
				Mat t_vec_L = (Mat_<double>(3, 1) << cere_tranf[0], cere_tranf[1], cere_tranf[2]);
				Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> t_n_1;

				Mat cv_Rmat;
				Rodrigues(R_vec, cv_Rmat);
				Eigen::Matrix3d E_Rmat;
				cv2eigen(cv_Rmat, E_Rmat);
				cv2eigen(t_vec_L, t_n_1);

				Eigen::Matrix4d T_1, T1;

				T_1 << E_Rmat(0), E_Rmat(1), E_Rmat(2), t_n_1(0),
					E_Rmat(3), E_Rmat(4), E_Rmat(5), t_n_1(1),
					E_Rmat(6), E_Rmat(7), E_Rmat(8), t_n_1(2),
					0, 0, 0, 1;

				cout << "T_1 is " << T_1 << endl;

				Eigen::Matrix4d rotate_Z;
				rotate_Z << cos(M_PI / 2), sin(M_PI / 2), 0, 0,
					-sin(M_PI / 2), cos(M_PI / 2), 0, 0,
					0, 0, 1, 0,
					0, 0, 0, 1;

				T1 = rotate_Z * T_1;

				cout << "T1 is " << T1 << endl;

				Eigen::Quaterniond q1 = Eigen::Quaterniond(T1.block<3, 3>(0, 0));
				q1.normalize();
				Eigen::Vector3d zt_n_1;

				zt_n_1 = T1.block<3, 1>(0, 3);


				cout << "zt_n_1 is " << zt_n_1 << endl;


				cout << "q1=\n" << q1.coeffs() << endl;//coeffs的顺序:(x,y,z,w)
				//q1.normalize();
				//cout << "qqqqqq=\n" << q1.coeffs()  << endl;//coeffs的顺序:(x,y,z,w)


				Eigen::Matrix4d T_01;

				CamPose0to1(q0, q1, T_n_0, zt_n_1, T_01);


				if (i == 0) {
					Eigen::Matrix3d q2r;

					q2r << T_01(0), T_01(1), T_01(2),
						T_01(3), T_01(4), T_01(5),
						T_01(6), T_01(7), T_01(8);

					cout << "q2r is " << q2r;

					q0 = Eigen::Quaterniond(q2r);
					q0.normalize();
					Eigen::Vector3d t12t0;
					t12t0 << T_01(3), T_01(7), T_01(11);

					T_n_0 = t12t0;

				}

				else
				{
					T_n_0 = zt_n_1;
					q0 = q1;
				}
				if (i == 2)
				{
					Tn = T_01;
				}
				else
				{

					if (T_01(3) < 200)
					{
						Tn = T_01 * Tn;
					}
					else
					{
						continue;
					}
				}


				cout << "Tn is " << Tn << endl;

				Mat cv_T01(4, 4, CV_32F);

				eigen2cv(Tn, cv_T01);

				cout << "cv_T01 is  " << cv_T01 << endl;

				double r1_1 = cv_T01.ptr<double>(0)[0];
				double r1_2 = cv_T01.ptr<double>(0)[1];
				double r1_3 = cv_T01.ptr<double>(0)[2];
				double r2_1 = cv_T01.ptr<double>(1)[0];
				double r2_2 = cv_T01.ptr<double>(1)[1];
				double r2_3 = cv_T01.ptr<double>(1)[2];
				double r3_1 = cv_T01.ptr<double>(2)[0];
				double r3_2 = cv_T01.ptr<double>(2)[1];
				double r3_3 = cv_T01.ptr<double>(2)[2];

				double t_x = cv_T01.ptr<double>(0)[3];
				double t_y = cv_T01.ptr<double>(1)[3];
				double t_z = cv_T01.ptr<double>(2)[3];

				//	vector<Point2f>  projectedPointsR, projectedPointsL;
				//	projectPoints(objectPointsL, R_vec, t_vec_L, camera0Matrix, camera0DistCoeffs, projectedPointsL);
				//	double err_x = 0, err_y = 0, error;
				//	for (int i = 0; i < projectedPointsL.size(); ++i)
				//	{
				//		err_x += abs(projectedPointsL[i].x - pointsL[i].x);
				//		err_y += abs(projectedPointsL[i].y - pointsL[i].y);
				//		error = sqrt(pow(projectedPointsL[i].x - pointsL[i].x, 2) + pow(projectedPointsL[i].y - pointsL[i].y, 2)) / 2;
				//	}
				//	double singleMeanError_x = 0;
				//	double singleMeanError_y = 0;

				//	singleMeanError_x += err_x / projectedPointsL.size();
				//	singleMeanError_y += err_y / projectedPointsL.size();

				//	cout << "图 " << i << " 平均重投影误差X= " << singleMeanError_x << endl;
				//	cout << "平均重投影误差Y= " << singleMeanError_y << endl;
				//	cout << "平均重投影误差 " << error << endl;

				//	cout << tvec;

				Mat rotation;

				Rodrigues(R_vec, rotation);

				//Eigen::Vector3f P_oc;

				//Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> R_n_1;
				//Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> T_n_1;

				//double r11 = rotation.ptr<double>(0)[0];
				//double r12 = rotation.ptr<double>(0)[1];
				//double r13 = rotation.ptr<double>(0)[2];
				//double r21 = rotation.ptr<double>(1)[0];
				//double r22 = rotation.ptr<double>(1)[1];
				//double r23 = rotation.ptr<double>(1)[2];
				//double r31 = rotation.ptr<double>(2)[0];
				//double r32 = rotation.ptr<double>(2)[1];
				//double r33 = rotation.ptr<double>(2)[2];

				//double thetaz = atan2(r21, r11) / CV_PI * 180;
				//double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33 * r33)) / CV_PI * 180;
				//double thetax = atan2(r32, r33) / CV_PI * 180;

				//double tx = tvec.ptr<double>(0)[0];
				//double ty = tvec.ptr<double>(0)[1];
				//double tz = tvec.ptr<double>(0)[2];

				//double x = tx, y = ty, z = tz;

				////进行三次反向旋转
				//codeRotateByZ(x, y, -1 * thetaz, x, y);
				//codeRotateByY(x, z, -1 * thetay, x, z);
				//codeRotateByX(y, z, -1 * thetax, y, z);

				//double Cx = x * -1;
				//double Cy = y * -1;
				//double Cz = z * -1;

				//cout << "左相机的世界坐标：" << Cx << ", " << Cy << ", " << Cz << endl;
				//cout << P_oc << endl;

				std::ofstream fout(poseSavePath + "pose.txt", std::ios::app);

				//fout << "image-" << i << " " << getSystemTime() << " " << Cx / 1000 << " " << Cy / 1000 << " " << Cz / 1000 << " " <<
				//	r11 << " " << r12 << " " << r13 << " " << r21 << " " << r22 << " " << r23 << " " << r31 << " " << r32 << " " << r33 << "\n";
				if (i > 3)
				{
					fout << "image-" << i << " " << getSystemTime() << " " << t_x / 1000 << " " << t_y / 1000 <<
						" " << t_z / 1000 << " " << r1_1 << " " << r1_2 << " " << r1_3 << " " << r2_1 << " "
						<< r2_2 << " " << r2_3 << " " << r3_1 << " " << r3_2 << " " << r3_3 << "\n";
				}
			}
		}
	}

};
#endif



//
////将空间点绕Z轴旋转
////输入参数 x y为空间点原始x y坐标
////thetaz为空间点绕Z轴旋转多少度，角度制范围在-180到180
////outx outy为旋转后的结果坐标
//template<typename T>
//inline void codeRotateByZ(T x, T y, T thetaz, T& outx, T& outy)
//{
//	T x1 = x;//将变量拷贝一次，保证&x == &outx这种情况下也能计算正确
//	T y1 = y;
//	T rz = thetaz * CV_PI / 180;
//	outx = cos(rz) * x1 - sin(rz) * y1;
//	outy = sin(rz) * x1 + cos(rz) * y1;
//}
//
////将空间点绕Y轴旋转
////输入参数 x z为空间点原始x z坐标
////thetay为空间点绕Y轴旋转多少度，角度制范围在-180到180
////outx outz为旋转后的结果坐标
//template<typename T>
//inline void codeRotateByY(T x, T z, T thetay, T& outx, T& outz)
//{
//	T x1 = x;
//	T z1 = z;
//	T ry = thetay * CV_PI / 180;
//	outx = cos(ry) * x1 + sin(ry) * z1;
//	outz = cos(ry) * z1 - sin(ry) * x1;
//}
//
////将空间点绕X轴旋转
////输入参数 y z为空间点原始y z坐标
////thetax为空间点绕X轴旋转多少度，角度制，范围在-180到180
////outy outz为旋转后的结果坐标
//template<typename T>
//inline void codeRotateByX(T y, T z, T thetax, T& outy, T& outz)
//{
//	T y1 = y;//将变量拷贝一次，保证&y == &y这种情况下也能计算正确
//	T z1 = z;
//	T rx = thetax * CV_PI / 180;
//	outy = cos(rx) * y1 - sin(rx) * z1;
//	outz = cos(rx) * z1 + sin(rx) * y1;
//}
//
//int64_t getSystemTime()//时间戳函数，返回值最好是int64_t，long long也可以
//{
//	struct timeb t;
//	ftime(&t);
//	return 1000 * t.time + t.millitm;
//}
//
//
////点乘
//template<typename T>
//inline T DotProduct(const T x[3], const T y[3]) {
//	return (x[0] * y[0] + x[1] * y[1] + x[2] * y[2]);
//}
////叉乘
//template<typename T>
//inline void CrossProduct(const T x[3], const T y[3], T result[3]) {
//	result[0] = x[1] * y[2] - x[2] * y[1];
//	result[1] = x[2] * y[0] - x[0] * y[2];
//	result[2] = x[0] * y[1] - x[1] * y[0];
//}
//
//template<typename T>
////矩阵相乘
//void matrix_Mul(const T src1[16], const T src2[16], T dst[16], int PointNum, int two, int two_B)
//{
//	T sub = T(0.0);
//	int i, j, k;
//	for (i = 0; i < PointNum; i++)
//	{
//		for (j = 0; j < two_B; j++)
//		{
//			sub = T(0.0);
//			for (k = 0; k < two; k++)
//			{
//				sub += src1[i*two + k] * src2[k*two_B + j];
//			}
//			dst[i*two_B + j] = sub;
//		}
//	}
//}
//
////旋转矩阵到向量
//template<typename T>
//void Rotationmatrix2AngleAxis(const T src[9], T dst[3])
//{
//	const T R_trace = src[0] + src[4] + src[8];
//	const T theta = acos((R_trace - T(1))*T(0.5));
//	T Right[9];
//	// 0 1 2   0 3 6
//	// 3 4 5   1 4 7
//	// 6 7 8   2 5 8
//	Right[1] = (src[1] - src[3])*T(0.5) / sin(theta);
//	Right[2] = (src[2] - src[6])*T(0.5) / sin(theta);
//	Right[3] = (src[3] - src[1])*T(0.5) / sin(theta);
//	Right[5] = (src[5] - src[7])*T(0.5) / sin(theta);
//	Right[6] = (src[6] - src[2])*T(0.5) / sin(theta);
//	Right[7] = (src[7] - src[5])*T(0.5) / sin(theta);
//
//	dst[0] = (Right[7] - Right[5])*T(0.5)*theta;
//	dst[1] = (Right[2] - Right[6])*T(0.5)*theta;
//	dst[2] = (Right[3] - Right[1])*T(0.5)*theta;
//
//}
//
////向量到矩阵
//template<typename T>
//void AngleAxis2Rotationmatrix(const T src[3], T dst[9])
//{
//	const T theta2 = DotProduct(src, src);
//
//	if (theta2 > T(std::numeric_limits<double>::epsilon())) {
//
//		const T theta = sqrt(theta2);
//		const T c = cos(theta);
//		const T c1 = 1. - c;
//		const T s = sin(theta);
//		const T theta_inverse = 1.0 / theta;
//
//		const T w[3] = { src[0] * theta_inverse,
//			src[1] * theta_inverse,
//			src[2] * theta_inverse };
//
//		dst[0] = c * T(1) + c1 * w[0] * w[0] + s * T(0);
//		dst[1] = c * T(0) + c1 * w[0] * w[1] + s * T(-w[2]);
//		dst[2] = c * T(0) + c1 * w[0] * w[2] + s * T(w[1]);
//		dst[3] = c * T(0) + c1 * w[0] * w[1] + s * T(w[2]);
//		dst[4] = c * T(1) + c1 * w[1] * w[1] + s * T(0);
//		dst[5] = c * T(0) + c1 * w[1] * w[2] + s * -w[0];
//		dst[6] = c * T(0) + c1 * w[0] * w[2] + s * -w[1];
//		dst[7] = c * T(0) + c1 * w[1] * w[2] + s * w[0];
//		dst[8] = c * T(1) + c1 * w[2] * w[2] + s * T(0);
//	}
//	else {//角度非常微小的情况
//		dst[0] = T(1);
//		dst[1] = T(0);
//		dst[2] = T(0);
//		dst[3] = T(0);
//		dst[4] = T(1);
//		dst[5] = T(0);
//		dst[6] = T(0);
//		dst[7] = T(0);
//		dst[8] = T(1);
//	}
//}
//
//template<typename T>
//void AngleAxisRotatePoint(const T angle_axis[3], const T pt[3], T result[3])
//{
//	const T theta2 = DotProduct(angle_axis, angle_axis);//angle_axis[3]为旋转向量，DotProduct()点乘求解模长的平方，计算角度大小，
//
//	if (theta2 > T(std::numeric_limits<double>::epsilon())) {//确保 theta2 跟0比足够大，否则开方后还是0，无法取倒数
//
//		const T theta = sqrt(theta2);
//		const T costheta = cos(theta);
//		const T sintheta = sin(theta);
//		const T theta_inverse = 1.0 / theta;
//
//		const T w[3] = { angle_axis[0] * theta_inverse,
//			angle_axis[1] * theta_inverse,
//			angle_axis[2] * theta_inverse };
//
//		T w_cross_pt[3];
//		CrossProduct(w, pt, w_cross_pt);//叉乘
//
//		const T tmp = DotProduct(w, pt) * (T(1.0) - costheta);
//
//		result[0] = pt[0] * costheta + w_cross_pt[0] * sintheta + w[0] * tmp;
//		result[1] = pt[1] * costheta + w_cross_pt[1] * sintheta + w[1] * tmp;
//		result[2] = pt[2] * costheta + w_cross_pt[2] * sintheta + w[2] * tmp;
//	}
//
//	else {//角度非常微小的情况，与0接近,利用一阶泰勒近似
//
//		T w_cross_pt[3];
//		CrossProduct(angle_axis, pt, w_cross_pt);
//
//		result[0] = pt[0] + w_cross_pt[0];
//		result[1] = pt[1] + w_cross_pt[1];
//		result[2] = pt[2] + w_cross_pt[2];
//	}
//}
//
///******************保存图像*****************/
//void saveImage(cv::Mat image, string &outPutPath, int index)// 保存spinview转换mat后的数据
//{
//	//定义保存图像的名字
//	string strSaveName;
//	char buffer[256];
//	sprintf_s(buffer, "D%04d", index);
//	strSaveName = buffer;
//
//	//定义保存图像的完整路径
//	string strImgSavePath = outPutPath + "\\" + strSaveName;
//	//定义保存图像的格式
//	strImgSavePath += ".jpg";
//	//strImgSavePath += ".png
//	ostringstream imgname;
//	//保存操作
//	imwrite(strImgSavePath.c_str(), image);
//}
//
///***************** Mat转vector **********************/
//template<typename _Tp>
//vector<_Tp> convertMat2Vector(const Mat &mat, int m, int n)
//{
//	return (vector<_Tp>)(mat.reshape(m, n));//通道数不变，按行转为一行 m=1,n=1
//}
//
//template<typename _Tp>
//cv::Mat convertVector2Mat(vector<_Tp> v, int channels, int rows)
//{
//	cv::Mat mat = cv::Mat(v);//将vector变成单列的mat
//	cv::Mat dest = mat.reshape(channels, rows).clone();//PS：必须clone()一份，否则返回出错
//	return dest;
//}
//
//
///************检测元素是否在向量Vector中****************/
//bool is_element_in_vector(vector<int> v, int element)
//{
//	vector<int>::iterator it;
//	it = find(v.begin(), v.end(), element);
//	if (it != v.end()) {
//		return true;
//	}
//	else {
//		return false;
//	}
//}
//
//bool getImages(vector<string>& files, string &path)
//{
//
//	DIR *dp;
//	struct dirent *dirp;
//	if ((dp = opendir(path.c_str())) == NULL) {
//		cout << "failed to get the images!" << endl;
//		return false;;
//	}
//	while ((dirp = readdir(dp)) != NULL) {
//		string name = string(dirp->d_name);
//		//cout<<name<<endl;
//		if (name.substr(0, 1) != "." && name != ".." && name.substr(name.size() - 3, name.size()) == "jpg")
//			files.push_back(name);
//	}
//	closedir(dp);
//	sort(files.begin(), files.end());
//	cout << "Got the images." << endl;
//	return true;
//}
//
//void read_camera_param(string &path_camera_yaml, Mat &cameraMatrix, Mat &cameraDistCoeffs) {
//
//	cv::FileStorage fs(path_camera_yaml, cv::FileStorage::READ);
//	if (!fs.isOpened()) // failed
//	{
//		cout << "Open File Failed!" << endl;
//	}
//
//	else // succeed
//	{
//		cout << "succeed open camera.yml" << endl;
//		char buf1[100];
//		sprintf_s(buf1, "camera_matrix");
//		char buf2[100];
//		sprintf_s(buf2, "distortion_coefficients");
//		fs[buf1] >> cameraMatrix;
//		fs[buf2] >> cameraDistCoeffs;
//		fs.release();
//		cout << cameraMatrix << endl;
//		cout << cameraDistCoeffs << endl;
//	}
//}
//
//void read_camera_ex_param(string &path_camera_yaml, Mat  &R_CamLeft2CamRight, Mat &T_CamLeft2CamRight) {
//
//	cv::FileStorage fs(path_camera_yaml, cv::FileStorage::READ);
//	if (!fs.isOpened()) // failed
//	{
//		cout << "Open File Failed!" << endl;
//	}
//	else // succeed
//	{
//		cout << "succeed open ex_camera.yml" << endl;
//		char buf1[100];
//		sprintf_s(buf1, "R");
//		char buf2[100];
//		sprintf_s(buf2, "T");
//		fs[buf1] >> R_CamLeft2CamRight;
//		fs[buf2] >> T_CamLeft2CamRight;
//		fs.release();
//		cout << R_CamLeft2CamRight << endl;
//		cout << T_CamLeft2CamRight << endl;
//	}
//}
//
//
//struct cost_function_define
//{
//	cost_function_define(Point3f P0, Point3f P1, Point2f uv0, Point2f uv1, Mat KL, Mat KR, Mat DL, Mat DR, Mat T01)
//		:_P0(P0), _P1(P1), _uv0(uv0), _uv1(uv1), _KL(KL), _KR(KR), _DL(DL), _DR(DR), _T01(T01) {}
//	template<typename T>
//	bool operator()(const T*  const cere_r, const T* const cere_t, T* residual)const
//	{
//		//第一步，根据 左相机位姿 和 外参 得到右相机
//
//		//利用手写的罗德里格斯公式将旋转向量 cere_r 转成矩阵 Rc0
//		T Rc0[9];
//		AngleAxis2Rotationmatrix(cere_r, Rc0);
//
//		T TL[16];
//
//		//把左相机位姿拼接成4*4矩阵，一维数组表示；
//		TL[0] = Rc0[0]; TL[1] = Rc0[1]; TL[2] = Rc0[2]; TL[3] = cere_t[0];
//		TL[4] = Rc0[3]; TL[5] = Rc0[4]; TL[6] = Rc0[5]; TL[7] = cere_t[1];
//		TL[8] = Rc0[6]; TL[9] = Rc0[7]; TL[10] = Rc0[8]; TL[11] = cere_t[2];
//		TL[12] = T(0);  TL[13] = T(0);  TL[14] = T(0);   TL[15] = T(1);
//
//		//准备外参矩阵
//		T T0_1[16];
//		T0_1[0] = T(_T01.at<double>(0, 0));
//		T0_1[1] = T(_T01.at<double>(0, 1));
//		T0_1[2] = T(_T01.at<double>(0, 2));
//		T0_1[3] = T(_T01.at<double>(0, 3));
//
//		T0_1[4] = T(_T01.at<double>(1, 0));
//		T0_1[5] = T(_T01.at<double>(1, 1));
//		T0_1[6] = T(_T01.at<double>(1, 2));
//		T0_1[7] = T(_T01.at<double>(1, 3));
//
//		T0_1[8] = T(_T01.at<double>(2, 0));
//		T0_1[9] = T(_T01.at<double>(2, 1));
//		T0_1[10] = T(_T01.at<double>(2, 2));
//		T0_1[11] = T(_T01.at<double>(2, 3));
//
//		T0_1[12] = T(_T01.at<double>(3, 0));
//		T0_1[13] = T(_T01.at<double>(3, 1));
//		T0_1[14] = T(_T01.at<double>(3, 2));
//		T0_1[15] = T(_T01.at<double>(3, 3));
//
//		//外参矩阵与左相机位姿矩阵相乘得到右相机位姿矩阵
//		T TR[16];
//		matrix_Mul(T0_1, TL, TR, 4, 4, 4);
//
//		//右相机旋转矩阵，旋转向量，位移
//		T Rc1[9]; T Rc1_v[3]; T tc1[3];
//		Rc1[0] = TR[0]; Rc1[1] = TR[1]; Rc1[2] = TR[2]; tc1[0] = TR[3];
//		Rc1[3] = TR[4]; Rc1[4] = TR[5]; Rc1[5] = TR[6]; tc1[1] = TR[7];
//		Rc1[6] = TR[8]; Rc1[7] = TR[9]; Rc1[8] = TR[10]; tc1[2] = TR[11];
//
//		//右相机旋转矩阵 转为 旋转向量
//		Rotationmatrix2AngleAxis(Rc1, Rc1_v);
//
//		//第二步 准备投影,做残差
//		//左右相机三d点
//		T p0_1[3], p1_1[3];//points of world
//		T p0_2[3], p1_2[3];//point in camera  coordinate system 
//
//		p0_1[0] = T(_P0.x);
//		p0_1[1] = T(_P0.y);
//		p0_1[2] = T(_P0.z);
//
//		p1_1[0] = T(_P1.x);
//		p1_1[1] = T(_P1.y);
//		p1_1[2] = T(_P1.z);
//
//		//cout << "point_3d: " << p_1[0] << " " << p_1[1] << "  " << p_1[2] << endl;
//		// 将世界坐标系中的特征点转换到相机坐标系中
//		AngleAxisRotatePoint(cere_r, p0_1, p0_2);
//		AngleAxisRotatePoint(Rc1_v, p1_1, p1_2);
//
//		p0_2[0] = p0_2[0] + cere_t[0];
//		p0_2[1] = p0_2[1] + cere_t[1];
//		p0_2[2] = p0_2[2] + cere_t[2];
//
//		p1_2[0] = p1_2[0] + tc1[0];
//		p1_2[1] = p1_2[1] + tc1[1];
//		p1_2[2] = p1_2[2] + tc1[2];
//
//		const T x0 = p0_2[0] / p0_2[2];
//		const T y0 = p0_2[1] / p0_2[2];
//
//		const T x1 = p1_2[0] / p1_2[2];
//		const T y1 = p1_2[1] / p1_2[2];
//
//		T DL1 = T(_DL.at<double>(0, 0));
//		T Dl2 = T(_DL.at<double>(0, 1));
//
//		T DR1 = T(_DR.at<double>(0, 0));
//		T DR2 = T(_DR.at<double>(0, 1));
//
//		T r2_L = x0 * x0 + y0 * y0;
//		T r2_R = x1 * x1 + y1 * y1;
//
//		T distortion_L = T(1.0) + r2_L * (DL1 + Dl2 * r2_L);
//		T distortion_R = T(1.0) + r2_R * (DR1 + DR2 * r2_R);
//
//		//三维点重投影计算的像素坐标
//		const T u0 = x0 * distortion_L*_KL.at<double>(0, 0) + _KL.at<double>(0, 2);
//		const T v0 = y0 * distortion_L*_KL.at<double>(1, 1) + _KL.at<double>(1, 2);
//
//		const T u1 = x1 * distortion_L*_KR.at<double>(0, 0) + _KR.at<double>(0, 2);
//		const T v1 = y1 * distortion_L*_KR.at<double>(1, 1) + _KR.at<double>(1, 2);
//
//		//观测的在图像坐标下的值
//		T uv0_u = T(_uv0.x);
//		T uv0_v = T(_uv0.y);
//
//		T uv1_u = T(_uv1.x);
//		T uv1_v = T(_uv1.y);
//
//		residual[0] = uv0_u - u0 + uv1_u - u1;
//		residual[1] = uv0_v - v0 + uv1_v - v1;
//
//		return true;
//	}
//	Point3f _P0, _P1;
//	Point2f _uv0, _uv1;
//	Mat _KL, _KR, _DL, _DR;
//	Mat _T01;
//};


vrslam vrSlamFunction;

__declspec(dllexport) int SiftPnP(string rawImagePath,string leftParam, string rightParam, string exParam,
string leftpicture, string rightpicture, string poseSavePath) {



	Mat camera0Matrix, camera0DistCoeffs;
	Mat camera1Matrix, camera1DistCoeffs;
	Mat R_CamLeft2CamRight, T_CamLeft2CamRight, transform_T;

	vrSlamFunction.read_camera_param(leftParam, camera0Matrix, camera0DistCoeffs);
	vrSlamFunction.read_camera_param(rightParam, camera1Matrix, camera1DistCoeffs);
	vrSlamFunction.read_camera_ex_param(exParam, R_CamLeft2CamRight, T_CamLeft2CamRight);

	Mat homo = (Mat_<double>(1, 4) << 0, 0, 0, 1);
	hconcat(R_CamLeft2CamRight, T_CamLeft2CamRight, transform_T);//矩阵合并，行数不变
	vconcat(transform_T, homo, transform_T);//矩阵合并，列数不变

	vector<string> LsImgs;
	vector<string> RsImgs;

	double singleMeanError_x = 0;
	double singleMeanError_y = 0;

	if (!vrSlamFunction.getImages(LsImgs, leftpicture))
	{
		cout << "Failed to get the  Limages!" << endl;
		return -1;
	}

	if (!vrSlamFunction.getImages(RsImgs, rightpicture))
	{
		cout << "Failed to get the Rimages!" << endl;
		return -1;
	}

	//Mat rawimg1 = imread("1.jpg", CV_WINDOW_AUTOSIZE);
	//Mat rawimg2 = imread("2.jpg", CV_WINDOW_AUTOSIZE);
	//Mat rawimg3 = imread("3.jpg", CV_WINDOW_AUTOSIZE);

	vrSlamFunction.Cam_Pose(rawImagePath,LsImgs, RsImgs, leftpicture, rightpicture,
		camera0Matrix, camera1Matrix, camera0DistCoeffs, camera1DistCoeffs, transform_T, poseSavePath);

	return 0;
}

__declspec(dllexport) int ChAruCoPnP(string leftParam, string rightParam, string exParam,
	string leftpicture, string rightpicture, string poseSavePath,
	int lenth_num, int width_num, double squareLength, double markerLength)
{
	// 从yml文件中读取相机参数 
	Mat camera0Matrix, camera0DistCoeffs;
	Mat camera1Matrix, camera1DistCoeffs;
	Mat R_CamLeft2CamRight, T_CamLeft2CamRight, transform_T;


	vrSlamFunction.read_camera_param(leftParam, camera0Matrix, camera0DistCoeffs);
	vrSlamFunction.read_camera_param(rightParam, camera1Matrix, camera1DistCoeffs);
	vrSlamFunction.read_camera_ex_param(exParam, R_CamLeft2CamRight, T_CamLeft2CamRight);

	Mat homo = (Mat_<double>(1, 4) << 0, 0, 0, 1);
	hconcat(R_CamLeft2CamRight, T_CamLeft2CamRight, transform_T);//矩阵合并，行数不变
	vconcat(transform_T, homo, transform_T);//矩阵合并，列数不变

	int dictionaryId = 2; // 二维码字典的编号。1表示有100种4*4格子的二维码

	Ptr<aruco::DetectorParameters> detectorParams = aruco::DetectorParameters::create();
	Ptr<aruco::DetectorParameters> detectorParams1 = aruco::DetectorParameters::create();

	Ptr<aruco::Dictionary> dictionary =
		aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));
	Ptr<aruco::Dictionary> dictionary1 =
		aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));

	Ptr<aruco::CharucoBoard> charucoboard =
		aruco::CharucoBoard::create(lenth_num, width_num, squareLength, markerLength, dictionary);
	Ptr<aruco::CharucoBoard> charucoboard1 =
		aruco::CharucoBoard::create(lenth_num, width_num, squareLength, markerLength, dictionary1);

	Ptr<aruco::Board> board = charucoboard.staticCast<aruco::Board>();
	Ptr<aruco::Board> board1 = charucoboard1.staticCast<aruco::Board>();

	vector<string> LsImgs;
	vector<string> RsImgs;

	double singleMeanError_x = 0;
	double singleMeanError_y = 0;

	if (!vrSlamFunction.getImages(LsImgs, leftpicture))
	{
		cout << "Failed to get the images!" << endl;
		return -1;
	}

	if (!vrSlamFunction.getImages(RsImgs, rightpicture))
	{
		cout << "Failed to get the images!" << endl;
		return -1;
	}

	Eigen::Matrix4d T0n;

	T0n << 1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1;

	Eigen::Matrix3d R_n_0;
	Eigen::Vector3d T_n_0;

	R_n_0 << 1, 0, 0, 0, 1, 0, 0, 0, 1;
	T_n_0 << 0, 0, 0;
	Eigen::Quaterniond q0 = Eigen::Quaterniond(R_n_0);
	cout << "q0 = " << q0.coeffs() << endl;
	Eigen::Matrix4d Tn;

	for (int i = 0; i < LsImgs.size(); ++i)
	{
		string LStrImg = leftpicture + LsImgs[i];
		string RStrImg = rightpicture + RsImgs[i];

		Mat Limage, Rimage;

		Limage = imread(LStrImg);
		Rimage = imread(RStrImg);

		vector<Point2d> idsCorner, idsCorner1;//存储每张图片的角点
		vector<int> cornerIds, cornerIds1;		  //角点的ID
		vector<Point3f> PointsOfWorld, PointsOfWorld1; //角点对应的3D点
		vector<Point2f> imagePoints, imagePoints1;//角点在图像的像素点坐标
		vector< int > ids, ids1;
		vector< vector< Point2f > > corners, corners1, rejected, rejected1;
		// detect markers
		aruco::detectMarkers(Limage, dictionary, corners, ids, detectorParams, rejected);
		aruco::detectMarkers(Rimage, dictionary1, corners1, ids1, detectorParams1, rejected1);

		// refind strategy to detect more markers
		aruco::refineDetectedMarkers(Limage, board, corners, ids, rejected);
		aruco::refineDetectedMarkers(Rimage, board1, corners1, ids1, rejected1);

		// interpolate charuco corners
		Mat currentCharucoCorners, currentCharucoIds, currentCharucoCorners1, currentCharucoIds1;
		if (ids.size() > 0)
			aruco::interpolateCornersCharuco(corners, ids, Limage,
				charucoboard, currentCharucoCorners, currentCharucoIds);

		if (ids1.size() > 0)
			aruco::interpolateCornersCharuco(corners1, ids1, Rimage,
				charucoboard1, currentCharucoCorners1, currentCharucoIds1);

		if (currentCharucoIds.cols != 0)//过滤掉没检查到Charuco的情况！！没检查到Charuco，currentCharucoIds是空的，无法进行mat2Vector转换。
		{
			cornerIds = vrSlamFunction.convertMat2Vector<int>(currentCharucoIds, 1, 1);
			//cout << "cornerIds.size" << cornerIds.size() << endl;
		}

		if (currentCharucoIds1.cols != 0)//过滤掉没检查到Charuco的情况！！没检查到Charuco，currentCharucoIds是空的，无法进行mat2Vector转换。
		{
			cornerIds1 = vrSlamFunction.convertMat2Vector<int>(currentCharucoIds1, 1, 1);
			//cout << "cornerIds.size" << cornerIds1.size() << endl;
		}

		for (int i = 0; i < currentCharucoCorners.rows; i++)
		{
			cv::Point2d p;
			p.x = currentCharucoCorners.at<float>(i, 0);
			p.y = currentCharucoCorners.at<float>(i, 1);
			idsCorner.push_back(p);
		}

		for (int i = 0; i < currentCharucoCorners1.rows; i++)
		{
			cv::Point2d p;
			p.x = currentCharucoCorners1.at<float>(i, 0);
			p.y = currentCharucoCorners1.at<float>(i, 1);
			idsCorner1.push_back(p);
		}

		//3D坐标点生成 
		vector<Point3f>objectPoints;
		for (int x = 0; x <= 9; x++)
		{
			for (int y = 13; y >= 0; y--)
			{
				objectPoints.push_back(cv::Point3f((float)x, (float)y, 0) * squareLength);
			}
		}

		//对齐2d-3d点，思路：cornerIds中的点是少的，objectPoints中的点是预设的最全的点，检查cornerIds的序号是否在objectPoints中，按照序号从objectPoints查找

		for (int j = 0; j < 126; ++j)
		{
			if (vrSlamFunction.is_element_in_vector(cornerIds, j))
			{
				PointsOfWorld.push_back(objectPoints[j]);
			}
		}

		for (int j = 0; j < 126; ++j)
		{
			if (vrSlamFunction.is_element_in_vector(cornerIds1, j))
			{
				PointsOfWorld1.push_back(objectPoints[j]);
			}
		}

		//计算PnP
		Mat homogeneous;

		Mat rvec, tvec, rvec1, tvec1, chrvec, chtvec;
		Mat rotation, rotation1;
		//vector<Point2f> projectedPointsL, projectedPointsR;

		//Left camera
		if (ids.size() > 15 && ids1.size() > 15)
		{
			solvePnP(PointsOfWorld, idsCorner,      // corresponding 3D/2D pts 
				camera0Matrix, camera0DistCoeffs,			  // calibration 
				rvec, tvec);

			//Mat pose_result;
			//Rodrigues(rvec, rotation);
			//hconcat(rotation, tvec, pose_result);     //合并r，t,行数不变
			//vconcat(pose_result, homo, pose_result);  //矩阵合并，列数不变

			double cere_rot[3], cere_tranf[3];

			cere_rot[0] = rvec.at<double>(0, 0);
			cere_rot[1] = rvec.at<double>(1, 0);
			cere_rot[2] = rvec.at<double>(2, 0);

			cere_tranf[0] = tvec.at<double>(0, 0);
			cere_tranf[1] = tvec.at<double>(1, 0);
			cere_tranf[2] = tvec.at<double>(2, 0);

			Mat KL = camera0Matrix;
			Mat KR = camera1Matrix;

			ceres::Problem problem;
			for (int i = 0; i < idsCorner.size(); i++)
			{

				if (currentCharucoIds.cols != 0 && currentCharucoIds1.cols != 0)
				{
					ceres::CostFunction* costfunction =
						new ceres::AutoDiffCostFunction<cost_function_define, 2, 3, 3>
						(new cost_function_define(PointsOfWorld[i], PointsOfWorld1[i],
							idsCorner[i], idsCorner1[i],
							KL, KR, camera0DistCoeffs, camera1DistCoeffs, transform_T));
					//(new cost_function_define(PointsOfWorld[i],idsCorner[i]));
					problem.AddResidualBlock(costfunction, NULL, cere_rot, cere_tranf);//注意，cere_rot不能为Mat类型 
				}
			}
			ceres::Solver::Options options;
			options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
			options.minimizer_progress_to_stdout = true;
			ceres::Solver::Summary summary;
			ceres::Solve(options, &problem, &summary);
			std::cout << summary.FullReport() << "\n";

			Mat R_vec = (Mat_<double>(3, 1) << cere_rot[0], cere_rot[1], cere_rot[2]); //数组转cv向量
			Mat t_vec_L = (Mat_<double>(3, 1) << cere_tranf[0], cere_tranf[1], cere_tranf[2]);

			Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> t_n_1;

			Mat cv_Rmat;
			Rodrigues(R_vec, cv_Rmat);
			Eigen::Matrix3d E_Rmat;
			cv2eigen(cv_Rmat, E_Rmat);
			cv2eigen(t_vec_L, t_n_1);

			Eigen::Matrix4d T_1, T1;

			T_1 << E_Rmat(0), E_Rmat(1), E_Rmat(2), t_n_1(0),
				E_Rmat(3), E_Rmat(4), E_Rmat(5), t_n_1(1),
				E_Rmat(6), E_Rmat(7), E_Rmat(8), t_n_1(2),
				0, 0, 0, 1;

			cout << "T_1 is " << T_1 << endl;

			Eigen::Matrix4d rotate_Z;
			rotate_Z << cos(M_PI / 2), sin(M_PI / 2), 0, 0,
				-sin(M_PI / 2), cos(M_PI / 2), 0, 0,
				0, 0, 1, 0,
				0, 0, 0, 1;

			T1 = rotate_Z * T_1;

			cout << "T1 is " << T1 << endl;

			Eigen::Quaterniond q1 = Eigen::Quaterniond(T1.block<3, 3>(0, 0));
			q1.normalize();
			Eigen::Vector3d zt_n_1;

			zt_n_1 = T1.block<3, 1>(0, 3);


			cout << "zt_n_1 is " << zt_n_1 << endl;


			cout << "q1=\n" << q1.coeffs() << endl;//coeffs的顺序:(x,y,z,w)
			//q1.normalize();
			//cout << "qqqqqq=\n" << q1.coeffs()  << endl;//coeffs的顺序:(x,y,z,w)


			Eigen::Matrix4d T_01;

			vrSlamFunction.CamPose0to1(q0, q1, T_n_0, zt_n_1, T_01);


			if (i == 0) {
				Eigen::Matrix3d q2r;

				q2r << T_01(0), T_01(1), T_01(2),
					T_01(3), T_01(4), T_01(5),
					T_01(6), T_01(7), T_01(8);

				cout << "q2r is " << q2r;

				q0 = Eigen::Quaterniond(q2r);
				q0.normalize();
				Eigen::Vector3d t12t0;
				t12t0 << T_01(3), T_01(7), T_01(11);

				T_n_0 = t12t0;

			}

			else
			{
				T_n_0 = zt_n_1;
				q0 = q1;
			}
			if (i == 2)
			{
				Tn = T_01;
			}
			else
			{

				if (T_01(3) < 300)
				{
					Tn = T_01 * Tn;
				}
				else
				{
					continue;
				}
			}


			cout << "Tn is " << Tn << endl;

			Mat cv_T01(4, 4, CV_32F);

			eigen2cv(Tn, cv_T01);

			cout << "cv_T01 is  " << cv_T01 << endl;

			double r1_1 = cv_T01.ptr<double>(0)[0];
			double r1_2 = cv_T01.ptr<double>(0)[1];
			double r1_3 = cv_T01.ptr<double>(0)[2];
			double r2_1 = cv_T01.ptr<double>(1)[0];
			double r2_2 = cv_T01.ptr<double>(1)[1];
			double r2_3 = cv_T01.ptr<double>(1)[2];
			double r3_1 = cv_T01.ptr<double>(2)[0];
			double r3_2 = cv_T01.ptr<double>(2)[1];
			double r3_3 = cv_T01.ptr<double>(2)[2];

			double t_x = cv_T01.ptr<double>(0)[3];
			double t_y = cv_T01.ptr<double>(1)[3];
			double t_z = cv_T01.ptr<double>(2)[3];

			//Rodrigues(R_vec, rotation);

			//double r11 = rotation.ptr<double>(0)[0];
			//double r12 = rotation.ptr<double>(0)[1];
			//double r13 = rotation.ptr<double>(0)[2];
			//double r21 = rotation.ptr<double>(1)[0];
			//double r22 = rotation.ptr<double>(1)[1];
			//double r23 = rotation.ptr<double>(1)[2];
			//double r31 = rotation.ptr<double>(2)[0];
			//double r32 = rotation.ptr<double>(2)[1];
			//double r33 = rotation.ptr<double>(2)[2];

			//double thetaz = atan2(r21, r11) / CV_PI * 180;
			//double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33 * r33)) / CV_PI * 180;
			//double thetax = atan2(r32, r33) / CV_PI * 180;

			//double tx = t_vec_L.ptr<double>(0)[0];
			//double ty = t_vec_L.ptr<double>(0)[1];
			//double tz = t_vec_L.ptr<double>(0)[2];

			//double x = tx, y = ty, z = tz;

			////进行三次反向旋转
			//codeRotateByZ(x, y, -1 * thetaz, x, y);
			//codeRotateByY(x, z, -1 * thetay, x, z);
			//codeRotateByX(y, z, -1 * thetax, y, z);

			//double Cx = x * -1;
			//double Cy = y * -1;
			//double Cz = z * -1;

			std::ofstream fout(poseSavePath + "pose.txt", std::ios::app);
			if (i > 3)
			{
				fout << "image-" << i << " " << vrSlamFunction.getSystemTime() << " " << t_x / 1000 << " " << t_y / 1000 <<
					" " << t_z / 1000 << " " << r1_1 << " " << r1_2 << " " << r1_3 << " " << r2_1 << " "
					<< r2_2 << " " << r2_3 << " " << r3_1 << " " << r3_2 << " " << r3_3 << "\n";
			}
		}
	}
	return 0;
}

//
//
//
//void saveImage(cv::Mat image, string &outPutPath, int index)// 保存spinview转换mat后的数据
//{
//	//定义保存图像的名字
//	string strSaveName;
//	char buffer[256];
//	sprintf_s(buffer, "D%04d", index);
//	strSaveName = buffer;
//
//	//定义保存图像的完整路径
//	string strImgSavePath = outPutPath + "\\" + strSaveName;
//	//定义保存图像的格式
//	strImgSavePath += ".jpg";
//	//strImgSavePath += ".png
//
//	ostringstream imgname;
//
//	//保存操作
//
//	imwrite(strImgSavePath.c_str(), image);
//}



static bool saveCameraParams(const string &filename, Size imageSize, float aspectRatio, int flags,
	const Mat &cameraMatrix, const Mat &distCoeffs, double totalAvgErr) {
	FileStorage fs(filename, FileStorage::WRITE);
	if (!fs.isOpened())
		return false;

	/*time_t tt;
	time(&tt);
	struct tm *t2 = localtime(&tt);

	strftime(buf, sizeof(buf) - 1, "%c", t2);

	fs << "calibration_time" << buf;*/
	char buf[1024];
	fs << "image_width" << imageSize.width;
	fs << "image_height" << imageSize.height;

	if (flags & CALIB_FIX_ASPECT_RATIO) fs << "aspectRatio" << aspectRatio;

	if (flags != 0) {
		sprintf_s(buf, "flags: %s%s%s%s",
			flags & CALIB_USE_INTRINSIC_GUESS ? "+use_intrinsic_guess" : "",
			flags & CALIB_FIX_ASPECT_RATIO ? "+fix_aspectRatio" : "",
			flags & CALIB_FIX_PRINCIPAL_POINT ? "+fix_principal_point" : "",
			flags & CALIB_ZERO_TANGENT_DIST ? "+zero_tangent_dist" : "");
	}

	fs << "flags" << flags;

	fs << "camera_matrix" << cameraMatrix;
	fs << "distortion_coefficients" << distCoeffs;

	fs << "avg_reprojection_error" << totalAvgErr;

	return true;
}

// stereoCalibrate1(string leftimage, string rightimage, string cameraLParam, string cameraRParam, string intrinsic_filename = "intrinsics.yml", string extrinsic_filename = "extrinsics.yml");





__declspec(dllexport) int camCalib(string leftimage, string rightimage, string left_ImgSave, string right_ImgSave, string ParamPath, double chessboard_length, double marker_length) {

	//双目标定


	vector<Point3f> objectPoints1;
	for (int x = 0; x <= 8; x++)
	{
		for (int y = 13; y >= 0; y--)
		{
			objectPoints1.push_back(cv::Point3f((float)x, (float)y, 0) * chessboard_length);
		}
	}


	Mat currentCharucoCornersL, currentCharucoIdsL;
	Mat currentCharucoCornersR, currentCharucoIdsR;
	Ptr<aruco::DetectorParameters> detectorParams = aruco::DetectorParameters::create();
	Ptr<aruco::Dictionary> dictionary =
		aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(2));
	Ptr<aruco::CharucoBoard> charucoboard =
		aruco::CharucoBoard::create(15, 10, chessboard_length, marker_length, dictionary);
	Ptr<aruco::Board> board = charucoboard.staticCast<aruco::Board>();

	std::vector<Mat> allimgids;


	//std::string pattern_jpg_0 = leftimage;//left
	//std::string pattern_jpg_1 = rightimage;//right

	//leftimage = "G:\\标定照片\\L";
	//rightimage = "G:\\标定照片\\R";

	std::vector<cv::String> image_files_0;
	std::vector<cv::String> image_files_1;

	cv::glob(leftimage, image_files_0);
	cv::glob(rightimage, image_files_1);

	if (image_files_0.size() == 0) {
		std::cout << "No image files[jpg]" << std::endl;
		return 0;
	}


	Mat image_size_read;
	for (unsigned int frame = 0; frame < 1; ++frame) {
		image_size_read = imread(image_files_0[frame]);
	}

	cv::Size imgSize = image_size_read.size();
	std::vector<std::vector<Point2f>> imagePointsL, imagePointsR;
	std::vector<std::vector< std::vector< cv::Point2f > >>allCornersL, allCornersR;
	std::vector<std::vector<int>>allIdsL, allIdsR;
	std::vector<cv::Mat> allImgsL, allImgsR;
	cv::Size imgSizeL, imgSizeR;
	std::vector<std::vector<cv::Point3f> > objectPoints;

	int count = 0;
	int id_Size = 0;
	for (int i = 0; i < image_files_0.size(); i++) {

		Mat imageL, imageCopyL;
		Mat imageR, imageCopyR;
		std::vector< int > idsL;
		std::vector< int > idsR;
		std::vector< std::vector< cv::Point2f > > cornersL, rejectedL;
		std::vector< std::vector< cv::Point2f > > cornersR, rejectedR;

		imageL = cv::imread(image_files_0[i]);       //切分得到的左原始图像
		imageR = cv::imread(image_files_1[i]);    //切分得到的右原始图像

		//cv::imshow("imageL", imageL);

		// detect markers
		aruco::detectMarkers(imageL, dictionary, cornersL, idsL, detectorParams, rejectedL);
		aruco::detectMarkers(imageR, dictionary, cornersR, idsR, detectorParams, rejectedR);

		aruco::refineDetectedMarkers(imageL, board, cornersL, idsL, rejectedL);
		aruco::refineDetectedMarkers(imageR, board, cornersR, idsR, rejectedR);

		if (idsL.size() > 0 && idsR.size() > 0) {

			/*imageL.copyTo(imageCopyL);
			imageR.copyTo(imageCopyR);
			aruco::drawDetectedMarkers(imageCopyL, cornersL);
			aruco::drawDetectedMarkers(imageCopyR, cornersR);
			imwrite("unreconstrL.jpg", imageCopyL);
			imwrite("unreconstrR.jpg", imageCopyR);*/

			std::vector<int> togeids;
			std::vector< std::vector< cv::Point2f > > recornersL, recornersR;
			for (int x = 0; x < idsL.size(); x++) {
				for (int y = 0; y < idsR.size(); y++) {
					if (idsL[x] == idsR[y]) {
						togeids.push_back(idsL[x]);
						recornersL.push_back(cornersL[x]);
						recornersR.push_back(cornersR[y]);
					}
				}
			}

			/*imageL.copyTo(imageCopyL);
			imageR.copyTo(imageCopyR);
			aruco::drawDetectedMarkers(imageCopyL, recornersL);
			aruco::drawDetectedMarkers(imageCopyR, recornersR);
			imwrite("reconstrL.jpg", imageCopyL);
			imwrite("reconstrR.jpg", imageCopyR);*/



			aruco::interpolateCornersCharuco(recornersL, togeids, imageL, charucoboard, currentCharucoCornersL,
				currentCharucoIdsL);
			aruco::interpolateCornersCharuco(recornersR, togeids, imageR, charucoboard, currentCharucoCornersR,
				currentCharucoIdsR);


			/*aruco::drawDetectedCornersCharuco(imageCopyL, currentCharucoCornersL, currentCharucoIdsL);
			aruco::drawDetectedCornersCharuco(imageCopyR, currentCharucoCornersR, currentCharucoIdsR);
			imwrite("idreconstrL.jpg", imageCopyL);
			imwrite("idreconstrR.jpg", imageCopyR);*/


			//double x, y;
			int rowL = currentCharucoCornersL.rows;
			int rowR = currentCharucoCornersR.rows;

			allimgids.push_back(currentCharucoIdsL);

			imagePointsL.push_back(Mat_<cv::Point2f>(currentCharucoCornersL));
			imagePointsR.push_back(Mat_<cv::Point2f>(currentCharucoCornersR));

			// draw results
			imageL.copyTo(imageCopyL);
			imageR.copyTo(imageCopyR);
			//if (ids.size() > 0) 
			///aruco::drawDetectedMarkers(imageCopyL, recornersL);
			//aruco::drawDetectedMarkers(imageCopyR, recornersR);

			aruco::drawDetectedCornersCharuco(imageCopyL, currentCharucoCornersL, currentCharucoIdsL);
			aruco::drawDetectedCornersCharuco(imageCopyR, currentCharucoCornersR, currentCharucoIdsR);

			vrSlamFunction.saveImage(imageCopyL, left_ImgSave, i);
			vrSlamFunction.saveImage(imageCopyR, right_ImgSave, i);

			if (togeids.size() > 3) {
				allCornersL.push_back(recornersL);
				allCornersR.push_back(recornersR);
				allIdsL.push_back(togeids);
				allIdsR.push_back(togeids);
				allImgsL.push_back(imageL);
				allImgsR.push_back(imageR);
				imgSizeL = imageL.size();
				imgSizeR = imageR.size();
			}

			if (allIdsL.size() < 1 || allIdsR.size() < 1) {
				cerr << "Not enough captures for calibration" << endl;
				return 0;
			}
			cout << "图- " << i << " -处理完毕。。" << endl;
			count++;
			

		}
	}

	std::vector< Mat > rvecsL, tvecsL;
	std::vector< Mat > rvecsR, tvecsR;
	double repErrorL, repErrorR;

	// prepare data for calibration
	std::vector< std::vector< cv::Point2f > > allCornersConcatenatedL, allCornersConcatenatedR;
	std::vector< int > allIdsConcatenatedL, allIdsConcatenatedR;
	std::vector< int > markerCounterPerFrameL, markerCounterPerFrameR;
	//left
	markerCounterPerFrameL.reserve(allCornersL.size());
	for (unsigned int i = 0; i < allCornersL.size(); i++) {
		markerCounterPerFrameL.push_back((int)allCornersL[i].size());
		for (unsigned int j = 0; j < allCornersL[i].size(); j++) {
			allCornersConcatenatedL.push_back(allCornersL[i][j]);
			allIdsConcatenatedL.push_back(allIdsL[i][j]);
		}
	}
	//right
	markerCounterPerFrameR.reserve(allCornersR.size());
	for (unsigned int i = 0; i < allCornersR.size(); i++) {
		markerCounterPerFrameR.push_back((int)allCornersR[i].size());
		for (unsigned int j = 0; j < allCornersR[i].size(); j++) {
			allCornersConcatenatedR.push_back(allCornersR[i][j]);
			allIdsConcatenatedR.push_back(allIdsL[i][j]);
		}
	}

	// calibrate camera using aruco markers
	//left
	double arucoRepErrL, arucoRepErrR;
	Mat cameraMatrixL, distCoeffsL;
	Mat cameraMatrixR, distCoeffsR;
	int calibrationFlags = 0;
	arucoRepErrL = aruco::calibrateCameraAruco(allCornersConcatenatedL, allIdsConcatenatedL,
		markerCounterPerFrameL, board, imgSizeL, cameraMatrixL,
		distCoeffsL, noArray(), noArray(), calibrationFlags);
	//right
	arucoRepErrR = aruco::calibrateCameraAruco(allCornersConcatenatedR, allIdsConcatenatedR,
		markerCounterPerFrameR, board, imgSizeR, cameraMatrixR,
		distCoeffsR, noArray(), noArray(), calibrationFlags);

	// prepare data for charuco calibration
	int nFramesL = (int)allCornersL.size();
	int nFramesR = (int)allCornersR.size();
	std::vector< Mat > allCharucoCornersL, allCharucoCornersR;
	std::vector< Mat > allCharucoIdsL, allCharucoIdsR;
	std::vector< Mat > filteredImagesL, filteredImagesR;
	allCharucoCornersL.reserve(nFramesL);
	allCharucoCornersR.reserve(nFramesR);
	allCharucoIdsL.reserve(nFramesL);
	allCharucoIdsR.reserve(nFramesR);
	//left
	for (int i = 0; i < nFramesL; i++) {
		// interpolate using camera parameters
		Mat currentCharucoCorners, currentCharucoIds;
		aruco::interpolateCornersCharuco(allCornersL[i], allIdsL[i], allImgsL[i], charucoboard,
			currentCharucoCorners, currentCharucoIds, cameraMatrixL,
			distCoeffsL);

		allCharucoCornersL.push_back(currentCharucoCorners);
		allCharucoIdsL.push_back(currentCharucoIds);
		filteredImagesL.push_back(allImgsL[i]);
	}

	if (allCharucoCornersL.size() < 1) {
		cerr << "Not enough left corners for calibration" << endl;
		return 0;
	}
	//right
	for (int i = 0; i < nFramesR; i++) {
		// interpolate using camera parameters
		Mat currentCharucoCorners, currentCharucoIds;
		aruco::interpolateCornersCharuco(allCornersR[i], allIdsR[i], allImgsR[i], charucoboard,
			currentCharucoCorners, currentCharucoIds, cameraMatrixR,
			distCoeffsR);

		allCharucoCornersR.push_back(currentCharucoCorners);
		allCharucoIdsR.push_back(currentCharucoIds);
		filteredImagesR.push_back(allImgsR[i]);
	}

	if (allCharucoCornersR.size() < 1) {
		cerr << "Not enough right corners for calibration" << endl;
		return 0;
	}

	// calibrate camera using charuco
	//left
	repErrorL =
		aruco::calibrateCameraCharuco(allCharucoCornersL, allCharucoIdsL, charucoboard, imgSizeL,
			cameraMatrixL, distCoeffsL, rvecsL, tvecsL, calibrationFlags);
	//right
	repErrorR =
		aruco::calibrateCameraCharuco(allCharucoCornersR, allCharucoIdsR, charucoboard, imgSizeR,
			cameraMatrixR, distCoeffsR, rvecsR, tvecsR, calibrationFlags);

	//reverse(tvecsL.begin(), tvecsL.end());
	//reverse(tvecsR.begin(), tvecsR.end());

	cout << "Left Rep Error: " << repErrorL << endl;
	cout << "Right Rep Error: " << repErrorR << endl;

	//cout << "Left Rep Error Aruco: " << arucoRepErrL << endl;
	//cout << "Right Rep Error Aruco: " << arucoRepErrR << endl;

	//创建objectPoints
	Mat R, T, E, F;


	objectPoints.resize(count);
	for (int i = 0; i < count; i++) {
		for (int nrow = 0; nrow < allimgids[i].rows; nrow++)
		{
			//cout << "allimgids[" << i << "].rows is " << allimgids[i].rows <<endl;

			for (int ncol = 0; ncol < allimgids[i].cols; ncol++)
			{
				//cout << "allimgids[" << i << "] is " << allimgids[i].rows << "," <<allimgids[i].cols << endl;
				int s = allimgids[i].at<int>(nrow, ncol);

				objectPoints[i].push_back(objectPoints1[s]);
			}
		}
	}

	double rms = stereoCalibrate(objectPoints, imagePointsL, imagePointsR,
		cameraMatrixL, distCoeffsL,
		cameraMatrixR, distCoeffsR,
		imgSize, R, T, E, F,
		CV_CALIB_FIX_INTRINSIC,
		TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, 1e-5));


	Mat r1, t1;
	r1 = R.clone();
	t1 = T.clone();

	cout << "stereo_reproErr:" << rms << endl;

	cv::FileStorage fs;
	fs.open(ParamPath + "L_cam_intrinsics.yml", CV_STORAGE_WRITE);

	if (fs.isOpened())
	{
		fs << "camera_matrix" << cameraMatrixL << "distortion_coefficients" << distCoeffsL << "Left_Rep_Error" << repErrorL;
		fs.release();
	}
	else
		std::cout << "Error: can not save the intrinsic parameters\n";



	fs.open(ParamPath + "R_cam_intrinsics.yml", CV_STORAGE_WRITE);

	if (fs.isOpened())
	{
		fs << "camera_matrix" << cameraMatrixR << "distortion_coefficients" << distCoeffsR << "Right_Rep_Error" << repErrorR;
		fs.release();
	}
	else
		std::cout << "Error: can not save the intrinsic parameters\n";
	fs.open(ParamPath + "extrinsics.yml", CV_STORAGE_WRITE);
	if (fs.isOpened())
	{
		fs << "R" << r1 << "T" << t1 << "stereo_reproErr" << rms;
		fs.release();
	}
	else
		std::cout << "Error: can not save the intrinsic parameters\n";

	std::cout << "双目标定完成..." << endl;
	return 0;
}