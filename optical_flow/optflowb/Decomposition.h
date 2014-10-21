#pragma once
#include <cv.h>

class Decomposition
{
public:	
	static void structureTextureDecompositionRof(const cv::Mat& in1,const cv::Mat& in2,cv::Mat& texture1,cv::Mat& texture2, cv::Mat* structure1, cv::Mat* structure2,float theta,int nIters,float alpha, bool display = false);

private:
	static void findMinMax(const cv::Mat & img,float &min,float &max);
	static void Reproject(cv::Mat & p0,cv::Mat & p1);
	static void rescale(cv::Mat & img1,cv::Mat & img2,float minv,float maxv);
};
