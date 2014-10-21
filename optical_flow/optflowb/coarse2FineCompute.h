#pragma once


#include "highgui.h" 
#include "cv.h"
#include "GaussPyramid.h"
#include "flowUV.h"
#include <math.h>
#include "LinearSolver.h"
#include "OpticalFlowParams.h"

class OpticalFlow
{
public:
	static flowUV* calculate (const cv::Mat& Im1, const cv::Mat& Im2, const OpticalFlowParams& params, flowUV* oldFlow = NULL);


private:
	static void WarpImage (const cv::Mat& im, const cv::Mat& u,const cv::Mat& v, cv::Mat& dst);
	static void baseCalculate (cv::Mat& Im1, cv::Mat& Im2, flowUV& UV, const OpticalFlowParams& params);
};
