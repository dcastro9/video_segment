#pragma once

#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "flowUV.h"
#include "GaussPyramid.h"
#include "LinearSolver.h"
#include "OpticalFlowParams.h"

class OpticalFlow {
public:
  static flowUV* calculate(
      const cv::Mat& Im1,
      const cv::Mat& Im2,
      const OpticalFlowParams& params,
      flowUV* oldFlow = NULL);


private:
  static void WarpImage(
      const cv::Mat& im,
      const cv::Mat& u,
      const cv::Mat& v,
      cv::Mat& dst);

  static void baseCalculate(
      cv::Mat& Im1,
      cv::Mat& Im2,
      flowUV& UV,
      const OpticalFlowParams& params);
};
