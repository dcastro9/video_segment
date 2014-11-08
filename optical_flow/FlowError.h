#pragma once
#include <opencv2/opencv.hpp>
#include "flowUV.h"

class FlowError {
public:	
	static float* calcError(flowUV& UV, flowUV& GT, bool display = true);
};
