#pragma once
#include <cv.h>
#include "flowUV.h"

class FlowError
{
public:	
	static float* calcError(flowUV& UV, flowUV& GT, bool display = true);
};
