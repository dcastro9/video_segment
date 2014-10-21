#ifndef PenaltyFunction_H_
#define PenaltyFunction_H_

#include "cv.h"

class PenaltyFunctionCompute{


public:

	enum RobustFunctionType
	{
		Charbonnier,
		GeneralizedCharbonnier,
		Gaussian,
		Geman_mcclure,
		Lorentzian,
		Quadratic
	};

	enum Derivative
	{
		Value,
		First,
		Second
	};

	static void compute(cv::Mat & mat, PenaltyFunctionCompute::RobustFunctionType type, double sigma = 1, PenaltyFunctionCompute::Derivative derivative = Value, const double aSigma = 0.45);

private:
	static void charbonnier(cv::Mat & mat, const double sigma, const PenaltyFunctionCompute::Derivative derivative);
	static void gaussian(cv::Mat & mat, const double sigma, const PenaltyFunctionCompute::Derivative derivative);
	static void generalizedCharbonnier(cv::Mat & mat, const double sigma, const PenaltyFunctionCompute::Derivative derivative, double aSigma);
	static void gemanMcclure(cv::Mat & mat, const double sigma, const PenaltyFunctionCompute::Derivative derivative);
	static void lorentzian(cv::Mat & mat, const double sigma, const PenaltyFunctionCompute::Derivative derivative);
	static void quadratic(cv::Mat & mat, const double sigma, const PenaltyFunctionCompute::Derivative derivative);
};

#endif