#include <iostream>
#include "PenaltyFunctionCompute.h"
#include "UtilsDebug.h"

#define _USE_MATH_DEFINES
#include <math.h>

// Charbonnier Penatly Function Implementation
void PenaltyFunctionCompute::charbonnier(cv::Mat & mat, const double sigma, const PenaltyFunctionCompute::Derivative derivative){
		cv::Mat tmp = mat.clone();		// Probably not so efficiant.
	
		switch(derivative){

			case PenaltyFunctionCompute::Value:
				tmp = (1 + (tmp.mul(tmp))) / (sigma * sigma);			// tmp not really needed here
				cv::sqrt(tmp,tmp);
				mat = (sigma * tmp);
				break;
				
			case PenaltyFunctionCompute::First:
				tmp = (1 + (tmp.mul(tmp))) / (sigma * sigma);
				cv::sqrt(tmp,tmp);
				mat = mat / (tmp / sigma);
				break;

			case PenaltyFunctionCompute::Second:
				tmp = (1 + (tmp.mul(tmp))) / (sigma * sigma);
				cv::sqrt(tmp,tmp);
				mat = 1 / (tmp / sigma);
				break;

			default:
						// Should throw an exception
				break;
		}
}

void PenaltyFunctionCompute::generalizedCharbonnier(cv::Mat & mat, const double sigma, const PenaltyFunctionCompute::Derivative derivative, double aSigma){
		cv::Mat tmp = mat.clone();		// Probably not so efficiant.
	
		aSigma = 0.45;
		switch(derivative){

			case PenaltyFunctionCompute::Value:
				mat = ((sigma * sigma) + mat.mul(mat));			// tmp not really needed here
				cv::pow(mat,aSigma,mat);
				break;
				
			case PenaltyFunctionCompute::First:
				mat = (2*aSigma*mat).mul((sigma * sigma) + mat.mul(mat));
				cv::pow(mat,(aSigma - 1),mat);
				break;

			case PenaltyFunctionCompute::Second:
				tmp = (sigma * sigma) + mat.mul(mat);
				cv::pow(tmp,(aSigma - 1),mat);
				mat *= 2*aSigma;

				break;
		}
}

// Gaussian Penatly Function Implementation
void PenaltyFunctionCompute::gaussian(cv::Mat & mat, const double sigma, const PenaltyFunctionCompute::Derivative derivative){
		switch(derivative){

			case PenaltyFunctionCompute::Value:
				mat =  (0.5 * log(2 * M_PI)) + (log(sigma)) + (0.5 * (mat.mul(mat)) / (sigma*sigma));
				break;

			case PenaltyFunctionCompute::First:
				mat = mat / (sigma * sigma);
				break;

			case PenaltyFunctionCompute::Second:
				mat.setTo(cv::Scalar(1/(sigma * sigma)));
				break;

			default:
						// Should throw an exception
				break;
		}
}

// Geman_mcclure Penatly Function Implementation
void PenaltyFunctionCompute::gemanMcclure(cv::Mat & mat, const double sigma, const PenaltyFunctionCompute::Derivative derivative){
		cv::Mat tmp = mat.clone();
	
		switch(derivative){

			case PenaltyFunctionCompute::Value:
				mat =  (mat.mul(mat)) / ((sigma*sigma) + (mat.mul(mat)));
				break;

			case PenaltyFunctionCompute::First:
				tmp = ((sigma*sigma) + tmp.mul(tmp));
				pow(tmp, 2, tmp);
				mat = (2 * (sigma * sigma) * mat) / tmp;
				break;

			case PenaltyFunctionCompute::Second:
				tmp = ((sigma*sigma) + tmp.mul(tmp));
				pow(tmp, 2, tmp);
				mat = (2 * (sigma * sigma)) / tmp;
				break;

			default:
						// Should throw an exception
				break;
		}
}

// Lorentzian Penatly Function Implementation
void PenaltyFunctionCompute::lorentzian(cv::Mat & mat, const double sigma, const PenaltyFunctionCompute::Derivative derivative){
		switch(derivative){

			case PenaltyFunctionCompute::Value:
				cv::log((1 + mat.mul(mat)) / (2 * (sigma * sigma)),mat);
				break;

			case PenaltyFunctionCompute::First:
				mat = (2 * mat) / ((2 * (sigma * sigma)) + mat.mul(mat));
				break;

			case PenaltyFunctionCompute::Second:
				mat = 2 / ((2 * (sigma * sigma)) + mat.mul(mat));
				break;

			default:
						// Should throw an exception
				break;
		}
}

// Quadratic Penatly Function Implementation
void PenaltyFunctionCompute::quadratic(cv::Mat & mat, const double sigma, const PenaltyFunctionCompute::Derivative derivative){
		switch(derivative){

			case PenaltyFunctionCompute::Value:
				mat = (mat.mul(mat)) / (sigma * sigma);
				break;

			case PenaltyFunctionCompute::First:
				mat = (mat * 2) / (sigma * sigma);
				break;

			case PenaltyFunctionCompute::Second:
				mat.setTo(cv::Scalar(2/(sigma * sigma)));
				break;

			default:
						// Should throw an exception
				break;
		}
}







void PenaltyFunctionCompute::compute(cv::Mat & mat, PenaltyFunctionCompute::RobustFunctionType type, const double sigma, const PenaltyFunctionCompute::Derivative derivative, const double aSigma){
	switch(type){
		case Charbonnier:
			charbonnier(mat, sigma, derivative);
			return;
		case GeneralizedCharbonnier:
			generalizedCharbonnier(mat, sigma, derivative, aSigma);
			return;
		case Gaussian:
			gaussian(mat, sigma, derivative);
			return;
		case Geman_mcclure:
			gemanMcclure(mat, sigma, derivative);
			return;
		case Lorentzian:
			lorentzian(mat, sigma, derivative);
			return;
		case Quadratic:
			quadratic(mat, sigma, derivative);
			return;
		default:
			return;
		}
}