#pragma once
#include "Defs.h"
class flowUV
{
public:

	explicit flowUV(){}

	flowUV(const cv::Mat U, const cv::Mat V){
		U.copyTo(this->_u);
		V.copyTo(this->_v);
	}

	flowUV(const flowUV& flow){
		flow._u.copyTo(this->_u);
		flow._v.copyTo(this->_v);
	}

	flowUV(int rows, int cols):_u(rows, cols, OPTFLOW_TYPE, cv::Scalar(0)), _v(rows, cols, OPTFLOW_TYPE, cv::Scalar(0)){}

	~flowUV(void)
	{
		_u.release();
		_v.release();
	}

	void reshape(int rows, int cols){
		float ratio = (float)rows / (float)_u.rows;
		cv::Mat tmpU(_u);
		_u.create(rows, cols, _u.type());
		cv::resize(tmpU, _u, cv::Size(cols, rows)); 	
		_u *= ratio;
		tmpU.release();

		cv::Mat tmpV(_v);
		_v.create(rows, cols, _v.type());
		cv::resize(tmpV, _v, cv::Size(cols, rows)); 	
		_v *= ratio;
		tmpV.release();

	}
	
	cv::Mat& getU(){
		return _u;
	}

	cv::Mat& getV(){
		return _v;
	}

private:
	cv::Mat _u;
	cv::Mat _v;
};
