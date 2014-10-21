#pragma once
#include "cv.h"
#include "FArray.h"

class UtilsMat
{
public:
	template<class T> static void scale(cv::Mat& m, T valHigh, T valLow){
		T min = INT_MAX, max = INT_MIN;
		for (int i = 0; i < m.rows * m.cols; ++i){
			if (((T*)m.data)[i] < min){
				min = ((T*)m.data)[i];
			}else if (((T*)m.data)[i] > max){
				max =  ((T*)m.data)[i];
			}
		}

		m = (m - min) / (max - min) * (valHigh - valLow) + valLow;
	}

	//floor every value of the src mat
	template<class T> static void floor(const cv::Mat& src, cv::Mat& dst){
		for(int i = 0; i < src.cols * src.rows; ++i){
			((T*)dst.data)[i] = std::floor(((T*)src.data)[i]);
		}
	}

	//ceil every value of the src mat
	template<class T> static void ceil(const cv::Mat& src, cv::Mat& dst){
		for(int i = 0; i < src.cols * src.rows; ++i){
			((T*)dst.data)[i] = std::ceil(((T*)src.data)[i]);
		}
	}

	//round every value of the src mat to the dst mat
	template<class T> static void round(const cv::Mat& src, cv::Mat& dst){
		T r = 0;
		for(int i = 0; i < src.cols * src.rows; ++i){
			r = ((T*)src.data)[i];
			((T*)dst.data)[i] = (r > 0.0) ? std::floor(r + 0.5) : std::ceil(r - 0.5);
		}
	}
	
	//round every value of the src mat
	template<class T> static void round(cv::Mat& src){
		T r = 0;
		for(int i = 0; i < src.cols * src.rows; ++i){
			r = ((T*)src.data)[i];
			((T*)src.data)[i] = (r > 0.0) ? std::floor(r + 0.5) : std::ceil(r - 0.5);
		}
	}

	//min every value of the m mat with the val param
	template<class T> static void min(const cv::Mat& m, T val){
		for(int i = 0; i < src.cols * src.rows; ++i){
			((T*)dst.data)[i] = std::min<T>(val, ((T*)src.data)[i]);
		}
	}

	//max every value of the src m with the val param
	template<class T> static void max(const cv::Mat& m, T val){
		for(int i = 0; i < src.cols * src.rows; ++i){
			((T*)dst.data)[i] = std::max<T>(val, ((T*)src.data)[i]);
		}
	}

	//clamp every value of the m mat between the 2 param values
	template<class T> static void clamp(const cv::Mat& m, T minVal, T maxVal){
		for(int i = 0; i < m.cols * m.rows; ++i){
			((T*)m.data)[i] = std::max<T>(minVal, std::min<T>(maxVal, ((T*)m.data)[i]));
		}
	}

	static void clamp(FArray& m, float minVal, float maxVal){
		for(size_t i = 0; i < m.size(); ++i){
			m.ptr[i] = std::max<float>(minVal, std::min<float>(maxVal, m.ptr[i]));
		}
	}

	//multiply a "vector" in a cv::Mat disguise (vec) and the diag cv::Mat which is a Matrix representing a diagonal
	//in a huge matrix (diag), the diagonal var specify exactly which diagonal it is, negative are the lower diagonals
	// positives are the upper diagonals and 0 is the diagonal
	template<class T> static void mulDiagVec(const cv::Mat& diag, const cv::Mat& vec, cv::Mat& dst, const int diagonal){
		T* p = (T*)dst.data;
		if (diagonal < 0){
			for(int i = 0; i < -diagonal; ++i){
				*(p++) = 0;
			}
		}
		T* pD = (T*)diag.data;
		T* pV = (T*)vec.data;
		for(int i = 0; i < diag.cols * diag.rows - abs(diagonal); ++i){
			*(p++) = *(pD++) * *(pV++);
		}
		if (diagonal > 0){
			for(int i = 0; i < diagonal; ++i){
				*(p++) = 0;
			}
		}
	}
};

