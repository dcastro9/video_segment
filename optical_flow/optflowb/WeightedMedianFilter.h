#ifndef WeightedMedianFilter_H_
#define WeightedMedianFilter_H_

#include "cv.h"

class WeightedMedianFilter{
	
public:
	static void computeMedianFilter(cv::Mat & flowU,cv::Mat & flowV,const  cv::Mat & image1,const  cv::Mat & image2, int radius = 5,int area_hsz = 7, int sigma_i = 7);


private:

	static void divergence(const cv::Mat & flowU, const cv::Mat & flowV, cv::Mat & dest);

	static void partialDeriv(const cv::Mat & flowU, const cv::Mat &  flowV, const cv::Mat & image1, const cv::Mat & image2, cv::Mat & It);

	static void detect_occlusion(const cv::Mat & flowU,const cv::Mat & flowV,const  cv::Mat & image1,const  cv::Mat & image2, cv::Mat & occ);

	static void medFilt(const cv::Mat & src, int filterSize ,cv::Mat & dest);

	static void findEdges(const cv::Mat & src, cv::Mat & dest,double threshold = -1);
		
	static void runMedianFilter(cv::Mat & flowU, cv::Mat &  flowV, const cv::Mat & image1, const cv::Mat & image2, int filterSize,int area_hsz, int sigma_i, cv::Mat & occ);
	

};

#endif