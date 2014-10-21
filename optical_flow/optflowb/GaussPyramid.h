#include "cv.h"
#include "UtilsDebug.h"
#include "highgui.h" 
#include "Defs.h"
#include <vector>
class GaussPyramid
{
private:
	vector<cv::Mat> ImPyramid;
	int nLevels;
	

public:
	GaussPyramid(const cv::Mat image, const int levels, const float levelSpacing) : nLevels(levels) {
		static float f[25] = {	0.00296901674395050f,	0.0133062098910137f,	0.0219382312797146f,	0.0133062098910137f,	0.00296901674395050f,
								0.0133062098910137f,	0.0596342954361801f,	0.0983203313488458f,	0.0596342954361801f,	0.0133062098910137f,
								0.0219382312797146f,	0.0983203313488458f,	0.162102821637127f,		0.0983203313488458f,	0.0219382312797146f,
								0.0133062098910137f,	0.0596342954361801f,	0.0983203313488458f,	0.0596342954361801f,	0.0133062098910137f,
								0.00296901674395050f,	0.0133062098910137f,	0.0219382312797146f,	0.0133062098910137f,	0.00296901674395050f};
		cv::Mat kernel(5, 5, OPTFLOW_TYPE, f);

		cv::Mat img(image);
		ImPyramid.push_back(img);
		for(int i = 1; i < levels; ++i){ 
			cv::Mat smooth(img.rows, img.cols, img.type());
			cv::filter2D(img, smooth, img.depth(), kernel, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
			cv::Mat tmp;
			cv::resize(smooth, tmp, cv::Size((int)round( smooth.cols * levelSpacing), (int)round(smooth.rows * levelSpacing)), 0, 0, cv::INTER_LINEAR);
			ImPyramid.push_back(tmp);
			img = tmp;
		}
	}

	~GaussPyramid(void) {
		for (size_t i = 0; i < ImPyramid.size(); ++i)
			ImPyramid[i].release();
	}

	int getNlevels() const {
		return nLevels;
	}

	cv::Mat& operator[](int level) {
		return ImPyramid[nLevels - level - 1];
	}

	static float round(float r) {
		return (r > 0.0f) ? floor(r + 0.5f) : ceil(r - 0.5f);
	}
};