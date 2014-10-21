#pragma once
#include "PenaltyFunctionCompute.h"

class OpticalFlowParams{
private:
	//operator smoothness param
	float _alpha;

	//pyramid parmas
	int _pyramidLevels;
	float _pyramidSpacing;

	//derivative params
	float _weightedDeriveFactor;
	bool _displayDerivativs;

	//median filter param
	int _medianFilterRadius;

	//sor params
	int _sorIters;
	float _overRelaxation;
	bool _checkResidualTolerance;
	float _residualTolerance;

	//number of warping per pyramid level
	int _iters;
	//maximum number of linearization performed per warping, 1 OK for HS
	int _linearIters;

	//penalty function params
	double _penaltySigma;
	PenaltyFunctionCompute::RobustFunctionType _pfSpatialX;
	PenaltyFunctionCompute::Derivative _pfSpatialXDeriv;
	PenaltyFunctionCompute::RobustFunctionType _pfSpatialY;
	PenaltyFunctionCompute::Derivative _pfSpatialYDeriv;
	PenaltyFunctionCompute::RobustFunctionType _pdData;
	PenaltyFunctionCompute::Derivative _pdDataDeriv;


	//texture dec params
	bool _useTextureDecomposition;
	float _TDtheta;
	int _TDnIters;
	float _TDalp;
	bool _displayTextureDecompositionOutput;


	bool _display;

public:

	OpticalFlowParams(	const float alpha = 3.0f, 
						const int pyramidLevels = 5, 
						const float pyramidSpacing = 0.5f,
						const float weightedDeriveFactor = 0.5f,
						const bool displayDerivativs = false,
						const int medianFilterRadius = 5,
						const int sorIters = 100,
						const float overRelaxation = 1.9f,
						const bool checkResidualTolerance = true,
						const float residualTolerance = 0.01f,
						const int iters = 3, 
						const int linearIters = 1,
						const double penaltySigma = 0.01f,
						const PenaltyFunctionCompute::RobustFunctionType pfSpatialX = PenaltyFunctionCompute::Quadratic,
						const PenaltyFunctionCompute::Derivative pfSpatialXDeriv = PenaltyFunctionCompute::Second,
						const PenaltyFunctionCompute::RobustFunctionType pfSpatialY = PenaltyFunctionCompute::Quadratic,
						const PenaltyFunctionCompute::Derivative pfSpatialYDeriv = PenaltyFunctionCompute::Second,
						const PenaltyFunctionCompute::RobustFunctionType pdData = PenaltyFunctionCompute::Quadratic,
						const PenaltyFunctionCompute::Derivative pdDataDeriv = PenaltyFunctionCompute::Second,
						const bool useTextureDecomposition = true,
						const float TDtheta = 1.0f/8.0f,
						const int TDnIters = 100,
						const float TDalp = 0.95f,
						const bool displayTextureDecompositionOutput = false,
						const bool display = true)
	{
		_alpha = alpha;
		_pyramidLevels = pyramidLevels;
		_pyramidSpacing = pyramidSpacing;
		_weightedDeriveFactor = weightedDeriveFactor;
		_displayDerivativs = displayDerivativs;
		_medianFilterRadius = medianFilterRadius;
		_sorIters = sorIters;
		_overRelaxation= overRelaxation;
		_checkResidualTolerance = checkResidualTolerance;
		_residualTolerance = residualTolerance;
		_iters = iters;
		_linearIters = linearIters;
		_penaltySigma = penaltySigma;
		_pfSpatialX = pfSpatialX;
		_pfSpatialXDeriv = pfSpatialXDeriv;
		_pfSpatialY = pfSpatialY;
		_pfSpatialYDeriv = pfSpatialYDeriv;
		_pdData = pdData;
		_pdDataDeriv = pdDataDeriv;
		_useTextureDecomposition = useTextureDecomposition;
		_TDtheta = TDtheta;
		_TDnIters = TDnIters;
		_TDalp = TDalp;
		_displayTextureDecompositionOutput = displayTextureDecompositionOutput;
		_display = display;
	}


	inline float getAlpha() const {return _alpha;}
	
	inline int getPyramidLevels() const {return _pyramidLevels;}
	inline float getPyramidSpacing() const {return _pyramidSpacing;}

	inline float getWeightedDeriveFactor() const {return _weightedDeriveFactor;}
	inline bool getDisplayDerivativs() const {return _displayDerivativs;}

	inline int getMedianFilterRadius() const {return _medianFilterRadius;}

	inline int getSorIters() const {return _sorIters;}
	inline float getOverRelaxation() const {return _overRelaxation;}
	inline bool getCheckResidualTolerance() const {return _checkResidualTolerance;}
	inline float getResidualTolerance() const {return _residualTolerance;}

	inline int getIters() const {return _iters;}
	inline int getLinearIters() const {return _linearIters;}

	inline double getPenaltySigma() const {return _penaltySigma;}
	inline PenaltyFunctionCompute::RobustFunctionType getPfSpatialX() const {return _pfSpatialX;}
	inline PenaltyFunctionCompute::Derivative getPfSpatialXDeriv() const {return _pfSpatialXDeriv;}
	inline PenaltyFunctionCompute::RobustFunctionType getPfSpatialY() const {return _pfSpatialY;}
	inline PenaltyFunctionCompute::Derivative getPfSpatialYDeriv() const {return _pfSpatialYDeriv;}
	inline PenaltyFunctionCompute::RobustFunctionType getPdData() const {return _pdData;}
	inline PenaltyFunctionCompute::Derivative getPdDataDeriv() const {return _pdDataDeriv;}

	inline bool isUseTextureDecomposition() const {return _useTextureDecomposition;}	
	inline float getTDtheta() const {return _TDtheta;}
	inline int getTDnIters() const {return _TDnIters;}
	inline float getTDalp() const {return _TDalp;}
	inline bool getDisplayTextureDecompositionOutput() const {return _displayTextureDecompositionOutput;}

	inline bool isDisplay() const {return _display;}


	~OpticalFlowParams(){}
};