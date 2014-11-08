#include "CRSSparseMat.h"
#include "FArray.h"

class LinearSolver
{
public:	
	static void sparseMatSor(CRSSparseMat& A, FArray& X0, FArray& X, FArray& B, const float w, const int numOfIterations, const float residualTolerance);
	static void sparseMatSorNoResidual(CRSSparseMat& A, FArray& X0, FArray& X, FArray& B, const float w, const int numOfIterations);
	static double residual(CRSSparseMat& A, FArray& X, FArray& B);

	//enum CycleType{
	//	vCycle=1,
	//	wCycle=2
	//};

	//static void multigrid(const int maxIterations, const int maxLevels, CRSSparseMat& A, FArray& B, const float w, FArray& X, const int preIterations, const int postIterations, const CycleType cycleType);

//private:
	//static void multigridIteration(const int currentLevel, CRSSparseMat& A, FArray& B, const float w, FArray& X, const int preIterations, const int postIterations, const CycleType cycleType);

	//static void restrictionX(const FArray& Xh, FArray& X2h);
	//static void restrictionV(const FArray& Xh, FArray& X2h);
	//static void restrictionM(CRSSparseMat& Ah, CRSSparseMat& A2h);
	//static void prolongation(const FArray& X2h, FArray& Xh);
	//static void coarseGridOperator(CRSSparseMat& Ah, const FArray& X2h, FArray& B2h);
	//static void mresidual(CRSSparseMat& A, const FArray& B, const FArray& X, FArray& residualVec);
	//static void sumVectors(const FArray& vec1, const FArray& vec2, FArray& vecResult, int sign);
};