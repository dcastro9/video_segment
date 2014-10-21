#include "CRSSparseMat.h"
#include "cv.h"
#include "Defs.h"

CRSSparseMat::CRSSparseMat()
{
}

//TODO:: BOAZ: might be able to cut on some milis. loop unwinding, init list
void CRSSparseMat::build(const COOrdSparseMat& s)
{
	build(s, s._nz);
}

void CRSSparseMat::build(const COOrdSparseMat& s, int nz)
{
	_rows = s._rows;
	_cols = s._cols;
	_rowPtr = new int[_rows + 1];
	_rowPtr[0] = 0;
	//cout << endl << "tally loop" << std::clock() << endl;
	_nz = 0;
	int* pTally;
	int* tally = pTally = new int[_rows + 1];
	memset(tally, 0 , (_rows + 1) * sizeof(int));
	float* pSVal = s._val;
	int* pSRow = s._rowIdx;
	for (int i = 0; i < nz; ++i, ++pSRow){
		if (*(pSVal++) == 0) continue;
		++tally[*pSRow], ++_nz;
	}
	//cout << endl << "tally loop end" << std::clock() << endl;

	int* pRow = _rowPtr;
	int* pFRow = (_rowPtr + 1);
	for (int i = 0; i < _rows; ++i){
		*(pFRow++) = *pRow + *pTally;
		*(pTally++) = *(pRow++);
	}
	//cout << endl << "row loop end" << std::clock() << endl;
	_val = new float[_nz];
	_colIdx = new int[_nz];
	float v;
	int t;
	pSVal = s._val;
	pSRow = s._rowIdx;
	int* pSCol = s._colIdx;
	//cout << endl << "sort loop" << std::clock() << endl;
	for (int i = 0; i < nz; ++i, ++pSCol, ++pSRow)
	{
		v = *(pSVal++);
		if (v == 0) continue;
		t = tally[*pSRow]++; 
		_val[t] = v;
		_colIdx[t] = *pSCol;
	}
	delete[] tally;
	//cout << endl << "sort loop end" << std::clock() << endl;
}


//TODO:: BOAZ: move _val[j] to pointer
void CRSSparseMat::MulVector(const float* x, float* b) {
	for (int i = 0; i < _rows; ++i){
		float temp = 0;
		for (int p = _rowPtr[i]; p < _rowPtr[i+1]; ++p)
			temp += x[_colIdx[p]] * _val[p];
		b[i] = temp;
	}
}


void CRSSparseMat::MulScalar(const float x){
	int* rows = _rowPtr;
	int endOfRow = 0;
	for (int i = 0; i < _nz; ++i)
			_val[i] *= x;
}


CRSSparseMat& CRSSparseMat::operator=(const CRSSparseMat &C)
{
		_rows	= C._rows;
		_cols	= C._cols;
		_nz     = C._nz;
		_val    = C._val;
		_rowPtr = C._rowPtr;
		_colIdx = C._colIdx;
		return *this;
}

//float* CRSSparseMat::operator()(int i, int j)
//{
//#if OPTFLOW_STRICT
//	if (i > _rows || j > _cols || i < 0 || j < 0)	
//		exit(-1); //BOAZ: dont want to throw an exception... but this means KAKA.
//#endif
//	for (int t = _rowPtr[i]; t < _rowPtr[i + 1]; t++)
//		if (_colIdx[t] == j) return &_val[t];
//	return 0;
//}

float CRSSparseMat::operator()(int i, int j)  const
{
#if OPTFLOW_STRICT
	if (i > _rows || j > _cols || i < 0 || j < 0)	
		exit(-1); //BOAZ: dont want to throw an exception... but this means KAKA.
#endif
	for (int t = _rowPtr[i]; t < _rowPtr[i + 1]; t++)
		if (_colIdx[t] == j) return _val[t];
		//else if (_colIdx[t] == j) return 0.0;
	return 0.0;
}

//void CRSSparseMat::set(int i, int j, float val)
//{
//	if (i > _rows || j > _cols || i < 0 || j < 0)	
//		exit(-1); //BOAZ: dont want to throw an exception... but this means KAKA.
//	for (int t = _rowPtr[i]; t < _rowPtr[i + 1]; t++)
//		if (_colIdx[t] == j) _val[t] = val;
//}

CRSSparseMat::~CRSSparseMat(void)
{
	delete[] _val;
	delete[] _rowPtr;
	delete[] _colIdx;
}
