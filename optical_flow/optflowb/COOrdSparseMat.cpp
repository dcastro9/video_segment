#include "COOrdSparseMat.h"
#include "cv.h"

COOrdSparseMat::COOrdSparseMat(	int M, int N, 
								int nz, float* val, int* row, int* col) :
								_val(val), _rowIdx(row),
								_colIdx(col), _nz(nz), _rows(M), _cols(N)
{
}


COOrdSparseMat& COOrdSparseMat::operator=(const COOrdSparseMat &other)
{
	_rows = other._rows;
	_cols = other._cols;
	_nz = other._nz;
	_val = other._val;
	_rowIdx = other._rowIdx;
	_colIdx = other._colIdx;
	return *this;
}

float COOrdSparseMat::operator()(int i, int j)  const
{
	if (i > _rows || j > _cols || i < 0 || j < 0)	
		exit(-1); //BOAZ: dont want to throw an exception... but this means KAKA.
	for (int t=0; t < _nz; t++)
		if (_rowIdx[t] == i && _colIdx[t] == j) return _val[t];
	return 0.0;
}


void COOrdSparseMat::set(int i, int j, float val)
{
	if (i > _rows || j > _cols || i < 0 || j < 0)	
		exit(-1); //BOAZ: dont want to throw an exception... but this means KAKA.
	for (int t = 0; t < _nz; t++)
		if (_rowIdx[t] == i && _colIdx[t] == j) _val[t] = val;
}


COOrdSparseMat::~COOrdSparseMat(void)
{
	delete[] _val;
	delete[] _rowIdx;
	delete[] _colIdx;
}
