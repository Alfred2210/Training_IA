#include "../pch.h"

#include"../include/Matrix.h"


const double& Matrix::operator()(int i, int j) const
{
	return m_data[i * m_columns + j];
}
double& Matrix::operator()(int i, int j)
{
	return m_data[i * m_columns + j];
}
int Matrix::getRows() const
{
	return m_rows;
}
int Matrix::getColums() const
{
	return m_columns;
}

int Matrix::getLength() const
{
	return m_rows * m_columns;
}

Matrix Matrix::operator-(const Matrix& matrix) const 
{
	Matrix result(m_rows,m_columns);

	for (size_t i = 0; i < matrix.getRows(); i++)
	{
		for (size_t j = 0; j < matrix.getColums(); j++)
		{
			result(i, j) = (*this)(i, j) - matrix(i, j);
		}
	}
	return result;
}

bool Matrix::operator==(const Matrix& A) const
{
	if(A.getRows() != m_rows || A.getColums() != m_columns)
		return false;

	for (size_t i = 0; i < m_data.size(); i++)
	{
		if (m_data[i] != A.m_data[i])
			return false;
	}

	return true;
}

bool Matrix::operator!=(const Matrix& matrix) const
{
	return !(*this == matrix);
}