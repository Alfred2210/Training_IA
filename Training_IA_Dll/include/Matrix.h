#pragma once

#include "../framework.h"

#include <vector>

class EXPORT_API Matrix
{
public :

	Matrix(int rows, int colums)
		:m_rows(rows), m_columns(colums), m_data(m_rows* m_columns) {
	};

	const double& operator()(int i, int j)const;
	double& operator()(int i, int j);
	 int getRows() const;
	 int getColums() const;
	 int getLength()const;
	 Matrix operator-(const Matrix& matrix)const;
	 bool operator==(const Matrix& A)const;
	 bool operator!=(const Matrix& matrix)const;
	 
private :
	int m_rows, m_columns;
	std::vector<double> m_data;
};