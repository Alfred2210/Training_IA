#pragma once
#include "../framework.h"
#include "../include/Matrix.h"
#include <string>

namespace FileUtils
{
	
	EXPORT_API void loadData(const std::string& pathFile, Matrix& X, Matrix& y);
	EXPORT_API std::string readFile(const std::string& pathFile);
	EXPORT_API void parse(const std::string& data, Matrix& X, Matrix& Y);
	EXPORT_API std::pair<int, int> count(const std::string& data);
}

