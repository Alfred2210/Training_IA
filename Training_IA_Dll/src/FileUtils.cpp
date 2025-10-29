//#include "../pch.h"
//
//#include "../include/FileUtils.h"
//
//#include <fstream>
//#include <sstream>
//#include <iostream>
//
//
//void FileUtils::loadData(const std::string& pathFile, Matrix& X, Matrix& y)
//{
//	std::string data = readFile(pathFile);
//	parse(data, X, y);
//}
//
//std::string FileUtils::readFile(const std::string& pathFile)
//{
//	std::ifstream file{ pathFile };
//
//
//	if (!file)
//	{
//		std::cerr << "Erreur : impossible d'ouvrir le fichier " << pathFile << std::endl;
//		return "";
//	}
//	std::string header;
//	std::getline(file, header);
//
//	std::ostringstream buffer;
//	buffer << file.rdbuf();
//
//	return buffer.str();
//}
//
//void FileUtils::parse(const std::string& data, Matrix& X, Matrix& y)
//{
//	std::stringstream filedata(data);
//	std::string line;
//	int row = 0;
//
//	while (std::getline(filedata, line))
//	{
//		std::vector<int> features;
//		std::stringstream parsedata(line);
//		std::string line_;
//
//		while (std::getline(parsedata, line_, ','))
//		{
//			//converti string en int
//			features.push_back(std::stoi(line_));
//		}
//
//		for (size_t i = 0; i < features.size() - 1; i++)
//		{
//			X(row, i) = features[i];
//		}
//
//		int action = features.back();
//
//		if (action >= 2)
//			y(row, 0) = 1.0;
//		else
//			y(row, 0) = -1.0;
//		row++;
//	}
//
//}
//
//std::pair<int, int> FileUtils::count(const std::string& data)
//{
//	std::stringstream filedata(data);
//	std::string line;
//	double rows = 0;
//	double colums = 0;
//
//	while (std::getline(filedata, line))
//	{
//		std::stringstream parsedata(line);
//		std::string line_;
//		if (rows == 0)
//		{
//			while (std::getline(parsedata, line_, ','))
//			{
//				colums++;
//			}
//			colums--;
//		}
//		rows++;
//	}
//
//	return { rows,colums };
//}