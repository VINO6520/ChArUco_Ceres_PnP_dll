#pragma once



#include<iostream>
using namespace std;

int ChAruCoPnP(string leftParam, string rightParam, string exParam,
	string leftpicture, string rightpicture, string poseSavePath,
	int lenth_num, int width_num,
	double squareLength, double markerLength);

int camCalib(string leftimage, string rightimage, string left_ImgSave, 
	string right_ImgSave, string ParamPath, double chessboard_length, double marker_length);

int SiftPnP(string rawImagePath,string leftParam, string rightParam, string exParam,
	string leftpicture, string rightpicture, string poseSavePath);