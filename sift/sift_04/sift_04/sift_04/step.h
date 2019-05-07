#pragma once
#include"pre.h"
#include"ach.h"

const string img_path1 = "C:\\Users\\28030\\Desktop\\shuzi\\l.bmp", img_path2 = "C:\\Users\\28030\\Desktop\\shuzi\\r.bmp";


class Res {
public:
	Mat img;
	sift *sift;
};

void process(Res r);
