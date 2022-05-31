//
//  main.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/29.
//

#include <iostream>

#include "Otter.hpp"
#include "NcnntoOtterConverter.hpp"

using namespace std;

using namespace otter::core;

int main(int argc, const char * argv[]) {
    
    if (argc < 2) {
        printf("Usage: ./ncnn2otter <ncnn-param> <otter-otter>");
        return -1;
    }
    
    OtterLeader convert = otter::ncnn2otter(argv[1]);
    
    convert.saveProject((argc > 2) ? argv[2] : "transform.otter");

	return 0;
}

