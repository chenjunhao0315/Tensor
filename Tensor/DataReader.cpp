//
//  DataReader.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/8.
//

#include "DataReader.hpp"

namespace otter {

DataReader::DataReader() {
}

DataReader::~DataReader() {
}

size_t DataReader::scan(const char* /*format*/, void* /*p*/) const {
    return 0;
}

size_t DataReader::read(void* /*buf*/, size_t /*size*/) const {
    return 0;
}

size_t DataReader::reference(size_t /*size*/, void** /*buf*/) const {
    return 0;
}

DataReaderFromStdio::DataReaderFromStdio(FILE *file) : DataReader(), file_(file) {
}

DataReaderFromStdio::~DataReaderFromStdio() {
    
}

size_t DataReaderFromStdio::scan(const char *format, void *p) const {
    return fscanf(file_, format, p);
}

size_t DataReaderFromStdio::read(void *buf, size_t size) const {
    return fread(buf, 1, size, file_);
}

}
