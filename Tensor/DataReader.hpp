//
//  DataReader.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/8.
//

#ifndef DataReader_hpp
#define DataReader_hpp

#include <stdio.h>

namespace otter {

class DataReader {
public:
    DataReader();
    virtual ~DataReader();
    
    virtual size_t scan(const char* format, void* p) const;
    
    virtual size_t read(void* buf, size_t size) const;
    
    virtual size_t remain() const;
    
    // For read from .hpp
    virtual size_t reference(size_t size, void** buf) const;
};

class DataReaderFromStdio : public DataReader {
public:
    DataReaderFromStdio(FILE *file);
    virtual ~DataReaderFromStdio();
    
    virtual size_t scan(const char* format, void* p) const;
    virtual size_t read(void* buf, size_t size) const;
    virtual size_t remain() const;
private:
    FILE *file_;
};

}

#endif /* DataReader_hpp */
