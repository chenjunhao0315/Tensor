//
//  Exception.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/19.
//

#ifndef Exception_hpp
#define Exception_hpp

#include <vector>
#include <ostream>
#include <sstream>

#include "StringUtils.hpp"

namespace otter {

struct SourceLocation {
    const char* function;
    const char* file;
    uint32_t line;
};

std::ostream& operator<<(std::ostream& out, const SourceLocation& loc);

class Error : public std::exception {
    std::string msg_;
    std::string backtrace_;
    const void* caller_;
    std::vector<std::string> context_;

    std::string what_;
    std::string what_without_backtrace_;
public:
    Error(SourceLocation source_location, std::string msg);
    Error(std::string msg, std::string backtrace, const void* caller = nullptr);
    
    const std::string& msg() const {
        return msg_;
    }
    
    const std::vector<std::string>& context() const {
        return context_;
    }
    
    const std::string& backtrace() const {
        return backtrace_;
    }
    
    const char* what() const noexcept override {
        return what_.c_str();
    }
    
    const void* caller() const noexcept {
        return caller_;
    }
private:
    void refresh_what();
    std::string compute_what(bool include_backtrace) const;
};

template <typename... Args>
decltype(auto) torchCheckMsgImpl(const char* msg, const Args&... args) {
    return str(args...);
}
inline const char* torchCheckMsgImpl(const char* msg) {
    return msg;
}

inline const char* torchCheckMsgImpl(const char* msg, const char* args) {
    return args;
}

#define OTTER_CHECK_MSG(cond, type, ...)                    \
    (torchCheckMsgImpl(                                     \
    "Expected " #cond                                       \
    " to be true, but got false.  "                         \
    "(Could this error message be improved?  If so, "       \
    "please report an enhancement request to PyTorch.)",    \
    ##__VA_ARGS__))

void otterCheckFail(const char* func, const char* file, uint32_t line, const std::string msg);

#define OTTER_CHECK(cond, ...) \
if (!(cond)) { \
    otter::otterCheckFail(__func__, __FILE__, static_cast<uint32_t>(__LINE__), OTTER_CHECK_MSG(cond, "", __VA_ARGS__)); \
}

}   // end namespace otter

#endif /* Exception_hpp */
