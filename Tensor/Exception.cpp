//
//  Exception.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/19.
//

#include "Exception.hpp"

namespace otter {

Error::Error(SourceLocation source_location, std::string msg) : Error(std::move(msg), str("Exception raised from ", source_location)) {}

Error::Error(std::string msg, std::string backtrace, const void* caller)
    : msg_(std::move(msg)), backtrace_(std::move(backtrace)), caller_(caller) {
    refresh_what();
}

std::string Error::compute_what(bool include_backtrace) const {
    std::ostringstream oss;

    oss << msg_;

    if (context_.size() == 1) {
    // Fold error and context in one line
        oss << " (" << context_[0] << ")";
    } else {
        for (const auto& c : context_) {
            oss << "\n  " << c;
        }
    }

    if (include_backtrace) {
        oss << "\n" << backtrace_;
    }

    return oss.str();
}

void Error::refresh_what() {
    what_ = compute_what(/*include_backtrace*/ true);
    what_without_backtrace_ = compute_what(/*include_backtrace*/ false);
}

void otterCheckFail(const char* func, const char* file, uint32_t line, const std::string msg) {
    throw Error({func, file, line}, msg);
}

std::ostream& operator<<(std::ostream& out, const SourceLocation& loc) {
    out << loc.function << " at " << loc.file << ":" << loc.line;
    return out;
}

}   // end namespace otter
