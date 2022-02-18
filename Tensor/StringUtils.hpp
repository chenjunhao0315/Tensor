//
//  StringUtils.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/19.
//

#ifndef StringUtils_hpp
#define StringUtils_hpp

#include <ostream>
#include <sstream>

namespace otter {

struct CompileTimeEmptyString {
    operator const std::string&() const {
        static const std::string empty_string_literal;
        return empty_string_literal;
    }
    operator const char*() const {
        return "";
    }
};

template <typename T>
struct CanonicalizeStrTypes {
    using type = const T&;
};

template <size_t N>
struct CanonicalizeStrTypes<char[N]> {
    using type = const char*;
};

inline std::ostream& _str(std::ostream& ss) {
    return ss;
}

template <typename T>
inline std::ostream& _str(std::ostream& ss, const T& t) {
    // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
    ss << t;
    return ss;
}

template <>
inline std::ostream& _str<CompileTimeEmptyString>(
                                                  std::ostream& ss,
                                                  const CompileTimeEmptyString&) {
    return ss;
}

template <typename T, typename... Args>
inline std::ostream& _str(std::ostream& ss, const T& t, const Args&... args) {
    return _str(_str(ss, t), args...);
}

template <typename... Args>
struct _str_wrapper final {
    static std::string call(const Args&... args) {
        std::ostringstream ss;
        _str(ss, args...);
        return ss.str();
    }
};

// Specializations for already-a-string types.
template <>
struct _str_wrapper<std::string> final {
    // return by reference to avoid the binary size of a string copy
    static const std::string& call(const std::string& str) {
        return str;
    }
};

template <>
struct _str_wrapper<const char*> final {
    static const char* call(const char* str) {
        return str;
    }
};

template <>
struct _str_wrapper<> final {
    static CompileTimeEmptyString call() {
        return CompileTimeEmptyString();
    }
};

template <typename... Args>
inline decltype(auto) str(const Args&... args) {
    return _str_wrapper<typename CanonicalizeStrTypes<Args>::type...>::call(args...);
}

}

#endif /* StringUtils_hpp */
