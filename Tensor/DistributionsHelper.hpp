//
//  DistributionsHelper.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/18.
//

#ifndef DistributionsHelper_hpp
#define DistributionsHelper_hpp

#include "Exception.hpp"
#include "Transformation.hpp"

namespace otter {
namespace {

template <typename T>
struct uniform_int_from_to_distribution {
    
    inline uniform_int_from_to_distribution(uint64_t range, int64_t base) {
        range_ = range;
        base_ = base;
    }
    
    template <typename RNG>
    inline T operator()(RNG generator) {
        if ((
             std::is_same<T, int64_t>::value ||
             std::is_same<T, double>::value ||
             std::is_same<T, float>::value) && range_ >= 1ULL << 32) {
                 
                 return transformation::uniform_int_from_to<T>(generator->random64(), range_, base_);
             } else {
                 return transformation::uniform_int_from_to<T>(generator->random(), range_, base_);
             }
    }
    
private:
    uint64_t range_;
    int64_t base_;
};


template <typename T>
struct uniform_int_full_range_distribution {
    
    template <typename RNG>
    inline T operator()(RNG generator) {
        return transformation::uniform_int_full_range<T>(generator->random64());
    }
    
};

template <typename T>
struct uniform_int_distribution {
    
    template <typename RNG>
    inline T operator()(RNG generator) {
        if (std::is_same<T, double>::value || std::is_same<T, int64_t>::value) {
            return transformation::uniform_int<T>(generator->random64());
        } else {
            return transformation::uniform_int<T>(generator->random());
        }
    }
    
};


template <typename T>
struct uniform_real_distribution {
    
    inline uniform_real_distribution(T from, T to) {
        OTTER_CHECK(from <= to, "Expect from <= to but get ", from, " > ", to);
        OTTER_CHECK(to - from <= std::numeric_limits<T>::max(), "Expect to - from < ", std::numeric_limits<T>::max());
        from_ = from;
        to_ = to;
    }
    
    template <typename RNG>
    inline dist_acctype<T> operator()(RNG generator){
        if(std::is_same<T, double>::value) {
            return transformation::uniform_real<T>(generator->random64(), from_, to_);
        } else {
            return transformation::uniform_real<T>(generator->random(), from_, to_);
        }
    }
    
private:
    T from_;
    T to_;
};

#define DISTRIBUTION_HELPER_GENERATE_HAS_MEMBER(member)              \
template <typename T>                                                \
struct has_member_##member                                           \
{                                                                    \
    typedef char yes;                                                \
    typedef long no;                                                 \
    template <typename U> static yes test(decltype(&U::member));     \
    template <typename U> static no test(...);                       \
    static constexpr bool value = sizeof(test<T>(0)) == sizeof(yes); \
}
DISTRIBUTION_HELPER_GENERATE_HAS_MEMBER(next_double_normal_sample);
DISTRIBUTION_HELPER_GENERATE_HAS_MEMBER(set_next_double_normal_sample);
DISTRIBUTION_HELPER_GENERATE_HAS_MEMBER(next_float_normal_sample);
DISTRIBUTION_HELPER_GENERATE_HAS_MEMBER(set_next_float_normal_sample);
#define DISTRIBUTION_HELPER_GENERATE_NEXT_NORMAL_METHODS(TYPE)                                      \
                                                                                                    \
template <typename RNG, typename ret_type,                                                          \
          typename std::enable_if_t<(                                                               \
            has_member_next_##TYPE##_normal_sample<RNG>::value &&                                   \
            has_member_set_next_##TYPE##_normal_sample<RNG>::value                                  \
          ), int> = 0>                                                                              \
inline bool maybe_get_next_##TYPE##_normal_sample(RNG* generator, ret_type* ret) {  \
  if (generator->next_##TYPE##_normal_sample()) {                                                   \
    *ret = *(generator->next_##TYPE##_normal_sample());                                             \
    generator->set_next_##TYPE##_normal_sample(TYPE());                              \
    return true;                                                                                    \
  }                                                                                                 \
  return false;                                                                                     \
}                                                                                                   \
                                                                                                    \
template <typename RNG, typename ret_type,                                                          \
          typename std::enable_if_t<(                                                               \
            !has_member_next_##TYPE##_normal_sample<RNG>::value ||                                  \
            !has_member_set_next_##TYPE##_normal_sample<RNG>::value                                 \
          ), int> = 0>                                                                              \
inline bool maybe_get_next_##TYPE##_normal_sample(RNG* /*generator*/, ret_type* /*ret*/) {  \
  return false;                                                                                     \
}                                                                                                   \
                                                                                                    \
template <typename RNG, typename ret_type,                                                          \
          typename std::enable_if_t<(                                                               \
            has_member_set_next_##TYPE##_normal_sample<RNG>::value                                  \
          ), int> = 0>                                                                              \
inline void maybe_set_next_##TYPE##_normal_sample(RNG* generator, ret_type cache) { \
  generator->set_next_##TYPE##_normal_sample(cache);                                                \
}                                                                                                   \
                                                                                                    \
template <typename RNG, typename ret_type,                                                          \
          typename std::enable_if_t<(                                                               \
            !has_member_set_next_##TYPE##_normal_sample<RNG>::value                                 \
          ), int> = 0>                                                                              \
inline void maybe_set_next_##TYPE##_normal_sample(RNG* /*generator*/, ret_type /*cache*/) { \
}
DISTRIBUTION_HELPER_GENERATE_NEXT_NORMAL_METHODS(double);
DISTRIBUTION_HELPER_GENERATE_NEXT_NORMAL_METHODS(float);

template <typename T>
struct normal_distribution {
    
    inline normal_distribution(T mean_in, T stdv_in) {
        OTTER_CHECK(stdv_in >= 0, "stdv_in must be positive: ", stdv_in);
        mean = mean_in;
        stdv = stdv_in;
    }
    
    template <typename RNG>
    inline dist_acctype<T> operator()(RNG generator){
        dist_acctype<T> ret;
        // return cached values if available
        if (std::is_same<T, double>::value) {
            if (maybe_get_next_double_normal_sample(generator, &ret)) {
                return transformation::normal(ret, mean, stdv);
            }
        } else {
            if (maybe_get_next_float_normal_sample(generator, &ret)) {
                return transformation::normal(ret, mean, stdv);
            }
        }
        // otherwise generate new normal values
        uniform_real_distribution<T> uniform(0.0, 1.0);
        const dist_acctype<T> u1 = uniform(generator);
        const dist_acctype<T> u2 = uniform(generator);
        const dist_acctype<T> r = ::sqrt(static_cast<T>(-2.0) * ::log(static_cast<T>(1.0)-u2));
        const dist_acctype<T> theta = static_cast<T>(2.0) * static_cast<T>(M_PI) * u1;
        if (std::is_same<T, double>::value) {
            maybe_set_next_double_normal_sample(generator, r * ::sin(theta));
        } else {
            maybe_set_next_float_normal_sample(generator, r * ::sin(theta));
        }
        ret = r * ::cos(theta);
        return transformation::normal(ret, mean, stdv);
    }
    
private:
    T mean;
    T stdv;
};

template <typename T>
struct DiscreteDistributionType { using type = float; };

template <> struct DiscreteDistributionType<double> { using type = double; };

template <typename T>
struct bernoulli_distribution {
    
    inline bernoulli_distribution(T p_in) {
        OTTER_CHECK(p_in >= 0 && p_in <= 1, "Expect 0 <= p_in <= 1 but get ", p_in);
        p = p_in;
    }
    
    template <typename RNG>
    inline T operator()(RNG generator) {
        uniform_real_distribution<T> uniform(0.0, 1.0);
        return transformation::bernoulli<T>(uniform(generator), p);
    }
    
private:
    T p;
};

template <typename T>
struct geometric_distribution {
    
    inline geometric_distribution(T p_in) {
        OTTER_CHECK(p_in > 0 && p_in < 1, "Expect 0 < p_in < 1 but get ", p_in);
        p = p_in;
    }
    
    template <typename RNG>
    inline T operator()(RNG generator) {
        uniform_real_distribution<T> uniform(0.0, 1.0);
        return transformation::geometric<T>(uniform(generator), p);
    }
    
private:
    T p;
};

template <typename T>
struct exponential_distribution {
    
    inline exponential_distribution(T lambda_in) {
        lambda = lambda_in;
    }
    
    template <typename RNG>
    inline T operator()(RNG generator) {
        uniform_real_distribution<T> uniform(0.0, 1.0);
        return transformation::exponential<T>(uniform(generator), lambda);
    }
    
private:
    T lambda;
};

template <typename T>
struct cauchy_distribution {
    
    inline cauchy_distribution(T median_in, T sigma_in) {
        median = median_in;
        sigma = sigma_in;
    }
    
    template <typename RNG>
    inline T operator()(RNG generator) {
        uniform_real_distribution<T> uniform(0.0, 1.0);
        return transformation::cauchy<T>(uniform(generator), median, sigma);
    }
    
private:
    T median;
    T sigma;
};

template <typename T>
struct lognormal_distribution {
    
    inline lognormal_distribution(T mean_in, T stdv_in) {
        OTTER_CHECK(stdv_in > 0, "Expect stdv_in > 0 but get ", stdv_in);
        mean = mean_in;
        stdv = stdv_in;
    }
    
    template<typename RNG>
    inline T operator()(RNG generator){
        normal_distribution<T> normal(mean, stdv);
        return transformation::log_normal<T>(normal(generator));
    }
    
private:
    T mean;
    T stdv;
};

}   // end namespace
}   // end namespace otter

#endif /* DistributionsHelper_hpp */
