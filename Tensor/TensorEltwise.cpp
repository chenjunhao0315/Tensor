//
//  TensorEltwise.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/6/20.
//

#include "TensorEltwise.hpp"
#include "VecIntrinsic.hpp"
#include "TensorFactory.hpp"
#include "Parallel.hpp"

#if __SSE2__
#include "sse_mathfun.hpp"
#if __AVX__
#include "Avx_Math.hpp"
#endif
#endif

#if __ARM_NEON__
#include "neon_mathfun.hpp"
#endif

namespace otter {

#if __SSE2__
template<typename Op>
static int binary_op_pack4(const Tensor& a, const Tensor& b, Tensor& c)
{
    Op op;

    int elempack = a.elempack();
    ScalarType dtype = a.scalar_type();

    int elempack1 = b.elempack();
    ScalarType dtype1 = b.scalar_type();

    if (a.dim() == 3) {
        int w = a.size(2);
        int h = a.size(1);
        int channels = a.size(0);
        int size = w * h;
        
        if (b.dim() == 3) {
            int w1 = b.size(2);
            int h1 = b.size(1);
            int channels1 = b.size(0);
            int size1 = w1 * h1;
            
            if (w1 == 1 && h1 == 1 && channels1 == channels) {
                // special type 1
                c = otter::empty({channels, h, w}, dtype);
                
                auto a_a = a.accessor<float, 3, 4>();
                auto b_a = b.accessor<float, 3, 4>();
                auto c_a = c.accessor<float, 3, 4>();

                otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        const float* ptr = a_a[q].data();
                        float* outptr = c_a[q].data();
                        const float* b0 = b_a[q].data();
                        __m128 _b0 = _mm_loadu_ps(b0);
                        for (int i = 0; i < size; i++) {
                            __m128 _p = _mm_loadu_ps(ptr);
                            __m128 _outp = op.func_pack4(_p, _b0);
                            _mm_storeu_ps(outptr, _outp);
                            ptr += 4;
                            outptr += 4;
                        }
                    }
                });

                return 0;
            }

            if (w1 == w && h1 == h && channels1 == 1 && elempack1 == 1) {
                // special type 2
                c = otter::empty({channels, h, w}, dtype);
                
                auto a_a = a.accessor<float, 3, 4>();
                auto b_a = b.accessor<float, 3>();
                auto c_a = c.accessor<float, 3, 4>();

                otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        const float* ptr = a_a[q].data();
                        const float* ptr1 = b_a.data();
                        float* outptr = c_a[q].data();
                        for (int i = 0; i < size; i++) {
                            __m128 _p = _mm_loadu_ps(ptr);
                            __m128 _p1 = _mm_set1_ps(*ptr1);
                            __m128 _outp = op.func_pack4(_p, _p1);
                            _mm_storeu_ps(outptr, _outp);
                            ptr += 4;
                            ptr1 += 1;
                            outptr += 4;
                        }
                    }
                });

                return 0;
            }

            if (w == 1 && h == 1 && channels1 == channels) {
                // special type 3
                c = otter::empty({channels1, h1, w1}, dtype1);
                
                auto a_a = a.accessor<float, 3, 4>();
                auto b_a = b.accessor<float, 3, 4>();
                auto c_a = c.accessor<float, 3, 4>();

                otter::parallel_for(0, channels1, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        const float* a0 = a_a[q].data();
                        float* outptr = c_a[q].data();
                        const float* ptr1 = b_a[q].data();
                        __m128 _a0 = _mm_loadu_ps(a0);
                        for (int i = 0; i < size1; i++) {
                            __m128 _p1 = _mm_loadu_ps(ptr1);
                            __m128 _outp = op.func_pack4(_a0, _p1);
                            _mm_storeu_ps(outptr, _outp);
                            ptr1 += 4;
                            outptr += 4;
                        }
                    }
                });

                return 0;
            }

            if (w1 == w && h1 == h && channels == 1 && elempack == 1) {
                // special type 4
                c = otter::empty({channels1, h1, w1}, dtype1);
                
                auto a_a = a.accessor<float, 3>();
                auto b_a = b.accessor<float, 3, 4>();
                auto c_a = c.accessor<float, 3, 4>();

                otter::parallel_for(0, channels1, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        const float* ptr = a_a.data();
                        const float* ptr1 = b_a[q].data();
                        float* outptr = c_a[q].data();
                        for (int i = 0; i < size1; i++) {
                            __m128 _p = _mm_set1_ps(*ptr);
                            __m128 _p1 = _mm_loadu_ps(ptr1);
                            __m128 _outp = op.func_pack4(_p, _p1);
                            _mm_storeu_ps(outptr, _outp);
                            ptr += 1;
                            ptr1 += 4;
                            outptr += 4;
                        }
                    }
                });

                return 0;
            }

            if (w != 1 && w1 == 1 && h1 == h && channels1 == channels) {
                // special type 5
                c = otter::empty({channels, h, w}, dtype);
                
                auto a_a = a.accessor<float, 3, 4>();
                auto b_a = b.accessor<float, 3, 4>();
                auto c_a = c.accessor<float, 3, 4>();

                otter::parallel_for(0, channels1, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        const float* ptr = a_a[q].data();
                        const float* ptr1 = b_a[q].data();
                        float* outptr = c_a[q].data();

                        for (int y = 0; y < h; y++) {
                            __m128 _p1 = _mm_loadu_ps(ptr1 + y * 4);
                            for (int x = 0; x < w; x++)
                            {
                                __m128 _p = _mm_loadu_ps(ptr);
                                __m128 _outp = op.func_pack4(_p, _p1);
                                _mm_storeu_ps(outptr, _outp);

                                ptr += 4;
                                outptr += 4;
                            }
                        }
                    }
                });

                return 0;
            }

            if (w1 == w && h != 1 && h1 == 1 && channels1 == channels) {
                // special type 6
                c = otter::empty({channels, h, w}, dtype);
                
                auto a_a = a.accessor<float, 3, 4>();
                auto b_a = b.accessor<float, 3, 4>();
                auto c_a = c.accessor<float, 3, 4>();

                otter::parallel_for(0, channels1, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        const float* ptr = a_a[q].data();
                        const float* ptr1 = b_a[q].data();
                        float* outptr = c_a[q].data();

                        for (int y = 0; y < h; y++) {
                            for (int x = 0; x < w; x++) {
                                __m128 _p = _mm_loadu_ps(ptr);
                                __m128 _p1 = _mm_loadu_ps(ptr1 + x * 4);
                                __m128 _outp = op.func_pack4(_p, _p1);
                                _mm_storeu_ps(outptr, _outp);

                                ptr += 4;
                                outptr += 4;
                            }
                        }
                    }
                });

                return 0;
            }

            if (w1 != 1 && w == 1 && h1 == h && channels1 == channels) {
                // special type 7
                c = otter::empty({channels1, h1, w1}, dtype1);
                
                auto a_a = a.accessor<float, 3, 4>();
                auto b_a = b.accessor<float, 3, 4>();
                auto c_a = c.accessor<float, 3, 4>();

                otter::parallel_for(0, channels1, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        const float* ptr = a_a[q].data();
                        const float* ptr1 = b_a[q].data();
                        float* outptr = c_a[q].data();

                        for (int y = 0; y < h1; y++) {
                            __m128 _p = _mm_loadu_ps(ptr + y * 4);
                            for (int x = 0; x < w1; x++) {
                                __m128 _p1 = _mm_loadu_ps(ptr1);
                                __m128 _outp = op.func_pack4(_p, _p1);
                                _mm_storeu_ps(outptr, _outp);

                                ptr1 += 4;
                                outptr += 4;
                            }
                        }
                    }
                });

                return 0;
            }

            if (w1 == w && h1 != 1 && h == 1 && channels1 == channels)
            {
                // special type 8
                c = otter::empty({channels1, h1, w1}, dtype1);
                
                auto a_a = a.accessor<float, 3, 4>();
                auto b_a = b.accessor<float, 3, 4>();
                auto c_a = c.accessor<float, 3, 4>();

                otter::parallel_for(0, channels1, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        const float* ptr = a_a[q].data();
                        const float* ptr1 = b_a[q].data();
                        float* outptr = c_a[q].data();

                        for (int y = 0; y < h1; y++) {
                            for (int x = 0; x < w1; x++) {
                                __m128 _p = _mm_loadu_ps(ptr + x * 4);
                                __m128 _p1 = _mm_loadu_ps(ptr1);
                                __m128 _outp = op.func_pack4(_p, _p1);
                                _mm_storeu_ps(outptr, _outp);

                                ptr1 += 4;
                                outptr += 4;
                            }
                        }
                    }
                });

                return 0;
            }

            // type 19
            c = otter::empty({channels, h, w}, dtype);
            
            auto a_a = a.accessor<float, 3, 4>();
            auto b_a = b.accessor<float, 3, 4>();
            auto c_a = c.accessor<float, 3, 4>();

            otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                for (const auto q : otter::irange(begin, end)) {
                    const float* ptr = a_a[q].data();
                    const float* ptr1 = b_a[q].data();
                    float* outptr = c_a[q].data();

                    for (int i = 0; i < size; i++) {
                        __m128 _p = _mm_loadu_ps(ptr);
                        __m128 _p1 = _mm_loadu_ps(ptr1);
                        __m128 _outp = op.func_pack4(_p, _p1);
                        _mm_storeu_ps(outptr, _outp);
                        ptr += 4;
                        ptr1 += 4;
                        outptr += 4;
                    }
                }
            });

            return 0;
        }

        c = otter::empty({channels, h, w}, dtype);

        if (b.dim() == 2) {
            
            auto a_a = a.accessor<float, 3, 4>();
            auto b_a = b.accessor<float, 2, 4>();
            auto c_a = c.accessor<float, 3, 4>();
            
            // type 18
            otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                for (const auto q : otter::irange(begin, end)) {
                    const float* ptr = a_a[q].data();
                    const float* ptr1 = b_a[q].data();
                    float* outptr = c_a[q].data();

                    for (int y = 0; y < h; y++) {
                        __m128 _b0 = _mm_loadu_ps(ptr1);
                        for (int x = 0; x < w; x++) {
                            __m128 _p = _mm_loadu_ps(ptr);
                            __m128 _outp = op.func_pack4(_p, _b0);
                            _mm_storeu_ps(outptr, _outp);
                            ptr += 4;
                            outptr += 4;
                        }

                        ptr1 += 4;
                    }
                }
            });

            return 0;
        }

        if (b.dim() == 1) {
            if (b.size(0) == 1 && elempack1 == 1) {
                // type 16
                auto a_a = a.accessor<float, 3, 4>();
                auto b_a = b.accessor<float, 1>();
                auto c_a = c.accessor<float, 3, 4>();
                
                __m128 _b0 = _mm_set1_ps(b_a[0]);
                otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        const float* ptr = a_a[q].data();
                        float* outptr = c_a[q].data();

                        for (int i = 0; i < size; i++) {
                            __m128 _p = _mm_loadu_ps(ptr);
                            __m128 _outp = op.func_pack4(_p, _b0);
                            _mm_storeu_ps(outptr, _outp);
                            ptr += 4;
                            outptr += 4;
                        }
                    }
                });

                return 0;
            }
            
            auto a_a = a.accessor<float, 3, 4>();
            auto b_a = b.accessor<float, 1, 4>();
            auto c_a = c.accessor<float, 3, 4>();

            // type 17
            otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                for (const auto q : otter::irange(begin, end)) {
                    const float* ptr = a_a[q].data();
                    __m128 _b0 = _mm_loadu_ps((const float*)b_a.data() + q * 4);
                    float* outptr = c_a[q].data();

                    for (int i = 0; i < size; i++)
                    {
                        __m128 _p = _mm_loadu_ps(ptr);
                        __m128 _outp = op.func_pack4(_p, _b0);
                        _mm_storeu_ps(outptr, _outp);
                        ptr += 4;
                        outptr += 4;
                    }
                }
            });

            return 0;
        }
    } else if (a.dim() == 2) {
        int w = a.size(1);
        int h = a.size(0);
        int size = w * h;
        
        if (b.dim() == 3) {
            int w1 = b.size(2);
            int h1 = b.size(1);
            int channels1 = b.size(0);
            
            // type 14
            c = otter::empty({channels1, h1, w1}, dtype1);
            
            auto a_a = a.accessor<float, 2, 4>();
            auto b_a = b.accessor<float, 3, 4>();
            auto c_a = c.accessor<float, 3, 4>();

            otter::parallel_for(0, channels1, 0, [&](int64_t begin, int64_t end) {
                for (const auto q : otter::irange(begin, end)) {
                    const float* ptr = a_a[q].data();
                    const float* ptr1 = b_a[q].data();
                    float* outptr = c_a[q].data();

                    for (int y = 0; y < h1; y++) {
                        __m128 _a0 = _mm_loadu_ps(ptr);
                        for (int x = 0; x < w1; x++) {
                            __m128 _p1 = _mm_loadu_ps(ptr1);
                            __m128 _outp = op.func_pack4(_a0, _p1);
                            _mm_storeu_ps(outptr, _outp);
                            ptr1 += 4;
                            outptr += 4;
                        }

                        ptr += 4;
                    }
                }
            });

            return 0;
        }

        c = otter::empty({h, w}, dtype);

        if (b.dim() == 2) {
            // type 13
            auto a_a = a.accessor<float, 2, 4>();
            auto b_a = b.accessor<float, 2, 4>();
            auto c_a = c.accessor<float, 2, 4>();
            
            const float* ptr = a_a.data();
            const float* ptr1 = b_a.data();
            float* outptr = c_a.data();
            for (int i = 0; i < size; i++) {
                __m128 _p = _mm_loadu_ps(ptr);
                __m128 _p1 = _mm_loadu_ps(ptr1);
                __m128 _outp = op.func_pack4(_p, _p1);
                _mm_storeu_ps(outptr, _outp);
                ptr += 4;
                ptr1 += 4;
                outptr += 4;
            }

            return 0;
        }

        if (b.dim() == 1) {
            c = otter::empty({h, w}, dtype);

            if (b.size(0) == 1 && elempack1 == 1) {
                auto a_a = a.accessor<float, 2, 4>();
                auto b_a = b.accessor<float, 1>();
                auto c_a = c.accessor<float, 2, 4>();
                
                // type 11
                __m128 _b0 = _mm_set1_ps(b_a[0]);
                const float* ptr = a_a.data();
                float* outptr = c_a.data();
                for (int i = 0; i < size; i++) {
                    __m128 _p = _mm_loadu_ps(ptr);
                    __m128 _outp = op.func_pack4(_p, _b0);
                    _mm_storeu_ps(outptr, _outp);
                    ptr += 4;
                    outptr += 4;
                }

                return 0;
            }
            
            auto a_a = a.accessor<float, 2, 4>();
            auto b_a = b.accessor<float, 1, 4>();
            auto c_a = c.accessor<float, 2, 4>();

            // type 12
            const float* ptr = a_a.data();
            const float* ptr1 = b_a.data();
            float* outptr = c_a.data();

            for (int y = 0; y < h; y++) {
                __m128 _b0 = _mm_loadu_ps(ptr1);
                for (int x = 0; x < w; x++) {
                    __m128 _p = _mm_loadu_ps(ptr);
                    __m128 _outp = op.func_pack4(_p, _b0);
                    _mm_storeu_ps(outptr, _outp);
                    ptr += 4;
                    outptr += 4;
                }

                ptr1 += 4;
            }

            return 0;
        }
    } else if (a.dim() == 1) {
        int w = a.size(0);
        
        if (a.size(0) == 1 && elempack == 1) {
            if (b.dim() == 3) {
                int w1 = b.size(2);
                int h1 = b.size(1);
                int channels1 = b.size(0);
                int size1 = w1 * h1;
                
                // type 4
                c = otter::empty({channels1, h1, w1}, dtype1);
                
                auto a_a = a.accessor<float, 1>();
                auto b_a = b.accessor<float, 3, 4>();
                auto c_a = c.accessor<float, 3, 4>();

                __m128 _a0 = _mm_set1_ps(a_a[0]);
                otter::parallel_for(0, channels1, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        const float* ptr1 = b_a[q].data();
                        float* outptr = c_a[q].data();

                        for (int i = 0; i < size1; i++) {
                            __m128 _p1 = _mm_loadu_ps(ptr1);
                            __m128 _outp = op.func_pack4(_a0, _p1);
                            _mm_storeu_ps(outptr, _outp);
                            ptr1 += 4;
                            outptr += 4;
                        }
                    }
                });

                return 0;
            }

            if (b.dim() == 2)
            {
                int w1 = b.size(1);
                int h1 = b.size(0);
                int size1 = w1 * h1;
                
                // type 3
                c = otter::empty({h1, w1}, dtype1);
                
                auto a_a = a.accessor<float, 1>();
                auto b_a = b.accessor<float, 2, 4>();
                auto c_a = c.accessor<float, 2, 4>();

                __m128 _a0 = _mm_set1_ps(a_a[0]);
                const float* ptr1 = b_a.data();
                float* outptr = c_a.data();
                for (int i = 0; i < size1; i++) {
                    __m128 _p1 = _mm_loadu_ps(ptr1);
                    __m128 _outp = op.func_pack4(_a0, _p1);
                    _mm_storeu_ps(outptr, _outp);
                    ptr1 += 4;
                    outptr += 4;
                }

                return 0;
            }

            if (b.dim() == 1) {
                int w1 = b.size(0);
                
                // type 2
                c = otter::empty({w1}, dtype1);
                
                auto a_a = a.accessor<float, 1>();
                auto b_a = b.accessor<float, 1, 4>();
                auto c_a = c.accessor<float, 2, 4>();

                __m128 _a0 = _mm_set1_ps(a_a[0]);
                const float* ptr1 = b_a.data();
                float* outptr = c_a.data();
                for (int i = 0; i < w1; i++) {
                    __m128 _p1 = _mm_loadu_ps(ptr1);
                    __m128 _outp = op.func_pack4(_a0, _p1);
                    _mm_storeu_ps(outptr, _outp);
                    ptr1 += 4;
                    outptr += 4;
                }

                return 0;
            }
        }

        if (b.dim() == 3) {
            int w1 = b.size(2);
            int h1 = b.size(1);
            int channels1 = b.size(0);
            int size1 = w1 * h1;
            
            // type 9
            c = otter::empty({channels1, h1, w1}, dtype1);
            
            auto a_a = a.accessor<float, 1, 4>();
            auto b_a = b.accessor<float, 3, 4>();
            auto c_a = c.accessor<float, 3, 4>();

            otter::parallel_for(0, channels1, 0, [&](int64_t begin, int64_t end) {
                for (const auto q : otter::irange(begin, end))
                {
                    __m128 _a0 = _mm_loadu_ps((const float*)a_a.data() + q * 4);
                    const float* ptr1 = b_a[q].data();
                    float* outptr = c_a[q].data();

                    for (int i = 0; i < size1; i++) {
                        __m128 _p1 = _mm_loadu_ps(ptr1);
                        __m128 _outp = op.func_pack4(_a0, _p1);
                        _mm_storeu_ps(outptr, _outp);
                        ptr1 += 4;
                        outptr += 4;
                    }
                }
            });

            return 0;
        }

        if (b.dim() == 2) {
            int w1 = b.size(1);
            int h1 = b.size(0);
            
            // type 8
            c = otter::empty({h1, w1}, dtype1);
            
            auto a_a = a.accessor<float, 1, 4>();
            auto b_a = b.accessor<float, 2, 4>();
            auto c_a = c.accessor<float, 2, 4>();

            const float* ptr = a_a.data();
            const float* ptr1 = b_a.data();
            float* outptr = c_a.data();

            for (int y = 0; y < h1; y++) {
                __m128 _a0 = _mm_loadu_ps(ptr);
                for (int x = 0; x < w1; x++) {
                    __m128 _p1 = _mm_loadu_ps(ptr1);
                    __m128 _outp = op.func_pack4(_a0, _p1);
                    _mm_storeu_ps(outptr, _outp);
                    ptr1 += 4;
                    outptr += 4;
                }

                ptr += 4;
            }

            return 0;
        }

        if (b.dim() == 1) {
            c = otter::empty({w}, dtype);

            if (b.size(0) == 1 && elempack1 == 1) {
                auto a_a = a.accessor<float, 1, 4>();
                auto b_a = b.accessor<float, 1>();
                auto c_a = c.accessor<float, 1, 4>();
                
                // type 6
                __m128 _b0 = _mm_set1_ps(b_a[0]);
                const float* ptr = a_a.data();
                float* outptr = c_a.data();
                for (int i = 0; i < w; i++)
                {
                    __m128 _p = _mm_loadu_ps(ptr);
                    __m128 _outp = op.func_pack4(_p, _b0);
                    _mm_storeu_ps(outptr, _outp);
                    ptr += 4;
                    outptr += 4;
                }

                return 0;
            }
            
            auto a_a = a.accessor<float, 1, 4>();
            auto b_a = b.accessor<float, 1, 4>();
            auto c_a = c.accessor<float, 1, 4>();

            // type 7
            const float* ptr = a_a.data();
            const float* ptr1 = b_a.data();
            float* outptr = c_a.data();
            for (int i = 0; i < w; i++) {
                __m128 _p = _mm_loadu_ps(ptr);
                __m128 _p1 = _mm_loadu_ps(ptr1);
                __m128 _outp = op.func_pack4(_p, _p1);
                _mm_storeu_ps(outptr, _outp);
                ptr += 4;
                ptr1 += 4;
                outptr += 4;
            }
        }
    }

    return 0;
}

#if __AVX__
template<typename Op>
static int binary_op_pack8(const Tensor& a, const Tensor& b, Tensor& c)
{
    Op op;

    int elempack = a.elempack();
    ScalarType dtype = a.scalar_type();

    int elempack1 = b.elempack();
    ScalarType dtype1 = b.scalar_type();

    if (a.dim() == 3) {
        int w = a.size(2);
        int h = a.size(1);
        int channels = a.size(0);
        int size = w * h;
        
        if (b.dim() == 3) {
            int w1 = b.size(2);
            int h1 = b.size(1);
            int channels1 = b.size(0);
            int size1 = w1 * h1;
            
            if (w1 == 1 && h1 == 1 && channels1 == channels)
            {
                // special type 1
                c = otter::empty({channels, h, w}, dtype);
                
                auto a_a = a.accessor<float, 3, 8>();
                auto b_a = b.accessor<float, 3, 8>();
                auto c_a = c.accessor<float, 3, 8>();
                
                otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        const float* ptr = a_a[q].data();
                        float* outptr = c_a[q].data();
                        const float* b0 = b_a[q].data();
                        __m256 _b0 = _mm256_loadu_ps(b0);
                        for (int i = 0; i < size; i++) {
                            __m256 _p = _mm256_loadu_ps(ptr);
                            __m256 _outp = op.func_pack8(_p, _b0);
                            _mm256_storeu_ps(outptr, _outp);
                            ptr += 8;
                            outptr += 8;
                        }
                    }
                });

                return 0;
            }

            if (w1 == w && h1 == h && channels1 == 1 && elempack1 == 1) {
                // special type 2
                c = otter::empty({channels, h, w}, dtype);
                
                auto a_a = a.accessor<float, 3, 8>();
                auto b_a = b.accessor<float, 3>();
                auto c_a = c.accessor<float, 3, 8>();
                
                otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        const float* ptr = a_a[q].data();
                        const float* ptr1 = b_a.data();
                        float* outptr = c_a[q].data();
                        for (int i = 0; i < size; i++) {
                            __m256 _p = _mm256_loadu_ps(ptr);
                            __m256 _p1 = _mm256_broadcast_ss(ptr1);
                            __m256 _outp = op.func_pack8(_p, _p1);
                            _mm256_storeu_ps(outptr, _outp);
                            ptr += 8;
                            ptr1 += 1;
                            outptr += 8;
                        }
                    }
                });

                return 0;
            }

            if (w == 1 && h == 1 && channels1 == channels)
            {
                // special type 3
                c = otter::empty({channels1, h1, w1}, dtype1);
                
                auto a_a = a.accessor<float, 3, 8>();
                auto b_a = b.accessor<float, 3, 8>();
                auto c_a = c.accessor<float, 3, 8>();
                
                otter::parallel_for(0, channels1, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        const float* a0 = a_a[q].data();
                        float* outptr = c_a[q].data();
                        const float* ptr1 = b_a[q].data();
                        __m256 _a0 = _mm256_loadu_ps(a0);
                        for (int i = 0; i < size1; i++) {
                            __m256 _p1 = _mm256_loadu_ps(ptr1);
                            __m256 _outp = op.func_pack8(_a0, _p1);
                            _mm256_storeu_ps(outptr, _outp);
                            ptr1 += 8;
                            outptr += 8;
                        }
                    }
                });

                return 0;
            }

            if (w1 == w && h1 == h && channels == 1 && elempack == 1)
            {
                // special type 4
                c = otter::empty({channels1, h1, w1}, dtype1);
                
                auto a_a = a.accessor<float, 3>();
                auto b_a = b.accessor<float, 3, 8>();
                auto c_a = c.accessor<float, 3, 8>();
                
                otter::parallel_for(0, channels1, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        const float* ptr = a_a.data();
                        const float* ptr1 = b_a[q].data();
                        float* outptr = c_a[q].data();
                        for (int i = 0; i < size1; i++) {
                            __m256 _p = _mm256_broadcast_ss(ptr);
                            __m256 _p1 = _mm256_loadu_ps(ptr1);
                            __m256 _outp = op.func_pack8(_p, _p1);
                            _mm256_storeu_ps(outptr, _outp);
                            ptr += 1;
                            ptr1 += 8;
                            outptr += 8;
                        }
                    }
                });

                return 0;
            }

            if (w != 1 && w1 == 1 && h1 == h && channels1 == channels)
            {
                // special type 5
                c = otter::empty({channels, h, w}, dtype);
                
                auto a_a = a.accessor<float, 3, 8>();
                auto b_a = b.accessor<float, 3, 8>();
                auto c_a = c.accessor<float, 3, 8>();
                
                otter::parallel_for(0, channels1, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        const float* ptr = a_a[q].data();
                        const float* ptr1 = b_a[q].data();
                        float* outptr = c_a[q].data();

                        for (int y = 0; y < h; y++) {
                            __m256 _p1 = _mm256_loadu_ps(ptr1 + y * 8);
                            for (int x = 0; x < w; x++) {
                                __m256 _p = _mm256_loadu_ps(ptr);
                                __m256 _outp = op.func_pack8(_p, _p1);
                                _mm256_storeu_ps(outptr, _outp);

                                ptr += 8;
                                outptr += 8;
                            }
                        }
                    }
                });

                return 0;
            }

            if (w1 == w && h != 1 && h1 == 1 && channels1 == channels) {
                // special type 6
                c = otter::empty({channels, h, w}, dtype);
                
                auto a_a = a.accessor<float, 3, 8>();
                auto b_a = b.accessor<float, 3, 8>();
                auto c_a = c.accessor<float, 3, 8>();
                
                otter::parallel_for(0, channels1, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        const float* ptr = a_a[q].data();
                        const float* ptr1 = b_a[q].data();
                        float* outptr = c_a[q].data();

                        for (int y = 0; y < h; y++) {
                            for (int x = 0; x < w; x++) {
                                __m256 _p = _mm256_loadu_ps(ptr);
                                __m256 _p1 = _mm256_loadu_ps(ptr1 + x * 8);
                                __m256 _outp = op.func_pack8(_p, _p1);
                                _mm256_storeu_ps(outptr, _outp);

                                ptr += 8;
                                outptr += 8;
                            }
                        }
                    }
                });

                return 0;
            }

            if (w1 != 1 && w == 1 && h1 == h && channels1 == channels) {
                // special type 7
                c = otter::empty({channels1, h1, w1}, dtype1);
                
                auto a_a = a.accessor<float, 3, 8>();
                auto b_a = b.accessor<float, 3, 8>();
                auto c_a = c.accessor<float, 3, 8>();
                
                otter::parallel_for(0, channels1, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        const float* ptr = a_a[q].data();
                        const float* ptr1 = b_a[q].data();
                        float* outptr = c_a[q].data();

                        for (int y = 0; y < h1; y++) {
                            __m256 _p = _mm256_loadu_ps(ptr + y * 8);
                            for (int x = 0; x < w1; x++) {
                                __m256 _p1 = _mm256_loadu_ps(ptr1);
                                __m256 _outp = op.func_pack8(_p, _p1);
                                _mm256_storeu_ps(outptr, _outp);

                                ptr1 += 8;
                                outptr += 8;
                            }
                        }
                    }
                });

                return 0;
            }

            if (w1 == w && h1 != 1 && h == 1 && channels1 == channels) {
                // special type 8
                c = otter::empty({channels1, h1, w1}, dtype1);
                
                auto a_a = a.accessor<float, 3, 8>();
                auto b_a = b.accessor<float, 3, 8>();
                auto c_a = c.accessor<float, 3, 8>();
                
                otter::parallel_for(0, channels1, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        const float* ptr = a_a[q].data();
                        const float* ptr1 = b_a[q].data();
                        float* outptr = c_a[q].data();

                        for (int y = 0; y < h1; y++) {
                            for (int x = 0; x < w1; x++) {
                                __m256 _p = _mm256_loadu_ps(ptr + x * 8);
                                __m256 _p1 = _mm256_loadu_ps(ptr1);
                                __m256 _outp = op.func_pack8(_p, _p1);
                                _mm256_storeu_ps(outptr, _outp);

                                ptr1 += 8;
                                outptr += 8;
                            }
                        }
                    }
                });

                return 0;
            }

            // type 19
            c = otter::empty({channels, h, w}, dtype);
            
            auto a_a = a.accessor<float, 3, 8>();
            auto b_a = b.accessor<float, 3, 8>();
            auto c_a = c.accessor<float, 3, 8>();
            
            size *= elempack;

            otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                for (const auto q : otter::irange(begin, end)) {
                    const float* ptr = a_a[q].data();
                    const float* ptr1 = b_a[q].data();
                    float* outptr = c_a[q].data();

                    int i = 0;
                    
            #if __SSE2__
            #if __AVX__
                    for (; i + 7 < size; i += 8)
                    {
                        __m256 _p = _mm256_loadu_ps(ptr);
                        __m256 _p1 = _mm256_loadu_ps(ptr1);
                        __m256 _outp = op.func_pack8(_p, _p1);
                        _mm256_storeu_ps(outptr, _outp);
                        ptr += 8;
                        ptr1 += 8;
                        outptr += 8;
                    }
            #endif // __AVX__
                    for (; i + 3 < size; i += 4)
                    {
                        __m128 _p = _mm_load_ps(ptr);
                        __m128 _p1 = _mm_load_ps(ptr1);
                        __m128 _outp = op.func_pack4(_p, _p1);
                        _mm_store_ps(outptr, _outp);
                        ptr += 4;
                        ptr1 += 4;
                        outptr += 4;
                    }
            #endif // __SSE2__
                    for (; i < size; i++)
                    {
                        *outptr = op.func(*ptr, *ptr1);
                        ptr += 1;
                        ptr1 += 1;
                        outptr += 1;
                    }
                }
            });
            
            return 0;
        }

        c = otter::empty({channels, h, w}, dtype);

        if (b.dim() == 2) {
            
            auto a_a = a.accessor<float, 3, 4>();
            auto b_a = b.accessor<float, 2, 4>();
            auto c_a = c.accessor<float, 3, 4>();
            
            // type 18
            otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                for (const auto q : otter::irange(begin, end)) {
                    const float* ptr = a_a[q].data();
                    const float* ptr1 = b_a[q].data();
                    float* outptr = c_a[q].data();

                    for (int y = 0; y < h; y++) {
                        __m256 _b0 = _mm256_loadu_ps(ptr1);
                        for (int x = 0; x < w; x++) {
                            __m256 _p = _mm256_loadu_ps(ptr);
                            __m256 _outp = op.func_pack8(_p, _b0);
                            _mm256_storeu_ps(outptr, _outp);
                            ptr += 8;
                            outptr += 8;
                        }

                        ptr1 += 8;
                    }
                }
            });

            return 0;
        }

        if (b.dim() == 1) {
            if (b.size(0) == 1 && elempack1 == 1) {
                // type 16
                auto a_a = a.accessor<float, 3, 8>();
                auto b_a = b.accessor<float, 1>();
                auto c_a = c.accessor<float, 3, 8>();
                
                const float b0 = b_a[0];
                
                size *= elempack;
                
                otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        const float* ptr = a_a[q].data();
                        float* outptr = c_a[q].data();

                        int i = 0;
                        __m256 _b0_avx = _mm256_set1_ps(b0);
                        for (; i + 7 < size; i += 8)
                        {
                            __m256 _p = _mm256_loadu_ps(ptr);
                            __m256 _outp = op.func_pack8(_p, _b0_avx);
                            _mm256_storeu_ps(outptr, _outp);
                            ptr += 8;
                            outptr += 8;
                        }
                        for (; i < size; i++)
                        {
                            *outptr = op.func(*ptr, b0);
                            ptr += 1;
                            outptr += 1;
                        }
                    }
                });

                return 0;
            }
            
            auto a_a = a.accessor<float, 3, 8>();
            auto b_a = b.accessor<float, 1, 8>();
            auto c_a = c.accessor<float, 3, 8>();
            
            otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                for (const auto q : otter::irange(begin, end)) {
                    const float* ptr = a_a[q].data();
                    __m256 _b0 = _mm256_loadu_ps((const float*)b_a.data() + q * 8);
                    float* outptr = c_a[q].data();

                    for (int i = 0; i < size; i++) {
                        __m256 _p = _mm256_loadu_ps(ptr);
                        __m256 _outp = op.func_pack8(_p, _b0);
                        _mm256_storeu_ps(outptr, _outp);
                        ptr += 8;
                        outptr += 8;
                    }
                }
            });

            return 0;
        }
    }
    else if (a.dim() == 2) {
        int w = a.size(1);
        int h = a.size(0);
        int size = w * h;
        
        if (b.dim() == 3) {
            int w1 = b.size(2);
            int h1 = b.size(1);
            int channels1 = b.size(0);
            
            // type 14
            c = otter::empty({channels1, h1, w1}, dtype1);
            
            auto a_a = a.accessor<float, 2, 8>();
            auto b_a = b.accessor<float, 3, 8>();
            auto c_a = c.accessor<float, 3, 8>();
            
            otter::parallel_for(0, channels1, 0, [&](int64_t begin, int64_t end) {
                for (const auto q : otter::irange(begin, end)) {
                    const float* ptr = a_a[q].data();
                    const float* ptr1 = b_a[q].data();
                    float* outptr = c_a[q].data();

                    for (int y = 0; y < h1; y++) {
                        __m256 _a0 = _mm256_loadu_ps(ptr);
                        for (int x = 0; x < w1; x++) {
                            __m256 _p1 = _mm256_loadu_ps(ptr1);
                            __m256 _outp = op.func_pack8(_a0, _p1);
                            _mm256_storeu_ps(outptr, _outp);
                            ptr1 += 8;
                            outptr += 8;
                        }

                        ptr += 8;
                    }
                }
            });

            return 0;
        }

        c = otter::empty({h, w}, dtype);

        if (b.dim() == 2)
        {
            // type 13
            auto a_a = a.accessor<float, 2, 8>();
            auto b_a = b.accessor<float, 2, 8>();
            auto c_a = c.accessor<float, 2, 8>();
            
            const float* ptr = a_a.data();
            const float* ptr1 = b_a.data();
            float* outptr = c_a.data();
            
            size *= elempack;
            
            int i = 0;
    #if __SSE2__
    #if __AVX__
            for (; i + 7 < size; i += 8)
            {
                __m256 _p = _mm256_loadu_ps(ptr);
                __m256 _p1 = _mm256_loadu_ps(ptr1);
                __m256 _outp = op.func_pack8(_p, _p1);
                _mm256_storeu_ps(outptr, _outp);
                ptr += 8;
                ptr1 += 8;
                outptr += 8;
            }
    #endif // __AVX__
            for (; i + 3 < size; i += 4)
            {
                __m128 _p = _mm_load_ps(ptr);
                __m128 _p1 = _mm_load_ps(ptr1);
                __m128 _outp = op.func_pack4(_p, _p1);
                _mm_store_ps(outptr, _outp);
                ptr += 4;
                ptr1 += 4;
                outptr += 4;
            }
    #endif // __SSE2__
            for (; i < size; i++)
            {
                *outptr = op.func(*ptr, *ptr1);
                ptr += 1;
                ptr1 += 1;
                outptr += 1;
            }

            return 0;
        }

        if (b.dim() == 1)
        {
            c = otter::empty({h, w}, dtype);

            if (b.size(0) == 1 && elempack1 == 1)
            {
                // type 11
                auto a_a = a.accessor<float, 2, 8>();
                auto b_a = b.accessor<float, 1>();
                auto c_a = c.accessor<float, 2, 8>();
                
                size *= elempack;

                const float* ptr = a_a.data();
                const float b0 = b_a[0];
                float* outptr = c_a.data();
                int i = 0;
        #if __SSE2__
        #if __AVX__
                __m256 _b0_avx = _mm256_set1_ps(b0);
                for (; i + 7 < size; i += 8)
                {
                    __m256 _p = _mm256_loadu_ps(ptr);
                    __m256 _outp = op.func_pack8(_p, _b0_avx);
                    _mm256_storeu_ps(outptr, _outp);
                    ptr += 8;
                    outptr += 8;
                }
        #endif // __AVX__
                __m128 _b0 = _mm_set1_ps(b0);
                for (; i + 3 < size; i += 4)
                {
                    __m128 _p = _mm_load_ps(ptr);
                    __m128 _outp = op.func_pack4(_p, _b0);
                    _mm_store_ps(outptr, _outp);
                    ptr += 4;
                    outptr += 4;
                }
        #endif // __SSE2__
                for (; i < size; i++)
                {
                    *outptr = op.func(*ptr, b0);
                    ptr += 1;
                    outptr += 1;
                }

                return 0;
            }

            auto a_a = a.accessor<float, 2, 4>();
            auto b_a = b.accessor<float, 1, 4>();
            auto c_a = c.accessor<float, 2, 4>();

            // type 12
            const float* ptr = a_a.data();
            const float* ptr1 = b_a.data();
            float* outptr = c_a.data();

            for (int y = 0; y < h; y++)
            {
                __m256 _b0 = _mm256_loadu_ps(ptr1);
                for (int x = 0; x < w; x++)
                {
                    __m256 _p = _mm256_loadu_ps(ptr);
                    __m256 _outp = op.func_pack8(_p, _b0);
                    _mm256_storeu_ps(outptr, _outp);
                    ptr += 8;
                    outptr += 8;
                }

                ptr1 += 8;
            }

            return 0;
        }
    }
    else if (a.dim() == 1) {
        int w = a.size(0);
        
        if (a.size(0) == 1 && elempack == 1) {
            if (b.dim() == 3) {
                int w1 = b.size(2);
                int h1 = b.size(1);
                int channels1 = b.size(0);
                int size1 = w1 * h1 * elempack1;
                
                // type 4
                c = otter::empty({channels1, h1, w1}, dtype1);
                
                auto a_a = a.accessor<float, 1>();
                auto b_a = b.accessor<float, 3, 8>();
                auto c_a = c.accessor<float, 3, 8>();
                
                const float a0 = a_a[0];

                otter::parallel_for(0, channels1, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        const float* ptr1 = b_a[q].data();
                        float* outptr = c_a[q].data();

                        int i = 0;
                #if __SSE2__
                #if __AVX__
                        __m256 _a0_avx = _mm256_set1_ps(a0);
                        for (; i + 7 < size1; i += 8)
                        {
                            __m256 _p = _mm256_loadu_ps(ptr1);
                            __m256 _outp = op.func_pack8(_a0_avx, _p);
                            _mm256_storeu_ps(outptr, _outp);
                            ptr1 += 8;
                            outptr += 8;
                        }
                #endif // __AVX__
                        __m128 _a0 = _mm_set1_ps(a0);
                        for (; i + 3 < size1; i += 4)
                        {
                            __m128 _p = _mm_load_ps(ptr1);
                            __m128 _outp = op.func_pack4(_a0, _p);
                            _mm_store_ps(outptr, _outp);
                            ptr1 += 4;
                            outptr += 4;
                        }
                #endif // __SSE2__
                        for (; i < size1; i++)
                        {
                            *outptr = op.func(a0, *ptr1);
                            ptr1 += 1;
                            outptr += 1;
                        }
                    }
                });

                return 0;
            }

            if (b.dim() == 2)
            {
                int w1 = b.size(1);
                int h1 = b.size(0);
                int size1 = w1 * h1 * elempack1;
                
                // type 3
                c = otter::empty({h1, w1}, dtype1);
                
                auto a_a = a.accessor<float, 1>();
                auto b_a = b.accessor<float, 2, 8>();
                auto c_a = c.accessor<float, 2, 8>();

                const float* ptr1 = b_a.data();
                float* outptr = c_a.data();
                
                const float a0 = a_a[0];
                
                int i = 0;
        #if __SSE2__
        #if __AVX__
                __m256 _a0_avx = _mm256_set1_ps(a0);
                for (; i + 7 < size1; i += 8)
                {
                    __m256 _p = _mm256_loadu_ps(ptr1);
                    __m256 _outp = op.func_pack8(_a0_avx, _p);
                    _mm256_storeu_ps(outptr, _outp);
                    ptr1 += 8;
                    outptr += 8;
                }
        #endif // __AVX__
                __m128 _a0 = _mm_set1_ps(a0);
                for (; i + 3 < size1; i += 4)
                {
                    __m128 _p = _mm_load_ps(ptr1);
                    __m128 _outp = op.func_pack4(_a0, _p);
                    _mm_store_ps(outptr, _outp);
                    ptr1 += 4;
                    outptr += 4;
                }
        #endif // __SSE2__
                for (; i < size1; i++)
                {
                    *outptr = op.func(a0, *ptr1);
                    ptr1 += 1;
                    outptr += 1;
                }

                return 0;
            }

            if (b.dim() == 1) {
                int w1 = b.size(0);
                int size1 = w1 * elempack1;
                
                // type 2
                c = otter::empty({w1}, dtype1);
                
                auto a_a = a.accessor<float, 1>();
                auto b_a = b.accessor<float, 1, 8>();
                auto c_a = c.accessor<float, 2, 8>();

                const float* ptr1 = b_a.data();
                float* outptr = c_a.data();
                
                const float a0 = a_a[0];
                
                int i = 0;
        #if __SSE2__
        #if __AVX__
                __m256 _a0_avx = _mm256_set1_ps(a0);
                for (; i + 7 < size1; i += 8)
                {
                    __m256 _p = _mm256_loadu_ps(ptr1);
                    __m256 _outp = op.func_pack8(_a0_avx, _p);
                    _mm256_storeu_ps(outptr, _outp);
                    ptr1 += 8;
                    outptr += 8;
                }
        #endif // __AVX__
                __m128 _a0 = _mm_set1_ps(a0);
                for (; i + 3 < size1; i += 4)
                {
                    __m128 _p = _mm_load_ps(ptr1);
                    __m128 _outp = op.func_pack4(_a0, _p);
                    _mm_store_ps(outptr, _outp);
                    ptr1 += 4;
                    outptr += 4;
                }
        #endif // __SSE2__
                for (; i < size1; i++)
                {
                    *outptr = op.func(a0, *ptr1);
                    ptr1 += 1;
                    outptr += 1;
                }

                return 0;
            }
        }

        if (b.dim() == 3) {
            int w1 = b.size(2);
            int h1 = b.size(1);
            int channels1 = b.size(0);
            int size1 = w1 * h1;
            
            // type 9
            c = otter::empty({channels1, h1, w1}, dtype1);
            
            auto a_a = a.accessor<float, 1, 8>();
            auto b_a = b.accessor<float, 3, 8>();
            auto c_a = c.accessor<float, 3, 8>();
            
            otter::parallel_for(0, channels1, 0, [&](int64_t begin, int64_t end) {
                for (const auto q : otter::irange(begin, end))
                {
                    __m256 _a0 = _mm256_loadu_ps((const float*)a_a.data() + q * 8);
                    const float* ptr1 = b_a[q].data();
                    float* outptr = c_a[q].data();

                    for (int i = 0; i < size1; i++) {
                        __m256 _p1 = _mm256_loadu_ps(ptr1);
                        __m256 _outp = op.func_pack8(_a0, _p1);
                        _mm256_storeu_ps(outptr, _outp);
                        ptr1 += 8;
                        outptr += 8;
                    }
                }
            });

            return 0;
        }

        if (b.dim() == 2) {
            int w1 = b.size(1);
            int h1 = b.size(0);
            
            // type 8
            c = otter::empty({h1, w1}, dtype1);

            auto a_a = a.accessor<float, 1, 4>();
            auto b_a = b.accessor<float, 2, 4>();
            auto c_a = c.accessor<float, 2, 4>();

            const float* ptr = a_a.data();
            const float* ptr1 = b_a.data();
            float* outptr = c_a.data();

            for (int y = 0; y < h1; y++) {
                __m256 _a0 = _mm256_loadu_ps(ptr);
                for (int x = 0; x < w1; x++) {
                    __m256 _p1 = _mm256_loadu_ps(ptr1);
                    __m256 _outp = op.func_pack8(_a0, _p1);
                    _mm256_storeu_ps(outptr, _outp);
                    ptr1 += 8;
                    outptr += 8;
                }

                ptr += 8;
            }

            return 0;
        }

        if (b.dim() == 1) {
            c = otter::empty({w}, dtype);

            if (b.size(0) == 1 && elempack1 == 1) {
                auto a_a = a.accessor<float, 1, 8>();
                auto b_a = b.accessor<float, 1>();
                auto c_a = c.accessor<float, 1, 8>();
                
                int size = w * elempack;
                
                // type 6
                const float* ptr = a_a.data();
                float* outptr = c_a.data();
                
                const float b0 = b_a[0];
                
                int i = 0;
        #if __SSE2__
        #if __AVX__
                __m256 _b0_avx = _mm256_set1_ps(b0);
                for (; i + 7 < size; i += 8)
                {
                    __m256 _p = _mm256_loadu_ps(ptr);
                    __m256 _outp = op.func_pack8(_p, _b0_avx);
                    _mm256_storeu_ps(outptr, _outp);
                    ptr += 8;
                    outptr += 8;
                }
        #endif // __AVX__
                __m128 _b0 = _mm_set1_ps(b0);
                for (; i + 3 < size; i += 4)
                {
                    __m128 _p = _mm_load_ps(ptr);
                    __m128 _outp = op.func_pack4(_p, _b0);
                    _mm_store_ps(outptr, _outp);
                    ptr += 4;
                    outptr += 4;
                }
        #endif // __SSE2__
                for (; i < size; i++)
                {
                    *outptr = op.func(*ptr, b0);
                    ptr += 1;
                    outptr += 1;
                }

                return 0;
            }

            auto a_a = a.accessor<float, 1, 8>();
            auto b_a = b.accessor<float, 1, 8>();
            auto c_a = c.accessor<float, 1, 8>();
            
            int size = w * elempack;

            // type 7
            const float* ptr = a_a.data();
            const float* ptr1 = b_a.data();
            float* outptr = c_a.data();
            int i = 0;
    #if __SSE2__
    #if __AVX__
            for (; i + 7 < size; i += 8)
            {
                __m256 _p = _mm256_loadu_ps(ptr);
                __m256 _p1 = _mm256_loadu_ps(ptr1);
                __m256 _outp = op.func_pack8(_p, _p1);
                _mm256_storeu_ps(outptr, _outp);
                ptr += 8;
                ptr1 += 8;
                outptr += 8;
            }
    #endif // __AVX__
            for (; i + 3 < size; i += 4)
            {
                __m128 _p = _mm_load_ps(ptr);
                __m128 _p1 = _mm_load_ps(ptr1);
                __m128 _outp = op.func_pack4(_p, _p1);
                _mm_store_ps(outptr, _outp);
                ptr += 4;
                ptr1 += 4;
                outptr += 4;
            }
    #endif // __SSE2__
            for (; i < size; i++)
            {
                *outptr = op.func(*ptr, *ptr1);
                ptr += 1;
                ptr1 += 1;
                outptr += 1;
            }
        }
    }

    return 0;
}
#endif // __AVX__

namespace BinaryOp_x86_functor {

struct binary_op_add
{
    float func(const float& x, const float& y) const
    {
        return x + y;
    }
#if __SSE2__
    __m128 func_pack4(const __m128& x, const __m128& y) const
    {
        return _mm_add_ps(x, y);
    }
#if __AVX__
    __m256 func_pack8(const __m256& x, const __m256& y) const
    {
        return _mm256_add_ps(x, y);
    }
#endif // __AVX__
#endif // __SSE2__
};

struct binary_op_sub
{
    float func(const float& x, const float& y) const
    {
        return x - y;
    }
#if __SSE2__
    __m128 func_pack4(const __m128& x, const __m128& y) const
    {
        return _mm_sub_ps(x, y);
    }
#if __AVX__
    __m256 func_pack8(const __m256& x, const __m256& y) const
    {
        return _mm256_sub_ps(x, y);
    }
#endif // __AVX__
#endif // __SSE2__
};

struct binary_op_mul
{
    float func(const float& x, const float& y) const
    {
        return x * y;
    }
#if __SSE2__
    __m128 func_pack4(const __m128& x, const __m128& y) const
    {
        return _mm_mul_ps(x, y);
    }
#if __AVX__
    __m256 func_pack8(const __m256& x, const __m256& y) const
    {
        return _mm256_mul_ps(x, y);
    }
#endif // __AVX__
#endif // __SSE2__
};

struct binary_op_div
{
    float func(const float& x, const float& y) const
    {
        return x / y;
    }
#if __SSE2__
    __m128 func_pack4(const __m128& x, const __m128& y) const
    {
        return _mm_div_ps(x, y);
    }
#if __AVX__
    __m256 func_pack8(const __m256& x, const __m256& y) const
    {
        return _mm256_div_ps(x, y);
    }
#endif // __AVX__
#endif // __SSE2__
};

struct binary_op_max
{
    float func(const float& x, const float& y) const
    {
        return std::max(x, y);
    }
#if __SSE2__
    __m128 func_pack4(const __m128& x, const __m128& y) const
    {
        return _mm_max_ps(x, y);
    }
#if __AVX__
    __m256 func_pack8(const __m256& x, const __m256& y) const
    {
        return _mm256_max_ps(x, y);
    }
#endif // __AVX__
#endif // __SSE2__
};

struct binary_op_min
{
    float func(const float& x, const float& y) const
    {
        return std::min(x, y);
    }
#if __SSE2__
    __m128 func_pack4(const __m128& x, const __m128& y) const
    {
        return _mm_min_ps(x, y);
    }
#if __AVX__
    __m256 func_pack8(const __m256& x, const __m256& y) const
    {
        return _mm256_min_ps(x, y);
    }
#endif // __AVX__
#endif // __SSE2__
};

struct binary_op_pow
{
    float func(const float& x, const float& y) const
    {
        return (float)pow(x, y);
    }
#if __SSE2__
    __m128 func_pack4(const __m128& x, const __m128& y) const
    {
        return pow_ps(x, y);
    }
#if __AVX__
    __m256 func_pack8(const __m256& x, const __m256& y) const
    {
        return pow256_ps(x, y);
    }
#endif // __AVX__
#endif // __SSE2__
};

struct binary_op_rsub
{
    float func(const float& x, const float& y) const
    {
        return y - x;
    }
#if __SSE2__
    __m128 func_pack4(const __m128& x, const __m128& y) const
    {
        return _mm_sub_ps(y, x);
    }
#if __AVX__
    __m256 func_pack8(const __m256& x, const __m256& y) const
    {
        return _mm256_sub_ps(y, x);
    }
#endif // __AVX__
#endif // __SSE2__
};

struct binary_op_rdiv
{
    float func(const float& x, const float& y) const
    {
        return y / x;
    }
#if __SSE2__
    __m128 func_pack4(const __m128& x, const __m128& y) const
    {
        return _mm_div_ps(y, x);
    }
#if __AVX__
    __m256 func_pack8(const __m256& x, const __m256& y) const
    {
        return _mm256_div_ps(y, x);
    }
#endif // __AVX__
#endif // __SSE2__
};

} // namespace BinaryOp_x86_functor

#endif // __SSE2__

#if __ARM_NEON__
template<typename Op>
static int binary_op_pack4(const Tensor& a, const Tensor& b, Tensor& c) {
    Op op;

    int elempack = a.elempack();
    ScalarType dtype = a.scalar_type();

    int elempack1 = b.elempack();
    ScalarType dtype1 = b.scalar_type();

    if (a.dim() == 3) {
        int w = a.size(2);
        int h = a.size(1);
        int channels = a.size(0);
        int size = w * h;

        if (b.dim() == 3) {
            int w1 = b.size(2);
            int h1 = b.size(1);
            int channels1 = b.size(0);
            int size1 = w1 * h1;

            if (w1 == 1 && h1 == 1 && channels1 == channels) {
                // special type 1
                c = otter::empty({channels, h, w}, dtype);

                auto a_a = a.accessor<float, 3, 4>();
                auto b_a = b.accessor<float, 3, 4>();
                auto c_a = c.accessor<float, 3, 4>();

                otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        const float* ptr = a_a[q].data();
                        const float* b0 = b_a[q].data();
                        float* outptr = c_a[q].data();
                        float32x4_t _b0 = vld1q_f32(b0);
                        for (int i = 0; i < size; i++) {
                            float32x4_t _p = vld1q_f32(ptr);
                            float32x4_t _outp = op(_p, _b0);
                            vst1q_f32(outptr, _outp);
                            ptr += 4;
                            outptr += 4;
                        }
                    }
                });

                return 0;
            }

            if (w1 == w && h1 == h && channels1 == 1 && elempack1 == 1) {
                // special type 2
                c = otter::empty({channels, h, w}, dtype);

                auto a_a = a.accessor<float, 3, 4>();
                auto b_a = b.accessor<float, 3>();
                auto c_a = c.accessor<float, 3, 4>();

                otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        const float* ptr = a_a[q].data();
                        const float* ptr1 = b_a.data();
                        float* outptr = c_a[q].data();
                        for (int i = 0; i < size; i++) {
                            float32x4_t _p = vld1q_f32(ptr);
                            float32x4_t _p1 = vld1q_dup_f32(ptr1);
                            float32x4_t _outp = op(_p, _p1);
                            vst1q_f32(outptr, _outp);
                            ptr += 4;
                            ptr1 += 1;
                            outptr += 4;
                        }
                    }
                });

                return 0;
            }

            if (w == 1 && h == 1 && channels1 == channels) {
                // special type 3
                c = otter::empty({channels1, h1, w1}, dtype1);

                auto a_a = a.accessor<float, 3, 4>();
                auto b_a = b.accessor<float, 3, 4>();
                auto c_a = c.accessor<float, 3, 4>();

                otter::parallel_for(0, channels1, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        const float* a0 = a_a[q].data();
                        float* outptr = c_a[q].data();
                        const float* ptr1 = b_a[q].data();
                        float32x4_t _a0 = vld1q_f32(a0);
                        for (int i = 0; i < size1; i++) {
                            float32x4_t _p1 = vld1q_f32(ptr1);
                            float32x4_t _outp = op(_a0, _p1);
                            vst1q_f32(outptr, _outp);
                            ptr1 += 4;
                            outptr += 4;
                        }
                    }
                });

                return 0;
            }

            if (w1 == w && h1 == h && channels == 1 && elempack == 1) {
                // special type 4
                c = otter::empty({channels1, h1, w1}, dtype1);

                auto a_a = a.accessor<float, 3>();
                auto b_a = b.accessor<float, 3, 4>();
                auto c_a = c.accessor<float, 3, 4>();

                otter::parallel_for(0, channels1, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        const float* ptr = a_a.data();
                        const float* ptr1 = b_a[q].data();
                        float* outptr = c_a[q].data();
                        for (int i = 0; i < size1; i++) {
                            float32x4_t _p = vld1q_dup_f32(ptr);
                            float32x4_t _p1 = vld1q_f32(ptr1);
                            float32x4_t _outp = op(_p, _p1);
                            vst1q_f32(outptr, _outp);
                            ptr += 1;
                            ptr1 += 4;
                            outptr += 4;
                        }
                    }
                });

                return 0;
            }

            if (w != 1 && w1 == 1 && h1 == h && channels1 == channels) {
                // special type 5
                c = otter::empty({channels, h, w}, dtype);

                auto a_a = a.accessor<float, 3, 4>();
                auto b_a = b.accessor<float, 3, 4>();
                auto c_a = c.accessor<float, 3, 4>();

                otter::parallel_for(0, channels1, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        const float* ptr = a_a[q].data();
                        const float* ptr1 = b_a[q].data();
                        float* outptr = c_a[q].data();

                        for (int y = 0; y < h; y++) {
                            float32x4_t _p1 = vld1q_f32(ptr1 + y * 4);
                            for (int x = 0; x < w; x++)
                            {
                                float32x4_t _p = vld1q_f32(ptr);
                                float32x4_t _outp = op(_p, _p1);
                                vst1q_f32(outptr, _outp);

                                ptr += 4;
                                outptr += 4;
                            }
                        }
                    }
                });

                return 0;
            }

            if (w1 == w && h != 1 && h1 == 1 && channels1 == channels) {
                // special type 6
                c = otter::empty({channels, h, w}, dtype);

                auto a_a = a.accessor<float, 3, 4>();
                auto b_a = b.accessor<float, 3, 4>();
                auto c_a = c.accessor<float, 3, 4>();

                otter::parallel_for(0, channels1, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        const float* ptr = a_a[q].data();
                        const float* ptr1 = b_a[q].data();
                        float* outptr = c_a[q].data();

                        for (int y = 0; y < h; y++) {
                            for (int x = 0; x < w; x++) {
                                float32x4_t _p = vld1q_f32(ptr);
                                float32x4_t _p1 = vld1q_f32(ptr1 + x * 4);
                                float32x4_t _outp = op(_p, _p1);
                                vst1q_f32(outptr, _outp);

                                ptr += 4;
                                outptr += 4;
                            }
                        }
                    }
                });

                return 0;
            }

            if (w1 != 1 && w == 1 && h1 == h && channels1 == channels) {
                // special type 7
                c = otter::empty({channels1, h1, w1}, dtype1);

                auto a_a = a.accessor<float, 3, 4>();
                auto b_a = b.accessor<float, 3, 4>();
                auto c_a = c.accessor<float, 3, 4>();

                otter::parallel_for(0, channels1, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        const float* ptr = a_a[q].data();
                        const float* ptr1 = b_a[q].data();
                        float* outptr = c_a[q].data();

                        for (int y = 0; y < h1; y++) {
                            float32x4_t _p = vld1q_f32(ptr + y * 4);
                            for (int x = 0; x < w1; x++)
                            {
                                float32x4_t _p1 = vld1q_f32(ptr1);
                                float32x4_t _outp = op(_p, _p1);
                                vst1q_f32(outptr, _outp);

                                ptr1 += 4;
                                outptr += 4;
                            }
                        }
                    }
                });

                return 0;
            }

            if (w1 == w && h1 != 1 && h == 1 && channels1 == channels)
            {
                // special type 8
                c = otter::empty({channels1, h1, w1}, dtype1);

                auto a_a = a.accessor<float, 3, 4>();
                auto b_a = b.accessor<float, 3, 4>();
                auto c_a = c.accessor<float, 3, 4>();

                otter::parallel_for(0, channels1, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        const float* ptr = a_a[q].data();
                        const float* ptr1 = b_a[q].data();
                        float* outptr = c_a[q].data();

                        for (int y = 0; y < h1; y++) {
                            for (int x = 0; x < w1; x++)
                            {
                                float32x4_t _p = vld1q_f32(ptr + x * 4);
                                float32x4_t _p1 = vld1q_f32(ptr1);
                                float32x4_t _outp = op(_p, _p1);
                                vst1q_f32(outptr, _outp);

                                ptr1 += 4;
                                outptr += 4;
                            }
                        }
                    }
                });

                return 0;
            }

            // type 19
            c = otter::empty({channels, h, w}, dtype);

            auto a_a = a.accessor<float, 3, 4>();
            auto b_a = b.accessor<float, 3, 4>();
            auto c_a = c.accessor<float, 3, 4>();

            otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                for (const auto q : otter::irange(begin, end)) {
                    const float* ptr = a_a[q].data();
                    const float* ptr1 = b_a[q].data();
                    float* outptr = c_a[q].data();

                    for (int i = 0; i < size; i++) {
                        float32x4_t _p = vld1q_f32(ptr);
                        float32x4_t _p1 = vld1q_f32(ptr1);
                        float32x4_t _outp = op(_p, _p1);
                        vst1q_f32(outptr, _outp);
                        ptr += 4;
                        ptr1 += 4;
                        outptr += 4;
                    }
                }
            });

            return 0;
        }

        c = otter::empty({channels, h, w}, dtype);

        if (b.dim() == 2) {

            auto a_a = a.accessor<float, 3, 4>();
            auto b_a = b.accessor<float, 2, 4>();
            auto c_a = c.accessor<float, 3, 4>();

            // type 18
            otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                for (const auto q : otter::irange(begin, end)) {
                    const float* ptr = a_a[q].data();
                    const float* ptr1 = b_a[q].data();
                    float* outptr = c_a[q].data();

                    for (int y = 0; y < h; y++) {
                        float32x4_t _b0 = vld1q_f32(ptr1);
                        for (int x = 0; x < w; x++)
                        {
                            float32x4_t _p = vld1q_f32(ptr);
                            float32x4_t _outp = op(_p, _b0);
                            vst1q_f32(outptr, _outp);
                            ptr += 4;
                            outptr += 4;
                        }

                        ptr1 += 4;
                    }
                }
            });

            return 0;
        }

        if (b.dim() == 1) {
            if (b.size(0) == 1 && elempack1 == 1) {
                // type 16
                auto a_a = a.accessor<float, 3, 4>();
                auto b_a = b.accessor<float, 1>();
                auto c_a = c.accessor<float, 3, 4>();

                float32x4_t _b0 = vdupq_n_f32(b_a[0]);
                otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        const float* ptr = a_a[q].data();
                        float* outptr = c_a[q].data();

                        for (int i = 0; i < size; i++) {
                            float32x4_t _p = vld1q_f32(ptr);
                            float32x4_t _outp = op(_p, _b0);
                            vst1q_f32(outptr, _outp);
                            ptr += 4;
                            outptr += 4;
                        }
                    }
                });

                return 0;
            }

            auto a_a = a.accessor<float, 3, 4>();
            auto b_a = b.accessor<float, 1, 4>();
            auto c_a = c.accessor<float, 3, 4>();

            // type 17
            otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                for (const auto q : otter::irange(begin, end)) {
                    const float* ptr = a_a[q].data();
                    float32x4_t _b0 = vld1q_f32((const float*)b_a.data() + q * 4);
                    float* outptr = c_a[q].data();

                    for (int i = 0; i < size; i++)
                    {
                        float32x4_t _p = vld1q_f32(ptr);
                        float32x4_t _outp = op(_p, _b0);
                        vst1q_f32(outptr, _outp);
                        ptr += 4;
                        outptr += 4;
                    }
                }
            });

            return 0;
        }
    } else if (a.dim() == 2) {
        int w = a.size(1);
        int h = a.size(0);
        int size = w * h;

        if (b.dim() == 3) {
            int w1 = b.size(2);
            int h1 = b.size(1);
            int channels1 = b.size(0);

            // type 14
            c = otter::empty({channels1, h1, w1}, dtype1);

            auto a_a = a.accessor<float, 2, 4>();
            auto b_a = b.accessor<float, 3, 4>();
            auto c_a = c.accessor<float, 3, 4>();

            otter::parallel_for(0, channels1, 0, [&](int64_t begin, int64_t end) {
                for (const auto q : otter::irange(begin, end)) {
                    const float* ptr = a_a[q].data();
                    const float* ptr1 = b_a[q].data();
                    float* outptr = c_a[q].data();

                    for (int y = 0; y < h1; y++) {
                        float32x4_t _a0 = vld1q_f32(ptr);
                        for (int x = 0; x < w1; x++)
                        {
                            float32x4_t _p1 = vld1q_f32(ptr1);
                            float32x4_t _outp = op(_a0, _p1);
                            vst1q_f32(outptr, _outp);
                            ptr1 += 4;
                            outptr += 4;
                        }

                        ptr += 4;
                    }
                }
            });

            return 0;
        }

        c = otter::empty({h, w}, dtype);

        if (b.dim() == 2) {
            // type 13
            auto a_a = a.accessor<float, 2, 4>();
            auto b_a = b.accessor<float, 2, 4>();
            auto c_a = c.accessor<float, 2, 4>();

            const float* ptr = a_a.data();
            const float* ptr1 = b_a.data();
            float* outptr = c_a.data();
            for (int i = 0; i < size; i++) {
                float32x4_t _p = vld1q_f32(ptr);
                float32x4_t _p1 = vld1q_f32(ptr1);
                float32x4_t _outp = op(_p, _p1);
                vst1q_f32(outptr, _outp);
                ptr += 4;
                ptr1 += 4;
                outptr += 4;
            }

            return 0;
        }

        if (b.dim() == 1) {
            c = otter::empty({h, w}, dtype);

            if (b.size(0) == 1 && elempack1 == 1) {
                auto a_a = a.accessor<float, 2, 4>();
                auto b_a = b.accessor<float, 1>();
                auto c_a = c.accessor<float, 2, 4>();

                // type 11
                float32x4_t _b0 = vdupq_n_f32(b_a[0]);
                const float* ptr = a_a.data();
                float* outptr = c_a.data();
                for (int i = 0; i < size; i++)
                {
                    float32x4_t _p = vld1q_f32(ptr);
                    float32x4_t _outp = op(_p, _b0);
                    vst1q_f32(outptr, _outp);
                    ptr += 4;
                    outptr += 4;
                }

                return 0;
            }

            auto a_a = a.accessor<float, 2, 4>();
            auto b_a = b.accessor<float, 1, 4>();
            auto c_a = c.accessor<float, 2, 4>();

            // type 12
            const float* ptr = a_a.data();
            const float* ptr1 = b_a.data();
            float* outptr = c_a.data();

            for (int y = 0; y < h; y++) {
                float32x4_t _b0 = vld1q_f32(ptr1);
                for (int x = 0; x < w; x++)
                {
                    float32x4_t _p = vld1q_f32(ptr);
                    float32x4_t _outp = op(_p, _b0);
                    vst1q_f32(outptr, _outp);
                    ptr += 4;
                    outptr += 4;
                }

                ptr1 += 4;
            }

            return 0;
        }
    } else if (a.dim() == 1) {
        int w = a.size(0);

        if (a.size(0) == 1 && elempack == 1) {
            if (b.dim() == 3) {
                int w1 = b.size(2);
                int h1 = b.size(1);
                int channels1 = b.size(0);
                int size1 = w1 * h1;

                // type 4
                c = otter::empty({channels1, h1, w1}, dtype1);

                auto a_a = a.accessor<float, 1>();
                auto b_a = b.accessor<float, 3, 4>();
                auto c_a = c.accessor<float, 3, 4>();

                float32x4_t _a0 = vdupq_n_f32(a_a[0]);
                otter::parallel_for(0, channels1, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        const float* ptr1 = b_a[q].data();
                        float* outptr = c_a[q].data();

                        for (int i = 0; i < size1; i++) {
                            float32x4_t _p1 = vld1q_f32(ptr1);
                            float32x4_t _outp = op(_a0, _p1);
                            vst1q_f32(outptr, _outp);
                            ptr1 += 4;
                            outptr += 4;
                        }
                    }
                });

                return 0;
            }

            if (b.dim() == 2)
            {
                int w1 = b.size(1);
                int h1 = b.size(0);
                int size1 = w1 * h1;

                // type 3
                c = otter::empty({h1, w1}, dtype1);

                auto a_a = a.accessor<float, 1>();
                auto b_a = b.accessor<float, 2, 4>();
                auto c_a = c.accessor<float, 2, 4>();

                float32x4_t _a0 = vdupq_n_f32(a_a[0]);
                const float* ptr1 = b_a.data();
                float* outptr = c_a.data();
                for (int i = 0; i < size1; i++) {
                    float32x4_t _p1 = vld1q_f32(ptr1);
                    float32x4_t _outp = op(_a0, _p1);
                    vst1q_f32(outptr, _outp);
                    ptr1 += 4;
                    outptr += 4;
                }

                return 0;
            }

            if (b.dim() == 1) {
                int w1 = b.size(0);

                // type 2
                c = otter::empty({w1}, dtype1);

                auto a_a = a.accessor<float, 1>();
                auto b_a = b.accessor<float, 1, 4>();
                auto c_a = c.accessor<float, 2, 4>();

                float32x4_t _a0 = vdupq_n_f32(a_a[0]);
                const float* ptr1 = b_a.data();
                float* outptr = c_a.data();
                for (int i = 0; i < w1; i++) {
                    float32x4_t _p1 = vld1q_f32(ptr1);
                    float32x4_t _outp = op(_a0, _p1);
                    vst1q_f32(outptr, _outp);
                    ptr1 += 4;
                    outptr += 4;
                }

                return 0;
            }
        }

        if (b.dim() == 3) {
            int w1 = b.size(2);
            int h1 = b.size(1);
            int channels1 = b.size(0);
            int size1 = w1 * h1;

            // type 9
            c = otter::empty({channels1, h1, w1}, dtype1);

            auto a_a = a.accessor<float, 1, 4>();
            auto b_a = b.accessor<float, 3, 4>();
            auto c_a = c.accessor<float, 3, 4>();

            otter::parallel_for(0, channels1, 0, [&](int64_t begin, int64_t end) {
                for (const auto q : otter::irange(begin, end))
                {
                    float32x4_t _a0 = vld1q_f32((const float*)a_a.data() + q * 4);
                    const float* ptr1 = b_a[q].data();
                    float* outptr = c_a[q].data();

                    for (int i = 0; i < size1; i++) {
                        float32x4_t _p1 = vld1q_f32(ptr1);
                        float32x4_t _outp = op(_a0, _p1);
                        vst1q_f32(outptr, _outp);
                        ptr1 += 4;
                        outptr += 4;
                    }
                }
            });

            return 0;
        }

        if (b.dim() == 2) {
            int w1 = b.size(1);
            int h1 = b.size(0);

            // type 8
            c = otter::empty({h1, w1}, dtype1);

            auto a_a = a.accessor<float, 1, 4>();
            auto b_a = b.accessor<float, 2, 4>();
            auto c_a = c.accessor<float, 2, 4>();

            const float* ptr = a_a.data();
            const float* ptr1 = b_a.data();
            float* outptr = c_a.data();

            for (int y = 0; y < h1; y++) {
                float32x4_t _a0 = vld1q_f32(ptr);
                for (int x = 0; x < w1; x++) {
                    float32x4_t _p1 = vld1q_f32(ptr1);
                    float32x4_t _outp = op(_a0, _p1);
                    vst1q_f32(outptr, _outp);
                    ptr1 += 4;
                    outptr += 4;
                }

                ptr += 4;
            }

            return 0;
        }

        if (b.dim() == 1) {
            c = otter::empty({w}, dtype);

            if (b.size(0) == 1 && elempack1 == 1) {
                auto a_a = a.accessor<float, 1, 4>();
                auto b_a = b.accessor<float, 1>();
                auto c_a = c.accessor<float, 1, 4>();

                // type 6
                float32x4_t _b0 = vdupq_n_f32(b_a[0]);
                const float* ptr = a_a.data();
                float* outptr = c_a.data();
                for (int i = 0; i < w; i++)
                {
                    float32x4_t _p = vld1q_f32(ptr);
                    float32x4_t _outp = op(_p, _b0);
                    vst1q_f32(outptr, _outp);
                    ptr += 4;
                    outptr += 4;
                }

                return 0;
            }

            auto a_a = a.accessor<float, 1, 4>();
            auto b_a = b.accessor<float, 1, 4>();
            auto c_a = c.accessor<float, 1, 4>();

            // type 7
            const float* ptr = a_a.data();
            const float* ptr1 = b_a.data();
            float* outptr = c_a.data();
            for (int i = 0; i < w; i++) {
                float32x4_t _p = vld1q_f32(ptr);
                float32x4_t _p1 = vld1q_f32(ptr1);
                float32x4_t _outp = op(_p, _p1);
                vst1q_f32(outptr, _outp);
                ptr += 4;
                ptr1 += 4;
                outptr += 4;
            }
        }
    }

    return 0;
}

namespace BinaryOp_arm_functor {

#define MAKE_FUNCTION(NAME, IMPL)                                                \
    struct NAME                                                                  \
    {                                                                            \
        float32x4_t operator()(const float32x4_t& x, const float32x4_t& y) const \
        {                                                                        \
            return IMPL;                                                         \
        }                                                                        \
    };

MAKE_FUNCTION(binary_op_add_pack4, vaddq_f32(x, y))
MAKE_FUNCTION(binary_op_sub_pack4, vsubq_f32(x, y))
MAKE_FUNCTION(binary_op_mul_pack4, vmulq_f32(x, y))
#if __aarch64__
MAKE_FUNCTION(binary_op_div_pack4, vdivq_f32(x, y))
#else
MAKE_FUNCTION(binary_op_div_pack4, div_ps(x, y))
#endif
MAKE_FUNCTION(binary_op_max_pack4, vmaxq_f32(x, y))
MAKE_FUNCTION(binary_op_min_pack4, vminq_f32(x, y))
MAKE_FUNCTION(binary_op_pow_pack4, pow_ps(x, y))
MAKE_FUNCTION(binary_op_rsub_pack4, vsubq_f32(y, x))
#if __aarch64__
MAKE_FUNCTION(binary_op_rdiv_pack4, vdivq_f32(y, x))
#else
MAKE_FUNCTION(binary_op_rdiv_pack4, div_ps(y, x))
#endif

#undef MAKE_FUNCTION

} // namespace BinaryOp_arm_functor
#endif // __ARM_NEON

Tensor eltwise_add_pack4(const Tensor& src1, const Tensor& src2) {
    int elempack1 = src1.elempack();
    int elempack2 = src2.elempack();
    
    Tensor output;
    
#if __SSE2__
    using namespace BinaryOp_x86_functor;
    
    if (elempack1 == 4 || elempack2 == 4) {
        binary_op_pack4<binary_op_add>(src1, src2, output);
    }
#elif __ARM_NEON__
    using namespace BinaryOp_arm_functor;
    
    if (elempack1 == 4 || elempack2 == 4) {
        binary_op_pack4<binary_op_add_pack4>(src1, src2, output);
    }
#endif
    
    return output;
}

Tensor eltwise_add_pack8(const Tensor& src1, const Tensor& src2) {
    int elempack1 = src1.elempack();
    int elempack2 = src2.elempack();
    
    Tensor output;
    
#if __SSE2__
    using namespace BinaryOp_x86_functor;
#if __AVX__
    if (elempack1 == 8 || elempack2 == 8) {
        binary_op_pack8<binary_op_add>(src1, src2, output);
    }
#endif
#endif
    
    return output;
}

}   // end namespace otter
