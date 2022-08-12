//
//  TensorSpectral.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/8/12.
//

#include <cmath>

#include "TensorSpectral.hpp"
#include "TensorFactory.hpp"
#include "Dispatch.hpp"
#include "Complex.hpp"

namespace otter {

/*
 FFT
 
 x[0] ---- * ------- * ------------- *   X[0]
       \/ W0N \   / W0N \         / W0N
       /\      \ /       \       /
 x[4] ---- * ------- * ------------- *   X[1]
          W4N \/ \/ W2N \  \   /  / W1N
              /\ /\      \  \ /  /
 x[2] ---- * ------- * ------------- *   X[2]
       \/ W0N  / \  W4N \  \/ \/  / W2N
       /\     /   \      \ /\ /\ /
 x[6] ---- * ------- * ------------- *   X[3]
          W4N       W6N \/ \/ \/ \/ W3N
                        /\ /\ /\ /\
 x[1] ---- * ------- * ------------- *   X[4]
       \/ W0N \   / W0N  / \/ \/ \  W4N
       /\      \ /      /  /\ /\  \
 x[5] ---- * ------- * ------------- *   X[5]
          W4N \/ \/ W2N  /  / \  \  W5N
              /\ /\     /  /   \  \
 x[3] ---- * ------- * ------------- *   X[6]
       \/ W0N  / \  W4N  /       \  W6N
       /\     /   \     /         \
 x[7] ---- * ------- * ------------- *   X[7]
          W4N       W6N             W7N
 
 => Simplify
 
           STAGE_1      STAGE_2           STAGE_3
 
 PROC_NUM     2            4                 8
 
 x[0]       ---- *      ------- *      ------------- *   X[0]
             \/ W0N      \   / W0N      \         / W0N
             /\           \ /            \       /
 x[4] * W0N ---- *      ------- *      ------------- *   X[1]
                -1       \/ \/ W2N      \  \   /  / W1N
                         /\ /\           \  \ /  /
 x[2]       ---- *  W0N ------- *      ------------- *   X[2]
             \/ W0N       / \  -1       \  \/ \/  / W2N
             /\          /   \           \ /\ /\ /
 x[6] * W0N ---- *  W2N ------- *      ------------- *   X[3]
                -1             -1       \/ \/ \/ \/ W3N
                                        /\ /\ /\ /\
 x[1]       ---- *      ------- *  W0N ------------- *   X[4]
             \/ W0N      \   / W0N       / \/ \/ \  -1
             /\           \ /           /  /\ /\  \
 x[5] * W0N ---- *      ------- *  W1N ------------- *   X[5]
                -1       \/ \/ W2N       /  / \  \  -1
                         /\ /\          /  /   \  \
 x[3]       ---- *  W0N ------- *  W2N ------------- *   X[6]
             \/ W0N       / \  -1        /       \  -1
             /\          /   \          /         \
 x[7] * W0N ---- *  W2N ------- *  W3N ------------- *   X[7]
                -1             -1                   -1
 
 =>
 
 STAGE = LOG2(N)
 
 PROC_NUM = 2^STAGE_I
 THETA = -2.0 * PI / PROC_NUM
 
 FOR STAGE_I IN RANGE(STAGE):
    PROC_NUM = 2^STAGE_I
    THETA = -2.0 * PI / PROC_NUM
    DELTA_W = W0(N / PROC_NUM)
    FOR PROC = 0 TO N STEP PROC_NUM:
        W = W0N
        FOR EVEN = PROC TO (PRCO + PROC_NUM / 2):
            ODD = EVEN + PROC_NUM / 2
            G = SRC[EVEN]
            H = SRC[ODD] * W
            SRC[EVEN] = G + H
            SRC[ODD]  = G - H
            W = W * DELTA_W
            
 */

std::tuple<Tensor, Tensor> fft(const Tensor& real_part, const Tensor& imag_part) {
    OTTER_CHECK(real_part.dim() == 1, "Expect 1D array");
    int N = real_part.size(0);
    OTTER_CHECK(N > 0 && !(N & (N - 1)), "Require length is the power of 2");
    
    auto real = real_part.clone();
    auto imag = imag_part.defined() ? imag_part.clone() : otter::zeros_like(real_part);
    
    OTTER_DISPATCH_FLOATING_TYPES(real.scalar_type(), "fft", [&]() {
        auto real_ptr = real.data_ptr<scalar_t>();
        auto imag_ptr = imag.data_ptr<scalar_t>();
        
        // bit-reverse permutation
        for (int i = 1, j = 0; i < N; ++i) {
            for (int k = N >> 1; !((j ^= k) & k); k >>= 1);
            
            if (i > j) {
                std::swap(real_ptr[i], real_ptr[j]);
                std::swap(imag_ptr[i], imag_ptr[j]);
            }
        }
        
        // bufferfly calculation
        for (int k = 2; k <= N; k <<= 1) {
            float theta = -2.0 * M_PI / k;
            otter::complex<scalar_t> delta_w(std::cos(theta), std::sin(theta));
            
            for (int j = 0; j < N; j += k) {
                otter::complex<scalar_t> w(1, 0);
                for (int i = j; i < j + k / 2; i++) {
                    auto h = otter::complex<scalar_t>(real_ptr[i], imag_ptr[i]);
                    auto g = otter::complex<scalar_t>(real_ptr[i + k / 2], imag_ptr[i + k / 2]) * w;
                    
                    auto even = h + g;
                    auto odd  = h - g;
                    
                    real_ptr[i] = even.real();
                    imag_ptr[i] = even.imag();
                    real_ptr[i + k / 2] = odd.real();
                    imag_ptr[i + k / 2] = odd.imag();
                    
                    w *= delta_w;
                }
            }
        }
    });

    return std::make_tuple(real, imag);
}

}   // namespace otter
