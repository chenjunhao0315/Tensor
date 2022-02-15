//
//  main.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/29.
//

#include "OTensor.hpp"
#include "TensorShape.hpp"

using namespace std;

int main(int argc, const char * argv[]) {
    
    auto linspace = otter::linspace(-10, 10, 21, ScalarType::Float);
    cout << "linspace:\n" << linspace << endl;
    
    auto sigmoid = 1 / (1 + exp(-linspace));
    cout << "sigmoid:\n" << sigmoid << endl;
    
    auto v1 = otter::tensor({1, 2, 3});
    auto v2 = otter::tensor({3, 2, 1});
    cout << "v1:\n" << v1 << endl << "v2:\n" << v2 << endl << "v1 dot v2:\n" << v1.dot(v2) << endl;
    
    auto m1 = otter::tensor({1, 2, 3, 4, 5, 6, 7, 8, 9}, ScalarType::Float).view({3, -1});
    auto m2 = m1.transpose(1, 0);
    cout << "m1:\n" << m1 << endl;
    cout << "m1 dot m1.T:\n" << m1.mm(m2) << endl;
    
    auto pre_prem = otter::tensor({1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3}).view({2, 2, 3});
    cout << pre_prem << endl;
    auto prem = pre_prem.permute({2, 0, 1});
    
    auto accessor = prem.accessor<int, 3>();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 2; ++k) {
                cout << accessor[i][j][k];
            }
            cout << endl;
        }
        cout << endl;
    }
    
    return 0;
}
