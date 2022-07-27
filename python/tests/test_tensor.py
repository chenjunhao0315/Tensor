import sys
import numpy as np
import pytest

import otter

def test_tensor():
    tensor = otter.empty((1, ))
    assert (
        tensor.dim() == 1
        and tensor.size(dim = 0) == 1
    )
    
    tensor = otter.empty((1, 2))
    assert (
        tensor.dim() == 2
        and tensor.size(dim = 0) == 1
        and tensor.size(dim = 1) == 2
    )
    
    tensor = otter.empty((1, 2, 3))
    assert (
        tensor.dim() == 3
        and tensor.size(dim = 0) == 1
        and tensor.size(dim = 1) == 2
        and tensor.size(dim = 2) == 3
    )
    
    tensor = otter.empty((1, 2, 3, 4))
    assert (
        tensor.dim() == 4
        and tensor.size(dim = 0) == 1
        and tensor.size(dim = 1) == 2
        and tensor.size(dim = 2) == 3
        and tensor.size(dim = 3) == 4
    )

def test_numpy():
    tensor = otter.empty((1,))
    array = np.array(tensor)
    assert tensor.dim() == array.ndim and tensor.size(dim = 0) == array.shape[0]
    
    tensor = otter.empty((2, 3))
    array = np.array(tensor)
    assert (
        tensor.dim() == array.ndim
        and tensor.size(dim = 0) == array.shape[0]
        and tensor.size(dim = 1) == array.shape[1]
    )
    
    tensor = otter.empty((4, 5, 6))
    array = np.array(tensor)
    assert (
        tensor.dim() == array.ndim
        and tensor.size(dim = 0) == array.shape[0]
        and tensor.size(dim = 1) == array.shape[1]
        and tensor.size(dim = 2) == array.shape[2]
    )
    
    tensor = otter.empty((7, 8, 9, 10))
    array = np.array(tensor)
    assert (
        tensor.dim() == array.ndim
        and tensor.size(dim = 0) == array.shape[0]
        and tensor.size(dim = 1) == array.shape[1]
        and tensor.size(dim = 2) == array.shape[2]
        and tensor.size(dim = 3) == array.shape[3]
    )

    tensor = otter.empty((1, ), dtype = otter.ScalarType.Byte)
    array = np.array(tensor)
    assert array.dtype == np.uint8

    tensor = otter.empty((1, ), dtype = otter.ScalarType.Char)
    array = np.array(tensor)
    assert array.dtype == np.int8
 
    tensor = otter.empty((1, ), dtype = otter.ScalarType.Short)
    array = np.array(tensor)
    assert array.dtype == np.int16
    
    tensor = otter.empty((1, ), dtype = otter.ScalarType.Int)
    array = np.array(tensor)
    assert array.dtype == np.int32
    
    tensor = otter.empty((1, ), dtype = otter.ScalarType.Long)
    array = np.array(tensor)
    assert array.dtype == np.int64
    
    tensor = otter.empty((1, ), dtype = otter.ScalarType.Float)
    array = np.array(tensor)
    assert array.dtype == np.float32
    
    tensor = otter.empty((1, ), dtype = otter.ScalarType.Double)
    array = np.array(tensor)
    assert array.dtype == np.float64
    
    tensor = otter.empty((1, ), dtype = otter.ScalarType.Bool)
    array = np.array(tensor)
    assert array.dtype == bool
    
    tensor = otter.empty((1, ), dtype = otter.ScalarType.HFloat)
    array = np.array(tensor)
    assert array.dtype == np.float16

    array = np.random.rand(1)
    tensor = otter.tensor(array)
    check = np.array(tensor)
    assert np.array_equal(check, array)
 
    array = np.random.rand(2, 3)
    tensor = otter.tensor(array)
    check = np.array(tensor)
    assert np.array_equal(check, array)
    
    array = np.random.rand(4, 5, 6)
    tensor = otter.tensor(array)
    check = np.array(tensor)
    assert np.array_equal(check, array)
 
    array = np.random.rand(7, 8, 9, 10)
    tensor = otter.tensor(array)
    check = np.array(tensor)
    assert np.array_equal(check, array)

def test_property():
    tensor = otter.Tensor()
    assert tensor.defined() == False
    
    tensor = otter.empty((1, ))
    assert tensor.defined() == True
    
    tensor = otter.empty((1, ), dtype = otter.ScalarType.Byte)
    assert tensor.is_floating_point() == False
    assert tensor.is_signed() == False

    tensor = otter.empty((1, ), dtype = otter.ScalarType.Char)
    assert tensor.is_floating_point() == False
    assert tensor.is_signed() == True
 
    tensor = otter.empty((1, ), dtype = otter.ScalarType.Short)
    assert tensor.is_floating_point() == False
    assert tensor.is_signed() == True
    
    tensor = otter.empty((1, ), dtype = otter.ScalarType.Int)
    assert tensor.is_floating_point() == False
    assert tensor.is_signed() == True
    
    tensor = otter.empty((1, ), dtype = otter.ScalarType.Long)
    assert tensor.is_floating_point() == False
    assert tensor.is_signed() == True
    
    tensor = otter.empty((1, ), dtype = otter.ScalarType.Float)
    assert tensor.is_floating_point() == True
    assert tensor.is_signed() == True
    
    tensor = otter.empty((1, ), dtype = otter.ScalarType.Double)
    assert tensor.is_floating_point() == True
    assert tensor.is_signed() == True
    
    tensor = otter.empty((1, ), dtype = otter.ScalarType.Bool)
    assert tensor.is_floating_point() == False
    assert tensor.is_signed() == False
    
    tensor = otter.empty((1, ), dtype = otter.ScalarType.HFloat)
    assert tensor.is_floating_point() == True
    assert tensor.is_signed() == True

def test_fill():
    tensor = otter.empty((1, ))
    tensor.fill(1)
    array = np.array(tensor)
    assert np.abs(array[0] - 1.0) < sys.float_info.min
   
def test_clone():
    tensor1 = otter.rand((1, ))
    tensor2 = tensor1.clone()
    
    assert (
        tensor1.dim() == tensor2.dim()
        and tensor1.size(dim = 0) == tensor2.size(dim = 0)
        and np.array_equal(np.array(tensor1), np.array(tensor2))
    )
    
    tensor1 = otter.rand((1, 2))
    tensor2 = tensor1.clone()
    
    assert (
        tensor1.dim() == tensor2.dim()
        and tensor1.size(dim = 0) == tensor2.size(dim = 0)
        and tensor1.size(dim = 1) == tensor2.size(dim = 1)
        and np.array_equal(np.array(tensor1), np.array(tensor2))
    )
    
    tensor1 = otter.rand((1, 2, 3))
    tensor2 = tensor1.clone()
    
    assert (
        tensor1.dim() == tensor2.dim()
        and tensor1.size(dim = 0) == tensor2.size(dim = 0)
        and tensor1.size(dim = 1) == tensor2.size(dim = 1)
        and tensor1.size(dim = 2) == tensor2.size(dim = 2)
        and np.array_equal(np.array(tensor1), np.array(tensor2))
    )
    
    tensor1 = otter.rand((1, 2, 3, 4))
    tensor2 = tensor1.clone()
    
    assert (
        tensor1.dim() == tensor2.dim()
        and tensor1.size(dim = 0) == tensor2.size(dim = 0)
        and tensor1.size(dim = 1) == tensor2.size(dim = 1)
        and tensor1.size(dim = 2) == tensor2.size(dim = 2)
        and tensor1.size(dim = 3) == tensor2.size(dim = 3)
        and np.array_equal(np.array(tensor1), np.array(tensor2))
    )

def test_comparator():
    array = np.array([[1, 2, 3], [4, 5, 6]])
    tensor1 = otter.tensor(array)
    tensor2 = otter.tensor(array)
    
    tensor3 = tensor1.eq(tensor2)
    check = np.array(tensor3)
    assert np.array_equal(check, np.array([[True, True, True], [True, True, True]]))
    
    tensor3 = tensor1.eq(1)
    check = np.array(tensor3)
    assert np.array_equal(check, np.array([[True, False, False], [False, False, False]]))
    
    tensor3 = tensor1 == tensor2
    check = np.array(tensor3)
    assert np.array_equal(check, np.array([[True, True, True], [True, True, True]]))
    
    tensor3 = tensor1 == 1
    check = np.array(tensor3)
    assert np.array_equal(check, np.array([[True, False, False], [False, False, False]]))
    
    tensor3 = tensor1.ne(tensor2)
    check = np.array(tensor3)
    assert np.array_equal(check, np.array([[False, False, False], [False, False, False]]))
    
    tensor3 = tensor1.ne(1)
    check = np.array(tensor3)
    assert np.array_equal(check, np.array([[False, True, True], [True, True, True]]))
    
    tensor3 = tensor1 != tensor2
    check = np.array(tensor3)
    assert np.array_equal(check, np.array([[False, False, False], [False, False, False]]))
    
    tensor3 = tensor1 != 1
    check = np.array(tensor3)
    assert np.array_equal(check, np.array([[False, True, True], [True, True, True]]))
    
    tensor3 = tensor1.gt(tensor2)
    check = np.array(tensor3)
    assert np.array_equal(check, np.array([[False, False, False], [False, False, False]]))
    
    tensor3 = tensor1.gt(4)
    check = np.array(tensor3)
    assert np.array_equal(check, np.array([[False, False, False], [False, True, True]]))
    
    tensor3 = tensor1 > tensor2
    check = np.array(tensor3)
    assert np.array_equal(check, np.array([[False, False, False], [False, False, False]]))
    
    tensor3 = tensor1 > 4
    check = np.array(tensor3)
    assert np.array_equal(check, np.array([[False, False, False], [False, True, True]]))
    
    tensor3 = tensor1.ge(tensor2)
    check = np.array(tensor3)
    assert np.array_equal(check, np.array([[True, True, True], [True, True, True]]))
    
    tensor3 = tensor1.ge(4)
    check = np.array(tensor3)
    assert np.array_equal(check, np.array([[False, False, False], [True, True, True]]))
    
    tensor3 = tensor1 >= tensor2
    check = np.array(tensor3)
    assert np.array_equal(check, np.array([[True, True, True], [True, True, True]]))
    
    tensor3 = tensor1 >= 4
    check = np.array(tensor3)
    assert np.array_equal(check, np.array([[False, False, False], [True, True, True]]))
    
    tensor3 = tensor1.lt(tensor2)
    check = np.array(tensor3)
    assert np.array_equal(check, np.array([[False, False, False], [False, False, False]]))
    
    tensor3 = tensor1.lt(4)
    check = np.array(tensor3)
    assert np.array_equal(check, np.array([[True, True, True], [False, False, False]]))
    
    tensor3 = tensor1 < tensor2
    check = np.array(tensor3)
    assert np.array_equal(check, np.array([[False, False, False], [False, False, False]]))
    
    tensor3 = tensor1 < 4
    check = np.array(tensor3)
    assert np.array_equal(check, np.array([[True, True, True], [False, False, False]]))
    
    tensor3 = tensor1.le(tensor2)
    check = np.array(tensor3)
    assert np.array_equal(check, np.array([[True, True, True], [True, True, True]]))
    
    tensor3 = tensor1.le(4)
    check = np.array(tensor3)
    assert np.array_equal(check, np.array([[True, True, True], [True, False, False]]))
    
    tensor3 = tensor1 <= tensor2
    check = np.array(tensor3)
    assert np.array_equal(check, np.array([[True, True, True], [True, True, True]]))
    
    tensor3 = tensor1 <= 4
    check = np.array(tensor3)
    assert np.array_equal(check, np.array([[True, True, True], [True, False, False]]))

# Use numpy as golden pattern
def test_operator():
    array = np.array([[1., 2., 3.], [4., 5., 6.]])
    tensor1 = otter.tensor(array)
    tensor2 = otter.tensor(array)

    tensor3 = tensor1.add(tensor2)
    check = np.array(tensor3)
    assert np.array_equal(check, (array + array))
    
    tensor3 = tensor1 + tensor2
    check = np.array(tensor3)
    assert np.array_equal(check, (array + array))
    
    tensor3 = tensor1.add(5)
    check = np.array(tensor3)
    assert np.array_equal(check, (array + 5))
    
    tensor3 = tensor1 + 5
    check = np.array(tensor3)
    assert np.array_equal(check, (array + 5))
    
    tensor3 = tensor1.sub(tensor2)
    check = np.array(tensor3)
    assert np.array_equal(check, (array - array))
    
    tensor3 = tensor1 - tensor2
    check = np.array(tensor3)
    assert np.array_equal(check, (array - array))
    
    tensor3 = tensor1.sub(5)
    check = np.array(tensor3)
    assert np.array_equal(check, (array - 5))
    
    tensor3 = tensor1 - 5
    check = np.array(tensor3)
    assert np.array_equal(check, (array - 5))
    
    tensor3 = tensor1.mul(tensor2)
    check = np.array(tensor3)
    assert np.array_equal(check, (array * array))
    
    tensor3 = tensor1 * tensor2
    check = np.array(tensor3)
    assert np.array_equal(check, (array * array))
    
    tensor3 = tensor1.mul(5)
    check = np.array(tensor3)
    assert np.array_equal(check, (array * 5))
    
    tensor3 = tensor1 * 5
    check = np.array(tensor3)
    assert np.array_equal(check, (array * 5))
    
    tensor3 = tensor1.truediv(tensor2)
    check = np.array(tensor3)
    assert np.array_equal(check, (array / array))
    
    tensor3 = tensor1 / tensor2
    check = np.array(tensor3)
    assert np.array_equal(check, (array / array))
    
    tensor3 = tensor1.truediv(5)
    check = np.array(tensor3)
    assert np.array_equal(check, (array / 5))
    
    tensor3 = tensor1 / 5
    check = np.array(tensor3)
    assert np.array_equal(check, (array / 5))
    
    tensor3 = tensor1.mod(tensor2)
    check = np.array(tensor3)
    assert np.array_equal(check, (array % array))
    
    tensor3 = tensor1 % tensor2
    check = np.array(tensor3)
    assert np.array_equal(check, (array % array))
    
    tensor3 = tensor1.mod(5)
    check = np.array(tensor3)
    assert np.array_equal(check, (array % 5))
    
    tensor3 = tensor1 % 5
    check = np.array(tensor3)
    assert np.array_equal(check, (array % 5))
    
    array1 = np.array([[True, True, False], [False, True, True]])
    array2 = np.array([[True, False, False], [False, False, True]])
    tensor1 = otter.tensor(array1)
    tensor2 = otter.tensor(array2)
    
    tensor3 = tensor1.and_(tensor2)
    check = np.array(tensor3)
    assert np.array_equal(check, (array1 & array2))
    
    tensor3 = tensor1 & tensor2
    check = np.array(tensor3)
    assert np.array_equal(check, (array1 & array2))
    
    tensor3 = tensor1.or_(tensor2)
    check = np.array(tensor3)
    assert np.array_equal(check, (array1 | array2))
    
    tensor3 = tensor1 | tensor2
    check = np.array(tensor3)
    assert np.array_equal(check, (array1 | array2))
    
    tensor3 = tensor1.xor(tensor2)
    check = np.array(tensor3)
    assert np.array_equal(check, (array1 ^ array2))
    
    tensor3 = tensor1 ^ tensor2
    check = np.array(tensor3)
    assert np.array_equal(check, (array1 ^ array2))
    
def test_transpose_and_permute():
    array = np.arange(120).reshape((2, 3, 4, 5))
    tensor = otter.tensor(array)
    
    tensor1 = tensor.transpose(0, 1)
    check = np.array(tensor1)
    assert np.array_equal(check, (np.transpose(array, (1, 0, 2, 3))))
    
    tensor1 = tensor.transpose(1, 2)
    check = np.array(tensor1)
    assert np.array_equal(check, (np.transpose(array, (0, 2, 1, 3))))
    
    tensor1 = tensor.transpose(2, 3)
    check = np.array(tensor1)
    assert np.array_equal(check, (np.transpose(array, (0, 1, 3, 2))))
    
    tensor1 = tensor.permute((3, 2, 1, 0))
    check = np.array(tensor1)
    assert np.array_equal(check, (np.transpose(array, (3, 2, 1, 0))))
    
    tensor1 = tensor.permute((2, 1, 0, 3))
    check = np.array(tensor1)
    assert np.array_equal(check, (np.transpose(array, (2, 1, 0, 3))))

def test_indexing():
    array = np.arange(120).reshape((5, 4, 3, 2))
    tensor = otter.tensor(array)
    
    tensor1 = tensor[3]
    check = np.array(tensor1)
    assert np.array_equal(check, array[3])
    
    tensor1 = tensor[3][2]
    check = np.array(tensor1)
    assert np.array_equal(check, array[3][2])
    
    tensor1 = tensor[3][2][1]
    check = np.array(tensor1)
    assert np.array_equal(check, array[3][2][1])
    
    tensor1 = tensor[3][2][1][0]
    check = np.array(tensor1)
    assert np.array_equal(check, array[3][2][1][0])
    
    tensor1 = tensor[:]
    check = np.array(tensor1)
    assert np.array_equal(check, array[:])
    
    tensor1 = tensor[::]
    check = np.array(tensor1)
    assert np.array_equal(check, array[::])
    
    tensor1 = tensor[1:]
    check = np.array(tensor1)
    assert np.array_equal(check, array[1:])
    
    tensor1 = tensor[1::]
    check = np.array(tensor1)
    assert np.array_equal(check, array[1::])
    
    tensor1 = tensor[:3]
    check = np.array(tensor1)
    assert np.array_equal(check, array[:3])

    tensor1 = tensor[:3:]
    check = np.array(tensor1)
    assert np.array_equal(check, array[:3:])
    
    tensor1 = tensor[::2]
    check = np.array(tensor1)
    assert np.array_equal(check, array[::2])
    
    tensor1 = tensor[1:3]
    check = np.array(tensor1)
    assert np.array_equal(check, array[1:3])

    tensor1 = tensor[1::2]
    check = np.array(tensor1)
    assert np.array_equal(check, array[1::2])
    
    tensor1 = tensor[:3:2]
    check = np.array(tensor1)
    assert np.array_equal(check, array[:3:2])
    
    tensor1 = tensor[1:3:2]
    check = np.array(tensor1)
    assert np.array_equal(check, array[1:3:2])
    
    tensor1 = tensor[...]
    check = np.array(tensor1)
    assert np.array_equal(check, array[...])
    
    tensor1 = tensor[None]
    check = np.array(tensor1)
    assert np.array_equal(check, array[None])
    
    tensor1 = tensor[..., 1]
    check = np.array(tensor1)
    assert np.array_equal(check, array[..., 1])
    
    tensor1 = tensor[1, ..., 1]
    check = np.array(tensor1)
    assert np.array_equal(check, array[1, ..., 1])
    
    tensor1 = tensor[1, ..., None, 1]
    check = np.array(tensor1)
    assert np.array_equal(check, array[1, ..., None, 1])

def test_math():
    tensor = otter.rand((3, 3, 3))
    array = np.array(tensor)
    
    tensor1 = otter.abs(tensor)
    check = np.array(tensor1)
    assert np.array_equal(check, np.abs(array))
    
    tensor1 = otter.sin(tensor)
    check = np.array(tensor1)
    assert np.isclose(check, np.sin(array), rtol = sys.float_info.min).any()
    
    tensor1 = otter.cos(tensor)
    check = np.array(tensor1)
    assert np.isclose(check, np.cos(array), rtol = sys.float_info.min).any()
    
    tensor1 = otter.tan(tensor)
    check = np.array(tensor1)
    assert np.isclose(check, np.tan(array), rtol = sys.float_info.min).any()
    
    tensor1 = otter.exp(tensor)
    check = np.array(tensor1)
    assert np.isclose(check, np.exp(array), rtol = sys.float_info.min).any()
    
    tensor1 = otter.sqrt(tensor)
    check = np.array(tensor1)
    assert np.isclose(check, np.sqrt(array), rtol = sys.float_info.min).any()

def test_sort():
    tensor = otter.rand((10, ))
    array = np.array(tensor)
    
    sorted, indices = tensor.sort(dim = 0, decreasing = True)
    check_sorted = np.array(sorted)
    check_indices = np.array(indices)
    sorted = np.sort(array, axis = 0)[::-1]
    indices = np.argsort(array, axis = 0)[::-1]
    assert np.array_equal(check_sorted, sorted)
    assert np.array_equal(check_indices, indices)
    
    sorted, indices = tensor.sort(dim = 0, decreasing = False)
    check_sorted = np.array(sorted)
    check_indices = np.array(indices)
    sorted = np.sort(array, axis = 0)
    indices = np.argsort(array, axis = 0)
    assert np.array_equal(check_sorted, sorted)
    assert np.array_equal(check_indices, indices)
    
