#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>

#include <iostream>
#include <vector>
#include <functional>

#include <Tensor.hpp>
#include <TensorFactory.hpp>
#include <TensorMaker.hpp>
#include <TensorOperator.hpp>
#include <TensorFunction.hpp>
#include <TensorIndexing.hpp>
#include <ArrayRef.hpp>
#include <TypeMeta.hpp>
#include <Formatting.hpp>
#include <Parallel.hpp>

#include "tensor_str.hpp"

namespace py = pybind11;

using namespace otter;

template <typename T>
std::vector<T> tuple_to_vector(py::tuple tuple);

template <typename T, typename U>
std::vector<T> convert_vector_dtype(std::vector<U>& vector);

std::string get_tensor_format(const Tensor& t);

Tensor tensor_index(const Tensor& tensor, const py::tuple& tuple);
Tensor tensor_index(const Tensor& tensor, int64_t dim);
Tensor tensor_index(const Tensor& tensor, const py::slice& slice);
Tensor tensor_index(const Tensor& tensor, const py::ellipsis&);
Tensor tensor_index(const Tensor& tensor, const py::none&);

PYBIND11_MODULE(otter, m) {
    
    py::class_<Tensor> tensor(m, "Tensor", py::buffer_protocol());
    tensor.def(py::init<>())
    .def_buffer([](Tensor& t) {
        std::string format = get_tensor_format(t);
        auto tensor_shape = t.sizes().vec();
        auto tensor_strides = t.strides().vec();
        int64_t itemsize = t.itemsize();
        std::transform(tensor_strides.begin(), tensor_strides.end(), tensor_strides.begin(),
                       [&itemsize](auto& c) { return c * itemsize; });
        std::vector<py::ssize_t> shape = convert_vector_dtype<py::ssize_t>(tensor_shape);
        std::vector<py::ssize_t> strides = convert_vector_dtype<py::ssize_t>(tensor_strides);
        
        return py::buffer_info(
            t.raw_data(),   /* Pointer to buffer */
            t.itemsize(),   /* Size of one scalar */
            format,         /* Python struct-style format descriptor */
            t.dim(),        /* Number of dimensions */
            shape,          /* Buffer dimensions */
            strides         /* Strides (in bytes) for each index */
        );
    })
    .def("is_floating_point", &Tensor::is_floating_point, "Check the data type is floating type or not")
    .def("is_signed", &Tensor::is_signed, "Check the data type is singed type or not")
    .def("defined", &Tensor::defined, "Check the tensor is defined or not")
    .def("print", &Tensor::print, "Print out shape info")
    .def("packing", (Tensor(Tensor::*)(int))&Tensor::packing, py::arg("packing"))
    .def("clone", [](Tensor& tensor) {
        return tensor.clone();
    })
    .def("copy_", &Tensor::copy_, "Copy from other tensor")
    .def("contiguous", &Tensor::contiguous, "Make memory layout contiguous")
    .def("dim", &Tensor::dim, "Dimension of tensor")
    .def("size", [](Tensor& tensor, int64_t dim) {
        return tensor.size(dim);
    }, py::arg("dim"))
    .def("itemsize", &Tensor::itemsize, "Get the itemsize in bytes")
    .def("fill", [](Tensor& tensor, float value) {
        return tensor.fill_(value);
    }, py::arg("value"))
    .def("permute", [](Tensor& tensor, py::tuple dims) {
        auto tensor_dim = tuple_to_vector<int64_t>(dims);
        
        return tensor.permute(tensor_dim);
    }, py::arg("dim"))
    .def("transpose_", [](Tensor& tensor, int dim1, int dim2) {
        return tensor.transpose_(dim1, dim2);
    }, py::arg("dim1"), py::arg("dim2"))
    .def("transpose", [](Tensor& tensor, int dim1, int dim2) {
        return tensor.transpose(dim1, dim2);
    }, py::arg("dim1"), py::arg("dim2"))
    .def("repeat", [](Tensor& tensor, py::tuple dims) {
        auto tensor_dim = tuple_to_vector<int64_t>(dims);
        
        return tensor.repeat(tensor_dim);
    }, py::arg("dim"))
    .def("expand", [](Tensor& tensor, py::tuple dims) {
        auto tensor_dim = tuple_to_vector<int64_t>(dims);
        
        return tensor.expand(tensor_dim);
    }, py::arg("dims"))
    .def("expand_as", [](Tensor& tensor, Tensor& tensor_target) {
        return tensor.expand_as(tensor_target);
    }, py::arg("dims"))
    .def("view", [](Tensor& tensor, py::tuple dims) {
        auto tensor_dim = tuple_to_vector<int64_t>(dims);
        
        return tensor.view(tensor_dim);
    }, py::arg("dims"))
    .def("reshape", [](Tensor& tensor, py::tuple dims) {
        auto tensor_dim = tuple_to_vector<int64_t>(dims);
        
        return tensor.reshape(tensor_dim);
    }, py::arg("dims"))
    .def("reshape_as", [](Tensor& tensor, Tensor& other) {
        return tensor.reshape_as(other);
    }, py::arg("other"))
    .def("slice", [](Tensor& tensor, int64_t dim, int64_t start, int64_t end, int64_t step) {
        return tensor.slice(dim, start, end, step);
    }, py::arg("dim") = 0, py::arg("start") = INT64_MAX, py::arg("end") = 0, py::arg("step") = 1)
    .def("unsqueeze", [](Tensor& tensor, int64_t dim) {
        return tensor.unsqueeze(dim);
    }, py::arg("dim"))
    .def("unsqueeze_", [](Tensor& tensor, int64_t dim) {
        return tensor.unsqueeze_(dim);
    }, py::arg("dim"))
    .def("squeeze", [](Tensor& tensor, int64_t dim) {
        return tensor.squeeze(dim);
    }, py::arg("dim"))
    .def("squeeze_", [](Tensor& tensor, int64_t dim) {
        return tensor.squeeze_(dim);
    }, py::arg("dim"))
    .def("flatten", [](Tensor& tensor, int64_t start_dim, int64_t end_dim) {
        return tensor.flatten(start_dim, end_dim);
    }, py::arg("start_dim") = 0, py::arg("end_dim") = -1)
    .def("__getitem__", [](const Tensor& tensor, const py::tuple tuple) {
        return tensor_index(tensor, tuple);
    }, py::arg("index"))
    .def("__getitem__", [](const Tensor& tensor, int64_t dim) {
        return tensor_index(tensor, dim);
    }, py::arg("index"))
    .def("__getitem__", [](const Tensor& tensor, const py::slice& slice) {
        return tensor_index(tensor, slice);
    }, py::arg("index"))
    .def("__getitem__", [](const Tensor& tensor, const py::ellipsis& ellipsis) {
        return tensor_index(tensor, ellipsis);
    }, py::arg("index"))
    .def("__getitem__", [](const Tensor& tensor, const py::none& none) {
        return tensor_index(tensor, none);
    }, py::arg("index"))
    .def("__repr__", [](const Tensor& tensor) {
        return tensor_str(tensor.to(ScalarType::Double).contiguous());
    })
    .def("__neg__", [](const Tensor& tensor1) {
        return tensor1.neg();
    }, py::is_operator())
    .def("neg", [](const Tensor& tensor1) {
        return tensor1.neg();
    })
    .def("__not__", [](const Tensor& tensor1) {
        return tensor1.bitwise_not();
    }, py::is_operator())
    .def("not", [](const Tensor& tensor1) {
        return tensor1.bitwise_not();
    })
    .def("__lt__", [](const Tensor& tensor1, const Tensor& tensor2) {
        return tensor1.lt(tensor2);
    }, py::is_operator())
    .def("lt", [](const Tensor& tensor1, const Tensor& tensor2) {
        return tensor1.lt(tensor2);
    })
    .def("__lt__", [](const Tensor& tensor1, float value) {
        return tensor1.lt(value);
    }, py::is_operator())
    .def("lt", [](const Tensor& tensor1, float value) {
        return tensor1.lt(value);
    })
    .def("__lt__", [](const Tensor& tensor1, int value) {
        return tensor1.lt(value);
    }, py::is_operator())
    .def("lt", [](const Tensor& tensor1, int value) {
        return tensor1.lt(value);
    })
    .def("__le__", [](const Tensor& tensor1, const Tensor& tensor2) {
        return tensor1.le(tensor2);
    }, py::is_operator())
    .def("le", [](const Tensor& tensor1, const Tensor& tensor2) {
        return tensor1.le(tensor2);
    })
    .def("__le__", [](const Tensor& tensor1, float value) {
        return tensor1.le(value);
    }, py::is_operator())
    .def("le", [](const Tensor& tensor1, float value) {
        return tensor1.le(value);
    })
    .def("__le__", [](const Tensor& tensor1, int value) {
        return tensor1.le(value);
    }, py::is_operator())
    .def("le", [](const Tensor& tensor1, int value) {
        return tensor1.le(value);
    })
    .def("__eq__", [](const Tensor& tensor1, const Tensor& tensor2) {
        return tensor1.eq(tensor2);
    }, py::is_operator())
    .def("eq", [](const Tensor& tensor1, const Tensor& tensor2) {
        return tensor1.eq(tensor2);
    })
    .def("__eq__", [](const Tensor& tensor1, float value) {
        return tensor1.eq(value);
    }, py::is_operator())
    .def("eq", [](const Tensor& tensor1, float value) {
        return tensor1.eq(value);
    })
    .def("__eq__", [](const Tensor& tensor1, int value) {
        return tensor1.eq(value);
    }, py::is_operator())
    .def("eq", [](const Tensor& tensor1, int value) {
        return tensor1.eq(value);
    })
    .def("__ne__", [](const Tensor& tensor1, const Tensor& tensor2) {
        return tensor1.ne(tensor2);
    }, py::is_operator())
    .def("ne", [](const Tensor& tensor1, const Tensor& tensor2) {
        return tensor1.ne(tensor2);
    })
    .def("__ne__", [](const Tensor& tensor1, float value) {
        return tensor1.ne(value);
    }, py::is_operator())
    .def("ne", [](const Tensor& tensor1, float value) {
        return tensor1.ne(value);
    })
    .def("__ne__", [](const Tensor& tensor1, int value) {
        return tensor1.ne(value);
    }, py::is_operator())
    .def("ne", [](const Tensor& tensor1, int value) {
        return tensor1.ne(value);
    })
    .def("__ge__", [](const Tensor& tensor1, const Tensor& tensor2) {
        return tensor1.ge(tensor2);
    }, py::is_operator())
    .def("ge", [](const Tensor& tensor1, const Tensor& tensor2) {
        return tensor1.ge(tensor2);
    })
    .def("__ge__", [](const Tensor& tensor1, float value) {
        return tensor1.ge(value);
    }, py::is_operator())
    .def("ge", [](const Tensor& tensor1, float value) {
        return tensor1.ge(value);
    })
    .def("__ge__", [](const Tensor& tensor1, int value) {
        return tensor1.ge(value);
    }, py::is_operator())
    .def("ge", [](const Tensor& tensor1, int value) {
        return tensor1.ge(value);
    })
    .def("__gt__", [](const Tensor& tensor1, const Tensor& tensor2) {
        return tensor1.gt(tensor2);
    }, py::is_operator())
    .def("gt", [](const Tensor& tensor1, const Tensor& tensor2) {
        return tensor1.gt(tensor2);
    })
    .def("__gt__", [](const Tensor& tensor1, float value) {
        return tensor1.gt(value);
    }, py::is_operator())
    .def("gt", [](const Tensor& tensor1, float value) {
        return tensor1.gt(value);
    })
    .def("__gt__", [](const Tensor& tensor1, int value) {
        return tensor1.gt(value);
    }, py::is_operator())
    .def("gt", [](const Tensor& tensor1, int value) {
        return tensor1.gt(value);
    })
    .def("__add__", [](const Tensor& tensor1, const Tensor& tensor2) {
        return tensor1.add(tensor2);
    }, py::is_operator())
    .def("add", [](const Tensor& tensor1, const Tensor& tensor2) {
        return tensor1.add(tensor2);
    })
    .def(py::self + float())
    .def(float() + py::self)
    .def("add", [](const Tensor& tensor1, float value) {
        return tensor1.add(value);
    })
    .def(py::self + int())
    .def(int() + py::self)
    .def("add", [](const Tensor& tensor1, int value) {
        return tensor1.add(value);
    })
    .def("__sub__", [](const Tensor& tensor1, const Tensor& tensor2) {
        return tensor1.sub(tensor2);
    }, py::is_operator())
    .def("sub", [](const Tensor& tensor1, const Tensor& tensor2) {
        return tensor1.sub(tensor2);
    })
    .def(py::self - float())
    .def(float() - py::self)
    .def("sub", [](const Tensor& tensor1, float value) {
        return tensor1.sub(value);
    })
    .def(py::self - int())
    .def(int() - py::self)
    .def("sub", [](const Tensor& tensor1, int value) {
        return tensor1.sub(value);
    })
    .def("__mul__", [](const Tensor& tensor1, const Tensor& tensor2) {
        return tensor1.mul(tensor2);
    }, py::is_operator())
    .def("mul", [](const Tensor& tensor1, const Tensor& tensor2) {
        return tensor1.mul(tensor2);
    })
    .def(py::self * float())
    .def(float() * py::self)
    .def("mul", [](const Tensor& tensor1, float value) {
        return tensor1.mul(value);
    })
    .def(py::self * int())
    .def(int() * py::self)
    .def("mul", [](const Tensor& tensor1, int value) {
        return tensor1.mul(value);
    })
    .def("__truediv__", [](const Tensor& tensor1, const Tensor& tensor2) {
        return tensor1.div(tensor2);
    }, py::is_operator())
    .def("truediv", [](const Tensor& tensor1, const Tensor& tensor2) {
        return tensor1.div(tensor2);
    })
    .def(py::self / float())
    .def(float() / py::self)
    .def("truediv", [](const Tensor& tensor1, float value) {
        return tensor1.div(value);
    })
    .def(py::self / int())
    .def(int() / py::self)
    .def("truediv", [](const Tensor& tensor1, int value) {
        return tensor1.div(value);
    })
    .def("__mod__", [](const Tensor& tensor1, const Tensor& tensor2) {
        return tensor1.remainder(tensor2);
    }, py::is_operator())
    .def("mod", [](const Tensor& tensor1, const Tensor& tensor2) {
        return tensor1.remainder(tensor2);
    })
    .def(py::self % float())
    .def(float() % py::self)
    .def("mod", [](const Tensor& tensor1, float value) {
        return tensor1.remainder(value);
    })
    .def(py::self % int())
    .def(int() % py::self)
    .def("mod", [](const Tensor& tensor1, int value) {
        return tensor1.remainder(value);
    })
    .def("__and__", [](const Tensor& tensor1, const Tensor& tensor2) {
        return tensor1.bitwise_and(tensor2);
    }, py::is_operator())
    .def("and_", [](const Tensor& tensor1, const Tensor& tensor2) {
        return tensor1.bitwise_and(tensor2);
    })
    .def("__and__", [](const Tensor& tensor1, float value) {
        return tensor1.bitwise_and(value);
    }, py::is_operator())
    .def("and_", [](const Tensor& tensor1, float value) {
        return tensor1.bitwise_and(value);
    })
    .def("__and__", [](const Tensor& tensor1, int value) {
        return tensor1.bitwise_and(value);
    }, py::is_operator())
    .def("and_", [](const Tensor& tensor1, int value) {
        return tensor1.bitwise_and(value);
    })
    .def("__or__", [](const Tensor& tensor1, const Tensor& tensor2) {
        return tensor1.bitwise_or(tensor2);
    }, py::is_operator())
    .def("or_", [](const Tensor& tensor1, const Tensor& tensor2) {
        return tensor1.bitwise_or(tensor2);
    })
    .def("__or__", [](const Tensor& tensor1, float value) {
        return tensor1.bitwise_or(value);
    }, py::is_operator())
    .def("or_", [](const Tensor& tensor1, float value) {
        return tensor1.bitwise_or(value);
    })
    .def("__or__", [](const Tensor& tensor1, int value) {
        return tensor1.bitwise_or(value);
    }, py::is_operator())
    .def("or_", [](const Tensor& tensor1, int value) {
        return tensor1.bitwise_or(value);
    })
    .def("__xor__", [](const Tensor& tensor1, const Tensor& tensor2) {
        return tensor1.bitwise_xor(tensor2);
    }, py::is_operator())
    .def("xor", [](const Tensor& tensor1, const Tensor& tensor2) {
        return tensor1.bitwise_xor(tensor2);
    })
    .def("__xor__", [](const Tensor& tensor1, float value) {
        return tensor1.bitwise_xor(value);
    }, py::is_operator())
    .def("xor", [](const Tensor& tensor1, float value) {
        return tensor1.bitwise_xor(value);
    })
    .def("__xor__", [](const Tensor& tensor1, int value) {
        return tensor1.bitwise_xor(value);
    }, py::is_operator())
    .def("xor", [](const Tensor& tensor1, int value) {
        return tensor1.bitwise_xor(value);
    });
    
    py::enum_<ScalarType>(m, "ScalarType")
    .value("Byte", ScalarType::Byte)
    .value("Char", ScalarType::Char)
    .value("Short", ScalarType::Short)
    .value("Int", ScalarType::Int)
    .value("Long", ScalarType::Long)
    .value("Float", ScalarType::Float)
    .value("Double", ScalarType::Double)
    .value("Bool", ScalarType::Bool)
    .value("HFloat", ScalarType::HFloat)
    .value("Byte4", ScalarType::Byte4)
    .value("Int4", ScalarType::Int4)
    .value("Float4", ScalarType::Float4)
    .value("HFloat4", ScalarType::HFloat4)
    .value("Byte8", ScalarType::Byte8)
    .value("Int8", ScalarType::Int8)
    .value("Float8", ScalarType::Float8)
    .value("HFloat8", ScalarType::HFloat8)
    .value("Byte16", ScalarType::Byte16)
    .value("Int16", ScalarType::Int16)
    .value("Float16", ScalarType::Float16)
    .value("HFloat16", ScalarType::HFloat16)
    .value("Byte32", ScalarType::Byte32)
    .value("Int32", ScalarType::Int32)
    .value("Float32", ScalarType::Float32)
    .value("HFloat32", ScalarType::HFloat32)
    .value("Byte64", ScalarType::Byte64)
    .value("Int64", ScalarType::Int64)
    .value("Float64", ScalarType::Float64)
    .value("HFloat64", ScalarType::HFloat64);
    
    m.def("tensor", [](py::buffer const b) {
        py::buffer_info info = b.request();
        
        auto tensor_shape = convert_vector_dtype<int64_t>(info.shape);
        auto tensor_strides = convert_vector_dtype<int64_t>(info.strides);
        
        ScalarType dtype = ScalarType::Undefined;
        
        if (info.format == py::format_descriptor<double>::format()) {
            dtype = ScalarType::Double;
        } else if (info.format == py::format_descriptor<float>::format()) {
            dtype = ScalarType::Float;
        } else if (info.format == py::format_descriptor<int>::format()) {
            dtype = ScalarType::Int;
        } else if (info.format == "l") {
            dtype = ScalarType::Long;
        } else if (info.format == "e") {
            dtype = ScalarType::HFloat;
        } else if (info.format == py::format_descriptor<int8_t>::format()) {
            dtype = ScalarType::Char;
        } else if (info.format == py::format_descriptor<uint8_t>::format()) {
            dtype = ScalarType::Byte;
        } else if (info.format == "?") {
            dtype = ScalarType::Bool;
        } else {
            std::stringstream ss;
            ss << "convert numpy.ndarray to otter.Tensor dtype fail get " << info.format;
            pybind11::pybind11_fail(ss.str());
        }
        
        TypeMeta meta = TypeMeta::fromScalarType(dtype);
        int64_t itemsize = meta.itemsize();
        std::transform(tensor_strides.begin(), tensor_strides.end(), tensor_strides.begin(),
                       [&itemsize](auto& c) { return c / itemsize; });
        
        return from_blob(info.ptr, tensor_shape, tensor_strides, dtype);
    }, py::arg("array"));
    m.def("empty", [](py::tuple shape, ScalarType dtype) {
        auto tensor_shape = tuple_to_vector<int64_t>(shape);
        
        return otter::empty(tensor_shape, dtype);
    }, py::arg("shape"), py::arg("dtype") = ScalarType::Float);
    m.def("empty_like", [](const Tensor& other) {
        
        return otter::empty_like(other);
    }, py::arg("other"));
    m.def("full", [](py::tuple shape, float value, ScalarType dtype) {
        auto tensor_shape = tuple_to_vector<int64_t>(shape);
        
        return otter::full(tensor_shape, value, dtype);
    }, py::arg("shape"), py::arg("value") = 0.f, py::arg("dtype") = ScalarType::Float);
    m.def("ones", [](py::tuple shape, ScalarType dtype) {
        auto tensor_shape = tuple_to_vector<int64_t>(shape);
        
        return otter::ones(tensor_shape, dtype);
    }, py::arg("shape"), py::arg("dtype") = ScalarType::Float);
    m.def("ones_like", [](const Tensor& other) {
        
        return otter::ones_like(other);
    }, py::arg("other"));
    m.def("zeros", [](py::tuple shape, ScalarType dtype) {
        auto tensor_shape = tuple_to_vector<int64_t>(shape);
        
        return otter::zeros(tensor_shape, dtype);
    }, py::arg("shape"), py::arg("dtype") = ScalarType::Float);
    m.def("zeros_like", [](const Tensor& other) {
        
        return otter::zeros_like(other);
    }, py::arg("other"));
    m.def("rand", [](py::tuple shape, ScalarType dtype) {
        auto tensor_shape = tuple_to_vector<int64_t>(shape);
        
        return otter::rand(tensor_shape, dtype);
    }, py::arg("shape"), py::arg("dtype") = ScalarType::Float);
    m.def("range", [](float start, float end, float step, ScalarType dtype) {
        return otter::range(start, end, step, dtype);
    }, py::arg("start") = 0, py::arg("end"), py::arg("step") = 1, py::arg("dtype") = ScalarType::Float);
    m.def("arange", [](float start, float end, float step, ScalarType dtype) {
        return otter::arange(start, end, step, dtype);
    }, py::arg("start") = 0, py::arg("end"), py::arg("step") = 1, py::arg("dtype") = ScalarType::Float);
    
    m.def("abs", &otter::native::abs);
    m.def("sin", &otter::native::sin);
    m.def("cos", &otter::native::cos);
    m.def("tan", &otter::native::tan);
    m.def("exp", &otter::native::exp);
    m.def("sqrt", &otter::native::sqrt);
    
    m.doc() = R"pbdoc(
        otter python wrapper
    )pbdoc";
    
    m.attr("__version__") = "dev";
}

Tensor tensor_index(const Tensor& tensor, const py::tuple& tuple) {
    std::vector<indexing::TensorIndex> indices;

    for (auto arg : tuple) {
        std::string info = arg.attr("__str__")().cast<std::string>();
        if (info.find("Ellipsis") != std::string::npos) {
            indices.push_back(indexing::TensorIndex(indexing::Ellipsis));
        } else if (info.find("None") != std::string::npos) {
            indices.push_back(indexing::TensorIndex(indexing::None));
        } else if (info.find("slice") != std::string::npos) {
            py::slice slice = arg.cast<py::slice>();
            py::ssize_t start = 0, stop = 0, step = 0, slicelength = 0;
            if (!slice.compute(tensor.numel(), &start, &stop, &step, &slicelength)) {
                throw py::error_already_set();
            }
            indices.push_back(indexing::TensorIndex(indexing::Slice(start, stop, step)));
        } else {
            int64_t dim = arg.cast<int64_t>();
            
            indices.push_back(indexing::TensorIndex(dim));
        }
    }

    return tensor.index(indices);
}

Tensor tensor_index(const Tensor& tensor, int64_t dim) {
    indexing::TensorIndex index(dim);
    
    return tensor.index(index);
}

Tensor tensor_index(const Tensor& tensor, const py::slice& slice) {
    py::ssize_t start = 0, stop = 0, step = 0, slicelength = 0;
    if (!slice.compute(tensor.numel(), &start, &stop, &step, &slicelength)) {
        throw py::error_already_set();
    }
    indexing::TensorIndex index(indexing::Slice(start, stop, step));
    
    return tensor.index(index);
}

Tensor tensor_index(const Tensor& tensor, const py::ellipsis&) {
    indexing::TensorIndex index(indexing::Ellipsis);
    
    return tensor.index(index);
}

Tensor tensor_index(const Tensor& tensor, const py::none&) {
    indexing::TensorIndex index(indexing::None);
    
    return tensor.index(index);
}


template <typename T>
std::vector<T> tuple_to_vector(py::tuple tuple) {
    std::vector<T> vector;
    for (const auto i : otter::irange(0, tuple.size())) {
        vector.push_back(tuple[i].cast<T>());
    }
    
    return vector;
}

template <typename T, typename U>
std::vector<T> convert_vector_dtype(std::vector<U>& vector) {
    std::vector<T> converted_vector(vector.begin(), vector.end());
    
    return converted_vector;
}

std::string get_tensor_format(const Tensor& t) {
    ScalarType dtype = t.scalar_type();
    
    if (dtype == ScalarType::Byte) {
        return "B";
    } else if (dtype == ScalarType::Char) {
        return "b";
    } else if (dtype == ScalarType::Bool) {
        return "?";
    } else if (dtype == ScalarType::Short) {
        return "h";
    } else if (dtype == ScalarType::Int) {
        return "i";
    } else if (dtype == ScalarType::Long) {
        return "l";
    } else if (dtype == ScalarType::Float) {
        return "f";
    } else if (dtype == ScalarType::HFloat) {
        return "e";
    } else if (dtype == ScalarType::Double) {
        return "d";
    }
    
    return "f";
}
