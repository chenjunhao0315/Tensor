configure_file(platform.hpp.in ${CMAKE_CURRENT_BINARY_DIR}/platform.hpp)

function(otter_src_group otter_src_string folder)
    string(REPLACE " " ";" _otter_src_list ${otter_src_string})

    string(REGEX REPLACE "/" "\\\\" _target_folder "${folder}")

    foreach(_file IN LISTS ${_otter_src_list})
        source_group ("${_target_folder}" FILES "${_file}")
    endforeach ()
endfunction()

file(GLOB otter_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

otter_src_group(otter_SRCS "sources")

include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

if(otter_SHARED_LIB)
    add_library(otter SHARED ${otter_SRCS})
else()
    add_library(otter STATIC ${otter_SRCS})
endif()
set_target_properties(otter PROPERTIES DEBUG_POSTFIX "d")
set_target_properties(otter PROPERTIES VERSION ${otter_VERSION_STRING} SOVERSION ${otter_VERSION_MAJOR})

include(GenerateExportHeader)
generate_export_header(otter)

if(NOT OTTER_SHARED_LIB)
    set_target_properties(otter PROPERTIES COMPILE_FLAGS -DOTTER_STATIC_DEFINE)
endif()

target_include_directories(otter
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include/otter>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

if(OTTER_OPENMP)
    if(NOT OTTER_SIMPLEOMP)
        find_package(OpenMP)
        if(NOT TARGET OpenMP::OpenMP_CXX AND (OpenMP_CXX_FOUND OR OPENMP_FOUND))
            target_compile_options(otter PRIVATE ${OpenMP_CXX_FLAGS})
        endif()
    endif()

    if(OTTER_SIMPLEOMP OR OpenMP_CXX_FOUND OR OPENMP_FOUND)
        if(OTTER_CMAKE_VERBOSE)
            message("Building with OpenMP")
        endif()

        if(OTTER_SIMPLEOMP)
            if(IOS OR APPLE)
                target_compile_options(otter PRIVATE -Xpreprocessor -fopenmp)
            else()
                target_compile_options(otter PRIVATE -fopenmp)
            endif()
        elseif(ANDROID_NDK_MAJOR AND (ANDROID_NDK_MAJOR GREATER 20))
            target_compile_options(otter PRIVATE -fopenmp)
            target_link_libraries(otter PUBLIC -fopenmp -static-openmp)
        elseif(OpenMP_CXX_FOUND)
            target_link_libraries(otter PUBLIC OpenMP::OpenMP_CXX)
        else()
            target_link_libraries(otter PRIVATE "${OpenMP_CXX_FLAGS}")
        endif()
    endif()
endif()

if(WIN32)
    target_compile_definitions(otter PUBLIC NOMINMAX)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC" OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_SIMULATE_ID MATCHES "MSVC" AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT MATCHES "MSVC"))
    target_compile_definitions(otter PRIVATE _SCL_SECURE_NO_WARNINGS _CRT_SECURE_NO_DEPRECATE)

    if(CMAKE_BUILD_TYPE MATCHES "(Release|RELEASE|release)")
        target_compile_options(otter PRIVATE /fp:fast)
    endif()

    if(OTTER_SHARED_LIB)
        # msvc argues about stl string and vector uses in exported functions
        target_compile_options(otter PRIVATE /wd4251)
    endif()
else()
    target_compile_options(otter PRIVATE -Wall -Wextra -Wno-unused-function)

    if(NOT OTTER_DISABLE_PIC)
        set_target_properties(otter PROPERTIES POSITION_INDEPENDENT_CODE ON INTERFACE_POSITION_INDEPENDENT_CODE ON)
    endif()

    if(CMAKE_BUILD_TYPE MATCHES "(Release|RELEASE|release)")
        if(NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten" AND NOT (CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.6))
            target_compile_options(otter PRIVATE -Ofast)
        endif()

        target_compile_options(otter PRIVATE -ffast-math)
    endif()

    if(PI3)
        target_compile_options(otter PRIVATE -march=native -mfpu=neon -mfloat-abi=hard)
        target_compile_definitions(otter PRIVATE __ARM_NEON __ANDROID__)
    endif()
    # target_compile_options(otter PRIVATE -march=native)
    target_compile_options(otter PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
    if(OTTER_SHARED_LIB AND OTTER_ENABLE_LTO)
        set_target_properties(otter PROPERTIES INTERPROCEDURAL_OPTIMIZATION ON)
    endif()
endif()

if(OTTER_DISABLE_RTTI)
    target_compile_options(otter PUBLIC -fno-rtti)
endif()

if(OTTER_DISABLE_EXCEPTION)
    target_compile_options(otter PUBLIC -fno-exceptions)
endif()

if(OTTER_TARGET_ARCH STREQUAL "x86")
    if(OTTER_SSE2)
        if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC" OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_SIMULATE_ID MATCHES "MSVC" AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT MATCHES "MSVC"))
            target_compile_options(otter PRIVATE /arch:SSE2 /D__SSE2__)
        else()
            target_compile_options(otter PRIVATE -msse2 -msse)
            if(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
                target_compile_options(otter PRIVATE -msimd128)
            endif()
        endif()
    endif()

#     if(NOT OTTER_RUNTIME_CPU AND OTTER_AVX512)
#         if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC" OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_SIMULATE_ID MATCHES "MSVC" AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT MATCHES "MSVC"))
#             target_compile_options(otter PRIVATE /arch:AVX512 /D__FMA__ /D__F16C__)
#             if(OTTER_AVX512VNNI)
#                 target_compile_options(otter PRIVATE /D__AVX512VNNI__)
#             endif()
#         else()
#             target_compile_options(otter PRIVATE -mavx512f -mavx512bw -mavx512vl -mfma -mf16c)
#             if(OTTER_AVX512VNNI)
#                 target_compile_options(otter PRIVATE -mavx512vnni)
#             endif()
#         endif()
#     else
    if(NOT OTTER_RUNTIME_CPU AND OTTER_FMA)
        if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC" OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_SIMULATE_ID MATCHES "MSVC" AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT MATCHES "MSVC"))
            if(OTTER_AVX2)
                target_compile_options(otter PRIVATE /arch:AVX2 /D__FMA__)
            else()
                target_compile_options(otter PRIVATE /arch:AVX /D__FMA__)
            endif()
            if(OTTER_AVXVNNI)
                target_compile_options(otter PRIVATE /D__AVXVNNI__)
            elseif(OTTER_XOP)
                target_compile_options(otter PRIVATE /D__XOP__)
            endif()
            if(OTTER_F16C)
                target_compile_options(otter PRIVATE /D__F16C__)
            endif()
        else()
            if(OTTER_AVX2)
                target_compile_options(otter PRIVATE -mavx2 -mfma)
            else()
                target_compile_options(otter PRIVATE -mavx -mfma)
            endif()
#             if(OTTER_AVXVNNI)
#                 target_compile_options(otter PRIVATE -mavxvnni)
#             elseif(OTTER_XOP)
#                 target_compile_options(otter PRIVATE -mxop)
#             endif()
            if(OTTER_F16C)
                target_compile_options(otter PRIVATE -mf16c)
            endif()
        endif()
    elseif(NOT OTTER_RUNTIME_CPU AND OTTER_AVX)
        if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC" OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_SIMULATE_ID MATCHES "MSVC" AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT MATCHES "MSVC"))
            target_compile_options(otter PRIVATE /arch:AVX)
            if(OTTER_XOP)
                target_compile_options(otter PRIVATE /D__XOP__)
            endif()
            if(OTTER_F16C)
                target_compile_options(otter PRIVATE /D__F16C__)
            endif()
        else()
            target_compile_options(otter PRIVATE -mavx)
#             if(OTTER_XOP)
#                 target_compile_options(otter PRIVATE -mxop)
#             endif()
            if(OTTER_F16C)
                target_compile_options(otter PRIVATE -mf16c)
            endif()
        endif()
    endif()
endif()

if(((IOS AND CMAKE_OSX_ARCHITECTURES MATCHES "arm64") OR (APPLE AND CMAKE_OSX_ARCHITECTURES MATCHES "arm64") OR (CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)")))
    if(NOT OTTER_RUNTIME_CPU AND OTTER_ARM82)
        if(OTTER_ARM82DOT)
            target_compile_options(otter PRIVATE -march=armv8.2-a+fp16+dotprod)
        else()
            target_compile_options(otter PRIVATE -march=armv8.2-a+fp16)
        endif()
    endif()
endif()

if(OTTER_TARGET_ARCH STREQUAL "mips")
    if(NOT OTTER_RUNTIME_CPU AND OTTER_MSA)
        target_compile_options(otter PRIVATE -mmsa)
    endif()
    if(NOT OTTER_RUNTIME_CPU AND OTTER_MMI)
        target_compile_options(otter PRIVATE -mloongson-mmi)
    endif()
endif()

if(OTTER_TARGET_ARCH STREQUAL "riscv" AND NOT C906)
    if(NOT OTTER_RUNTIME_CPU AND OTTER_RVV)
        if(OTTER_COMPILER_SUPPORT_RVV_FP16)
            target_compile_options(otter PRIVATE -march=rv64gcv_zfh)
        elseif(OTTER_COMPILER_SUPPORT_RVV)
            target_compile_options(otter PRIVATE -march=rv64gcv)
        endif()
    endif()
endif()

if(OTTER_INSTALL_SDK)
    install(TARGETS otter EXPORT otter
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
    )
    install(FILES
        Activation.hpp
        ConcatLayer.hpp
        flat_hash_map.hpp
        ConvolutionMM2DNeon.hpp
        TensorShape.hpp
        ThreadPool.hpp
        TensorAccessor.hpp
        PoseEstimation.hpp
        ExclusivelyOwned.hpp
        DilatedConvolutionUtils.hpp
        VecBase.hpp
        ConvolutionMM2D.hpp
        WarpDimMinimal.hpp
        Tensor.hpp
        TensorIteratorDynamicCasting.hpp
        Dispatch.hpp
        TensorInterpolation.hpp
        TensorCopy.hpp
        DeconvolutionLayer.hpp
        DistributionsHelper.hpp
        CPUGenerator.hpp
        TensorPixel.hpp
        ConvolutionMM2DTranspose.hpp
        Relu6Layer.hpp
        TypeCast.hpp
        CPUCachingAllocator.hpp
        CPUProfilingAllocator.hpp
        TensorFunction.hpp
        MT19937.hpp
        KalmanFilter.hpp
        TensorOptions.hpp
        Avx_Math.hpp
        ConvolutionMM2DX86.hpp
        DefaultColor.hpp
        MemoryFormat.hpp
        UpSampleKernel.hpp
        Convolution.hpp
        MaybeOwned.hpp
        CropLayer.hpp
        TensorCatKernel.hpp
        OTensor.hpp
        EmptyTensor.hpp
        Formatting.hpp
        TensorConversion.hpp
        ParallelNative.hpp
        VecFunctional.hpp
        BinaryOpsKernel.hpp
        ParamDict.hpp
        Vision.hpp
        TensorBase.hpp
        GraphicAPI.hpp
        Drawing.hpp
        DilatedConvolution.hpp
        ShortCutLayer.hpp
        Parallel-inline.hpp
        ChannelShuffleKernel.hpp
        ImageThreshold.hpp
        ReluLayer.hpp
        C++17.hpp
        AutoBuffer.hpp
        TensorDistributionTemplate.hpp
        TensorMaker.hpp
        Vec256_float.hpp
        NetOption.hpp
        Vec256.hpp
        DropoutLayer.hpp
        RangeFactoryKernel.hpp
        UpsampleLayer.hpp
        Allocator.hpp
        Config.hpp
        ScalarType.hpp
        RangeFactory.hpp
        TensorCompare.hpp
        TypeMeta.hpp
        FillKernel.hpp
        DataReader.hpp
        MaxPoolKernel.hpp
        DepthwiseConvKernelNeon.hpp
        BoxPrediction.hpp
        VecIntrinsic.hpp
        Function_Trait.hpp
        TensorProperties.hpp
        Stabilizer.hpp
        hershey_fonts.hpp
        TensorLinearAlgebra.hpp
        Padding.hpp
        SmallVector.hpp
        PermuteLayer.hpp
        LayerDeclaration.hpp
        UnaryOpsKernel.hpp
        TensorDistribution.hpp
        SplitLayer.hpp
        DispatchStub.hpp
        Pool.hpp
        Transformation.hpp
        Accumulator.hpp
        Module.hpp
        RefPtr.hpp
        SmallBuffer.hpp
        CPUAllocator.hpp
        Vec256_float_neon.hpp
        TensorBlas.hpp
        Blob.hpp
        TensorIterator.hpp
        Math.hpp
        ExpandUtils.hpp
        LineIterator.hpp
        Scalar.hpp
        Layer.hpp
        TensorTransform.hpp
        Fill.hpp
        SigmoidLayer.hpp
        LayerRegistry.hpp
        TensorCat.hpp
        Hungarian.hpp
        IsContiguous.hpp
        Observer.hpp
        TensorFactory.hpp
        DepthwiseConvTransposeKernelNeon.hpp
        Unfold2DKernel.hpp
        Benchmark.hpp
        Clock.hpp
        UnaryOps.hpp
        Device.hpp
        StringUtils.hpp
        KalmanTracker.hpp
        neon_mathfun.hpp
        ArrayRef.hpp
        ColorConvert.hpp
        TensorScalar.hpp
        BatchNormalizationLayer.hpp
        ParallelOpenMP.hpp
        ChannelShuffle.hpp
        Net.hpp
        NanodetPlusDetectionOutputLayer.hpp
        DefaultDtype.hpp
        SliceLayer.hpp
        TensorSoftmaxKernel.hpp
        ConvolutionUtils.hpp
        TensorDistributionKernel.hpp
        TensorCopyKernel.hpp
        TensorUtils.hpp
        Generator.hpp
        MemoryOverlap.hpp
        Loop.hpp
        BatchNormalization.hpp
        Normalization.hpp
        TensorBlasKernel.hpp
        InputLayer.hpp
        ColorMap.hpp
        DrawDetection.hpp
        Dropout.hpp
        LReluLayer.hpp
        EltwiseLayer.hpp
        Memory.hpp
        in_place.hpp
        Initializer.hpp
        ActivationKernel.hpp
        Yolov3DetectionOutputLayer.hpp
        GeneratorNucleus.hpp
        BatchNormalizationKernel.hpp
        im2col.hpp
        ReshapeLayer.hpp
        ScalarOps.hpp
        Utils.hpp
        BinaryOps.hpp
        TensorCompareKernel.hpp
        DepthwiseConvKernelX86.hpp
        Unfold2D.hpp
        MaxPoolLayer.hpp
        WarpDimUtils.hpp
        TensorSoftmax.hpp
        Macro.hpp
        TensorResize.hpp
        vol2col.hpp
        Vec.hpp
        Parallel.hpp
        TensorOperator.hpp
        TypeProperties.hpp
        UpSample.hpp
        FunctionRef.hpp
        Interpreter.hpp
        ConvolutionLayer.hpp
        ConvolutionMM2DTransposeNeon.hpp
        Exception.hpp
        PerspectiveView.hpp
        ChannelShuffleLayer.hpp
        LineDetection.hpp
        Otter.hpp
        ${CMAKE_CURRENT_BINARY_DIR}/platform.hpp
        DESTINATION include/otter
    )
    install(EXPORT otter DESTINATION lib/cmake/otter)
    configure_file(${CMAKE_CURRENT_LIST_DIR}/../cmake/otterConfig.cmake.in otterConfig.cmake @ONLY)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/otterConfig.cmake DESTINATION lib/cmake/otter)
endif()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(TARGET otter PROPERTY FOLDER "libotter")
