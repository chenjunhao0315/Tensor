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
    message("runtime: ${OTTER_RUNTIME_CPU} arm82: ${OTTER_ARM82} arm82dot: ${OTTER_ARM82DOT}")
    if(NOT OTTER_RUNTIME_CPU AND OTTER_ARM82)
        if(OTTER_ARM82DOT)
            message("compile dotprod")
            target_compile_options(otter PRIVATE -march=armv8.2-a+fp16+dotprod)
        else()
            message("compile not dotprod")
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
        vol2col.hpp
        sse_mathfun.hpp
        neon_mathfun.hpp
        in_place.hpp
        im2col.hpp
        hershey_fonts.hpp
        flat_hash_map.hpp
        Yolov3DetectionOutputLayer.hpp
        WarpDimUtils.hpp
        WarpDimMinimal.hpp
        Vision.hpp
        VecIntrinsic.hpp
        VecFunctional.hpp
        VecBase.hpp
        Vec256_float_neon.hpp
        Vec256_float.hpp
        Vec256.hpp
        Vec.hpp
        Utils.hpp
        UpsampleLayer.hpp
        UpSampleKernel.hpp
        UpSample.hpp
        Unfold2DKernel.hpp
        Unfold2D.hpp
        UnaryOpsKernel.hpp
        UnaryOps.hpp
        TypeProperties.hpp
        TypeMeta.hpp
        TypeCast.hpp
        Transformation.hpp
        ThreadPool.hpp
        TensorUtils.hpp
        TensorTransform.hpp
        TensorSoftmaxKernel.hpp
        TensorSoftmax.hpp
        TensorShape.hpp
        TensorScalar.hpp
        TensorResize.hpp
        TensorProperties.hpp
        TensorPixel.hpp
        TensorPacking.hpp
        TensorOptions.hpp
        TensorOperator.hpp
        TensorMaker.hpp
        TensorLinearAlgebra.hpp
        TensorIteratorDynamicCasting.hpp
        TensorIterator.hpp
        TensorInterpolation.hpp
        TensorFunction.hpp
        TensorFactory.hpp
        TensorEltwise.hpp
        TensorDistributionTemplate.hpp
        TensorDistributionKernel.hpp
        TensorDistribution.hpp
        TensorCopyKernel.hpp
        TensorCopy.hpp
        TensorConversion.hpp
        TensorCompareKernel.hpp
        TensorCompare.hpp
        TensorCatKernel.hpp
        TensorCat.hpp
        TensorBlasKernel.hpp
        TensorBlas.hpp
        TensorBase.hpp
        TensorAccessor.hpp
        Tensor.hpp
        StringUtils.hpp
        Stabilizer.hpp
        SplitLayer.hpp
        SmallVector.hpp
        SmallBuffer.hpp
        SliceLayer.hpp
        SigmoidLayer.hpp
        ShortCutLayer.hpp
        ScalarType.hpp
        ScalarOps.hpp
        Scalar.hpp
        ReshapeLayer.hpp
        ReluLayer.hpp
        Relu6Layer.hpp
        RefPtr.hpp
        RangeFactoryKernel.hpp
        RangeFactory.hpp
        QuantizeX86.hpp
        QuantizeNeon.hpp
        Quantize.hpp
        PoseEstimation.hpp
        Pool.hpp
        PerspectiveView.hpp
        PermuteLayer.hpp
        ParamDict.hpp
        ParallelOpenMP.hpp
        ParallelNative.hpp
        Parallel.hpp
        Parallel-inline.hpp
        Padding.hpp
        PackedData.hpp
        Otter.hpp
        Observer.hpp
        OTensor.hpp
        Normalization.hpp
        NetOption.hpp
        Net.hpp
        NanodetPlusDetectionOutputLayer.hpp
        Module.hpp
        MemoryOverlap.hpp
        MemoryFormat.hpp
        Memory.hpp
        MaybeOwned.hpp
        MaxPoolLayer.hpp
        MaxPoolKernel.hpp
        Math.hpp
        Macro.hpp
        MT19937.hpp
        Loop.hpp
        LineIterator.hpp
        LineDetection.hpp
        LayerRegistry.hpp
        LayerDeclaration.hpp
        Layer.hpp
        LReluLayer.hpp
        KalmanTracker.hpp
        KalmanFilter.hpp
        IsContiguous.hpp
        Interpreter.hpp
        InputLayer.hpp
        Initializer.hpp
        ImageThreshold.hpp
        Hungarian.hpp
        GraphicAPI.hpp
        GeneratorNucleus.hpp
        Generator.hpp
        Function_Trait.hpp
        FunctionRef.hpp
        Formatting.hpp
        FillKernel.hpp
        Fill.hpp
        ExpandUtils.hpp
        ExclusivelyOwned.hpp
        Exception.hpp
        EmptyTensor.hpp
        EltwiseLayer.hpp
        DropoutLayer.hpp
        Dropout.hpp
        Drawing.hpp
        DrawDetection.hpp
        DistributionsHelper.hpp
        DispatchStub.hpp
        Dispatch.hpp
        DilatedConvolutionUtils.hpp
        DilatedConvolution.hpp
        Device.hpp
        DepthwiseConvTransposeKernelNeon.hpp
        DepthwiseConvKernelX86Pack.hpp
        DepthwiseConvKernelX86.hpp
        DepthwiseConvKernelNeonPack.hpp
        DepthwiseConvKernelNeon.hpp
        DepthwiseConvKernelInt8X86Pack.hpp
        DefaultDtype.hpp
        DefaultColor.hpp
        DeconvolutionLayer.hpp
        DataReader.hpp
        CropLayer.hpp
        ConvolutionUtils.hpp
        ConvolutionMM2DX86Pack.hpp
        ConvolutionMM2DX86.hpp
        ConvolutionMM2DTransposeNeon.hpp
        ConvolutionMM2DTranspose.hpp
        ConvolutionMM2DNeonPack.hpp
        ConvolutionMM2DNeon.hpp
        ConvolutionMM2DInt8X86Pack.hpp
        ConvolutionMM2DInt8X86.hpp
        ConvolutionMM2DInt8Neon.hpp
        ConvolutionMM2D.hpp
        ConvolutionLayer.hpp
        Convolution.hpp
        Config.hpp
        ConcatLayer.hpp
        ColorMap.hpp
        ColorConvert.hpp
        Clock.hpp
        ChannelShuffleLayer.hpp
        ChannelShuffleKernel.hpp
        ChannelShuffle.hpp
        CPUProfilingAllocator.hpp
        CPUGenerator.hpp
        CPUCachingAllocator.hpp
        CPUAllocator.hpp
        C++17.hpp
        BoxPrediction.hpp
        Blob.hpp
        BinaryOpsKernel.hpp
        BinaryOps.hpp
        Benchmark.hpp
        BatchNormalizationLayer.hpp
        BatchNormalizationKernel.hpp
        BatchNormalization.hpp
        Avx_Math.hpp
        AutoBuffer.hpp
        ArrayRef.hpp
        Allocator.hpp
        ActivationLayer.hpp
        ActivationKernel.hpp
        Activation.hpp
        Accumulator.hpp
        ${CMAKE_CURRENT_BINARY_DIR}/platform.hpp
        DESTINATION include/otter
    )
    install(EXPORT otter DESTINATION lib/cmake/otter)
    configure_file(${CMAKE_CURRENT_LIST_DIR}/../cmake/otterConfig.cmake.in otterConfig.cmake @ONLY)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/otterConfig.cmake DESTINATION lib/cmake/otter)
endif()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(TARGET otter PROPERTY FOLDER "libotter")
