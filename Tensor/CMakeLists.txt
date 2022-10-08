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
        AccmulateType.hpp
        Accumulator.hpp
        Activation.hpp
        ActivationKernel.hpp
        ActivationLayer.hpp
        AffineGridGenerator.hpp
        Allocator.hpp
        ArrayRef.hpp
        AutoBuffer.hpp
        Avx_Math.hpp
        BatchNormalization.hpp
        BatchNormalizationKernel.hpp
        BatchNormalizationLayer.hpp
        Benchmark.hpp
        BinaryOps.hpp
        BinaryOpsKernel.hpp
        Blob.hpp
        BoxPrediction.hpp
        C++17.hpp
        CPUAllocator.hpp
        CPUCachingAllocator.hpp
        CPUGenerator.hpp
        CPUProfilingAllocator.hpp
        ChannelShuffle.hpp
        ChannelShuffleKernel.hpp
        ChannelShuffleLayer.hpp
        Clock.hpp
        ColorConvert.hpp
        ColorMap.hpp
        Complex.hpp
        Composer.hpp
        ConcatLayer.hpp
        Config.hpp
        Convolution.hpp
        Convolution1DLayer.hpp
        ConvolutionLayer.hpp
        ConvolutionMM2D.hpp
        ConvolutionMM2DInt8Neon.hpp
        ConvolutionMM2DInt8NeonPack.hpp
        ConvolutionMM2DInt8X86.hpp
        ConvolutionMM2DInt8X86Pack.hpp
        ConvolutionMM2DNeon.hpp
        ConvolutionMM2DNeonPack.hpp
        ConvolutionMM2DTranspose.hpp
        ConvolutionMM2DTransposeNeon.hpp
        ConvolutionMM2DX86.hpp
        ConvolutionMM2DX86Pack.hpp
        ConvolutionUtils.hpp
        CropLayer.hpp
        DataReader.hpp
        DeconvolutionLayer.hpp
        DefaultColor.hpp
        DefaultDtype.hpp
        DepthwiseConvKernelInt8NeonPack.hpp
        DepthwiseConvKernelInt8X86Pack.hpp
        DepthwiseConvKernelNeon.hpp
        DepthwiseConvKernelNeonPack.hpp
        DepthwiseConvKernelX86.hpp
        DepthwiseConvKernelX86Pack.hpp
        DepthwiseConvTransposeKernelNeon.hpp
        DepthwiseConvTransposeKernelNeonPack.hpp
        DepthwiseConvTransposeKernelX86Pack.hpp
        Device.hpp
        DilatedConvolution.hpp
        DilatedConvolutionUtils.hpp
        Dispatch.hpp
        DispatchStub.hpp
        DistributionsHelper.hpp
        DrawDetection.hpp
        Drawing.hpp
        Dropout.hpp
        DropoutLayer.hpp
        EltwiseLayer.hpp
        EmptyTensor.hpp
        Exception.hpp
        ExclusivelyOwned.hpp
        ExpandUtils.hpp
        Fill.hpp
        FillKernel.hpp
        FlattenLayer.hpp
        Formatting.hpp
        FunctionRef.hpp
        Function_Trait.hpp
        Generator.hpp
        GeneratorNucleus.hpp
        GraphicAPI.hpp
        GridSampler.hpp
        GridSamplerKernel.hpp
        HFloat-inl.hpp
        HFloat.hpp
        Hungarian.hpp
        ImageThreshold.hpp
        IndexKernelUtils.hpp
        IndexingKernel.hpp
        Initializer.hpp
        InnerProductLayer.hpp
        InputLayer.hpp
        Interpreter.hpp
        IsContiguous.hpp
        KalmanFilter.hpp
        KalmanPointTracker.hpp
        KalmanTracker.hpp
        LReluLayer.hpp
        Layer.hpp
        LayerDeclaration.hpp
        LayerRegistry.hpp
        LineDetection.hpp
        LineIterator.hpp
        Loop.hpp
        MT19937.hpp
        Macro.hpp
        Math.hpp
        MaxPoolKernel.hpp
        MaxPoolLayer.hpp
        MaybeOwned.hpp
        Memory.hpp
        MemoryFormat.hpp
        MemoryOverlap.hpp
        Module.hpp
        NanodetPlusDetectionOutputLayer.hpp
        Net.hpp
        NetOption.hpp
        Normalization.hpp
        OTensor.hpp
        Observer.hpp
        Optional.hpp
        Otter.hpp
        PackedData.hpp
        Padding.hpp
        Parallel-inline.hpp
        Parallel.hpp
        ParallelNative.hpp
        ParallelOpenMP.hpp
        ParamDict.hpp
        PermuteLayer.hpp
        PerspectiveView.hpp
        Pool.hpp
        PoseEstimation.hpp
        PoseStabilizer.hpp
        Quantize.hpp
        QuantizeNeon.hpp
        QuantizeX86.hpp
        ROIAlignLayer.hpp
        RangeFactory.hpp
        RangeFactoryKernel.hpp
        Reduce.hpp
        ReduceOps.hpp
        ReduceOpsKernel.hpp
        ReduceOpsUtils.hpp
        RefPtr.hpp
        Relu6Layer.hpp
        ReluLayer.hpp
        ReshapeLayer.hpp
        Scalar.hpp
        ScalarOps.hpp
        ScalarType.hpp
        ScatterGatherKernel.hpp
        ShortCutLayer.hpp
        SigmoidLayer.hpp
        SimpleROIAlignLayer.hpp
        SliceLayer.hpp
        SmallBuffer.hpp
        SmallVector.hpp
        Sorting.hpp
        SortingKernel.hpp
        SplitLayer.hpp
        Stabilizer.hpp
        StringUtils.hpp
        Tensor.hpp
        TensorAccessor.hpp
        TensorAdvancedIndexing.hpp
        TensorAdvancedIndexingUtils.hpp
        TensorBase.hpp
        TensorBlas.hpp
        TensorBlasKernel.hpp
        TensorCat.hpp
        TensorCatKernel.hpp
        TensorCompare.hpp
        TensorCompareKernel.hpp
        TensorConversion.hpp
        TensorCopy.hpp
        TensorCopyKernel.hpp
        TensorDistribution.hpp
        TensorDistributionKernel.hpp
        TensorDistributionTemplate.hpp
        TensorEltwise.hpp
        TensorFactory.hpp
        TensorFunction.hpp
        TensorGeometry.hpp
        TensorIndexing.hpp
        TensorInterpolation.hpp
        TensorIterator.hpp
        TensorIteratorDynamicCasting.hpp
        TensorLinearAlgebra.hpp
        TensorMaker.hpp
        TensorOperator.hpp
        TensorOptions.hpp
        TensorPacking.hpp
        TensorPixel.hpp
        TensorProperties.hpp
        TensorResize.hpp
        TensorScalar.hpp
        TensorShape.hpp
        TensorSoftmax.hpp
        TensorSoftmaxKernel.hpp
        TensorSpectral.hpp
        TensorTransform.hpp
        TensorUtils.hpp
        ThreadPool.hpp
        Transformation.hpp
        TypeCast.hpp
        TypeMeta.hpp
        TypeProperties.hpp
        TypeSafeSignMath.hpp
        UnaryOps.hpp
        UnaryOpsKernel.hpp
        Unfold2D.hpp
        Unfold2DKernel.hpp
        UpSample.hpp
        UpSampleKernel.hpp
        UpsampleLayer.hpp
        Utils.hpp
        Vec.hpp
        Vec256.hpp
        Vec256_float.hpp
        Vec256_float_neon.hpp
        VecBase.hpp
        VecFunctional.hpp
        VecIntrinsic.hpp
        Vision.hpp
        WarpDimMinimal.hpp
        WarpDimUtils.hpp
        Yolov3DetectionOutputLayer.hpp
        flat_hash_map.hpp
        hershey_fonts.hpp
        im2col.hpp
        in_place.hpp
        neon_mathfun.hpp
        sse_mathfun.hpp
        vol2col.hpp
        zmath.hpp
        ${CMAKE_CURRENT_BINARY_DIR}/platform.hpp
        DESTINATION include/otter
    )
    install(EXPORT otter DESTINATION lib/cmake/otter)
    configure_file(${CMAKE_CURRENT_LIST_DIR}/../cmake/otterConfig.cmake.in otterConfig.cmake @ONLY)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/otterConfig.cmake DESTINATION lib/cmake/otter)
endif()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(TARGET otter PROPERTY FOLDER "libotter")
