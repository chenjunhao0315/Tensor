//
//  OtterConverter.cpp
//  Otter
//
//  Created by 陳均豪 on 2022/03/10.
//

#include "NcnntoOtterConverter.hpp"
#include <cstdio>
#include <iostream>
#include "paramdict.h"
#include <map>
#include <sstream>
#include <string>

using namespace std;

using namespace ncnn;

#define NCNN_LOGE(...) do { \
fprintf(stderr, ##__VA_ARGS__); fprintf(stderr, "\n"); } while(0)

#define EARSE_CHARACTER(str, c) str.erase(std::remove_if(str.begin(), str.end(), [](unsigned char x) { return x == c; }), str.end());

#define EARSE_SPACE(str) EARSE_CHARACTER(str, ' ')

#define WRITE_SPACE(file, n) for (int i = n; i--; ) fprintf(file, " ");

#define SCAN_VALUE(fmt, v)              \
if (dr.scan(fmt, &v) != 1) {            \
    NCNN_LOGE("parse " #v " failed");   \
    exit(-1);                           \
}

#define ADD_PARAM_STRING(str, value)    \
    team.addParam({str, value});
    
#define ADD_PARAM(str, value)   \
    sub_team.addParam({str, std::to_string(value)});
    
#define WRITE_LAYER_NAME(str, value)    \
    team.addParam({"name", get_layer_name(str, value)})
    
#define ADD_TRANSFORM_MAP(origin, transform)    \
{int count = std::count(origin.begin(), origin.end(), ',') + 1;  \
    std::stringstream ss(origin);   \
    for (int i = 0; i < count; ++i) {   \
        std::string name;   \
        getline(ss, name, ','); \
        EARSE_SPACE(name);  \
    printf("add \"%s\" with \"%s\"\n", name.c_str(), transform.c_str());    \
    transform_map[name] = transform;    \
}}
    
#define ACTIVATION(type) \
    team.addParam({"activation", type});

map<string, string> transform_map;

namespace otter {

void ncnn2team(Otter &team, ParamDict &pd, std::string input_name, std::string output_name);

void rename_and_reconstruct(OtterLeader& graph);

int data = 1;
int conv = 1;
int deconv = 1;
int concat = 1;
int pool = 1;
int upsample = 1;
int shuffle = 1;
int crop = 1;
int shortcut = 1;
int yolo = 1;
int slice = 1;
int sigmoid = 1;
int reshape = 1;
int permute = 1;
int relu = 1;
int innerproduct = 1;
int flatten = 1;

OtterLeader ncnn2otter(const char *model_path) {
    OtterLeader new_model(model_path);
    
    FILE* fp = fopen(model_path, "rb");
    DataReaderFromStdio dr(fp);
    
    ParamDict pd;
    
    int magic = 0;
    SCAN_VALUE("%d", magic)
    if (magic != 7767517)
    {
        NCNN_LOGE("param is too old, please regenerate");
        exit(-1);
    }
    
    // parse
    int layer_count = 0;
    int blob_count = 0;
    SCAN_VALUE("%d", layer_count)
    SCAN_VALUE("%d", blob_count)
    if (layer_count <= 0 || blob_count <= 0)
    {
        NCNN_LOGE("invalid layer_count or blob_count");
        exit(-1);
    }
    
    printf("magic number: %d\n", magic);
    printf("layer count: %d\n", layer_count);
    printf("blob_count: %d\n", blob_count);
    
    for (int i = 0; i < layer_count; ++i) {
        char layer_type[256];
        char layer_name[256];
        int bottom_count = 0;
        int top_count = 0;
        SCAN_VALUE("%255s", layer_type)
        SCAN_VALUE("%255s", layer_name)
        SCAN_VALUE("%d", bottom_count)
        SCAN_VALUE("%d", top_count)
        
        Otter team(layer_type);
        
        std::string input_name;
        for (int j = 0; j < bottom_count; ++j) {
            char bottom_name[256];
            SCAN_VALUE("%255s", bottom_name)
            
            if (j)
                input_name += ", ";
            input_name += bottom_name;
        }
        
        std::string output_name;
        for (int j = 0; j < top_count; ++j) {
            char blob_name[256];
            SCAN_VALUE("%255s", blob_name)
            
            if (j)
                output_name += ", ";
            output_name += blob_name;
        }
        
        int pdlr = pd.load_param(dr);
        if (pdlr != 0)
        {
            NCNN_LOGE("ParamDict load_param %d %s failed", i, layer_name);
            continue;
        }
        
        ncnn2team(team, pd, input_name, output_name);
        
        if (team.getName() != "Split")
            new_model.addTeam(team);
    }
    
    return new_model;
}

std::string get_layer_name(std::string str, int value) {
    std::string name(str);
    name += "_" + std::to_string(value);
    
    return name;
}

void ncnn2team(Otter &team, ParamDict &pd, std::string input_name, std::string output_name) {
    
    std::string layer_name;
    std::string type = team.getName();
    Otter sub_team("Param");
    
    if (type == "Input") {
        WRITE_LAYER_NAME("data", data);
        std::string output = get_layer_name("data", data++);
        ADD_PARAM_STRING("output", output);
        ADD_TRANSFORM_MAP(output_name, output);
    } else if (type == "Split") {
        std::string bottom = transform_map[input_name];
        printf("input: %s bottom: %s\n", input_name.c_str(), bottom.c_str());
        ADD_TRANSFORM_MAP(output_name, bottom);
        
    } else if (type == "Pooling") {
        team.setName("MaxPool");
        int kernel_w = pd.get(1, 0);
        int kernel_h = pd.get(11, kernel_w);
        int stride_w = pd.get(2, 1);
        int stride_h = pd.get(12, stride_w);
        int pad_left = pd.get(3, 0);
        int pad_right = pd.get(14, pad_left);
        int pad_top = pd.get(13, pad_left);
        int pad_bottom = pd.get(15, pad_top);
        if (pad_top != pad_bottom || pad_right != pad_left) {
            printf("Pooling unsupport!\n");
            exit(-100);
        }
        ADD_PARAM("kernel_h", kernel_h);
        ADD_PARAM("kernel_w", kernel_w);
        ADD_PARAM("stride_h", stride_h);
        ADD_PARAM("stride_w", stride_w);
        ADD_PARAM("padding_h", pad_top);
        ADD_PARAM("padding_w", pad_left);
        
        std::string input = transform_map[input_name];
        std::string output = get_layer_name("pool", pool);
        ADD_TRANSFORM_MAP(output_name, output);
        WRITE_LAYER_NAME("pool", pool++);
        ADD_PARAM_STRING("input", input);
        ADD_PARAM_STRING("output", output);
        
    } else if (type == "Convolution" || type == "ConvolutionDepthWise") {
        int out_channels = pd.get(0, 0);
        int kernel_w = pd.get(1, 0);
        int kernel_h = pd.get(11, kernel_w);
        int stride_w = pd.get(3, 1);
        int stride_h = pd.get(13, stride_w);
        int dilation_w = pd.get(2, 1);
        int dilation_h = pd.get(12, dilation_w);
        int pad_left = pd.get(4, 0);
        int pad_right = pd.get(15, pad_left);
        int pad_top = pd.get(14, pad_left);
        int pad_bottom = pd.get(16, pad_top);
        int pad_value = pd.get(18, 0);
        int bias_term = pd.get(5, 0);
        int weight_data_size = pd.get(6, 0);
        int activation_type = pd.get(9, 0);
        auto activation_param = pd.get(10, std::vector<float>());
        if (pad_top != pad_bottom || pad_right != pad_left) {
            printf("Padding unsupport!\n");
            exit(-100);
        }
        ADD_PARAM("out_channels", out_channels);
        ADD_PARAM("kernel_h", kernel_h);
        ADD_PARAM("kernel_w", kernel_w);
        ADD_PARAM("stride_h", stride_h);
        ADD_PARAM("stride_w", stride_w);
        ADD_PARAM("padding_h", pad_top);
        ADD_PARAM("padding_w", pad_left);
        ADD_PARAM("dilation_h", dilation_h);
        ADD_PARAM("dilation_w", dilation_w);
        (bias_term) ? sub_team.addParam({"bias_term", "true"}) : sub_team.addParam({"bias_term", "false"});
        
        if (type == "ConvolutionDepthWise") {
            team.setName("Convolution");
            int groups = pd.get(7, 1);
            ADD_PARAM("groups", groups);
        }
        
        if (activation_type == 1) {
            ACTIVATION("Relu");
        } else if (activation_type == 2) {
            ACTIVATION("LRelu");
            sub_team.addParam({"alpha", std::to_string(activation_param[0])});
        } else if (activation_type == 3) {
            // Suppose it is relu 6
            if (activation_param[0] == 0 && activation_param[1] == 6) {
                ACTIVATION("Relu6");
            } else {
                printf("Unsupport activation!\n");
                exit(-100);
            }
        } else if (activation_type == 4) {
            ACTIVATION("Sigmoid");
        }
        
        std::string input = transform_map[input_name];
        std::string output = get_layer_name("conv", conv);
        ADD_TRANSFORM_MAP(output_name, output);
        WRITE_LAYER_NAME("conv", conv++);
        ADD_PARAM_STRING("input", input);
        ADD_PARAM_STRING("output", output);
        
    } else if (type == "Deconvolution" || type == "DeconvolutionDepthWise") {
        int out_channels = pd.get(0, 0);
        int kernel_w = pd.get(1, 0);
        int kernel_h = pd.get(11, kernel_w);
        int stride_w = pd.get(3, 1);
        int stride_h = pd.get(13, stride_w);
        int dilation_w = pd.get(2, 1);
        int dilation_h = pd.get(12, dilation_w);
        int pad_left = pd.get(4, 0);
        int pad_right = pd.get(15, pad_left);
        int pad_top = pd.get(14, pad_left);
        int pad_bottom = pd.get(16, pad_top);
        int output_pad_right = pd.get(18, 0);
        int output_pad_bottom = pd.get(19, output_pad_right);
        int output_w = pd.get(20, 0);
        int output_h = pd.get(21, output_w);
        int bias_term = pd.get(5, 0);
        int weight_data_size = pd.get(6, 0);
        int activation_type = pd.get(9, 0);
        auto activation_param = pd.get(10, std::vector<float>());
        if (pad_top != pad_bottom || pad_right != pad_left) {
            printf("Padding unsupport!\n");
            exit(-100);
        }
        
        ADD_PARAM("out_channels", out_channels);
        ADD_PARAM("kernel_h", kernel_h);
        ADD_PARAM("kernel_w", kernel_w);
        ADD_PARAM("stride_h", stride_h);
        ADD_PARAM("stride_w", stride_w);
        ADD_PARAM("padding_h", pad_top);
        ADD_PARAM("padding_w", pad_left);
        ADD_PARAM("dilation_h", dilation_h);
        ADD_PARAM("dilation_w", dilation_w);
        ADD_PARAM("output_padding_h", output_pad_bottom);
        ADD_PARAM("output_padding_w", output_pad_right);
        (bias_term) ? sub_team.addParam({"bias_term", "true"}) : sub_team.addParam({"bias_term", "false"});
        
        if (type == "DeconvolutionDepthWise") {
            team.setName("Deconvolution");
            int groups = pd.get(7, 1);
            ADD_PARAM("groups", groups);
        }
        
        if (activation_type == 1) {
            ACTIVATION("Relu");
        } else if (activation_type == 2) {
            ACTIVATION("LRelu");
            sub_team.addParam({"alpha", std::to_string(activation_param[0])});
        } else if (activation_type == 3) {
            // Suppose it is relu 6
            if (activation_param[0] == 0 && activation_param[1] == 6) {
                ACTIVATION("Relu6");
            } else {
                printf("Unsupport activation!\n");
                exit(-100);
            }
        } else if (activation_type == 4) {
            ACTIVATION("Sigmoid");
        }
        
        std::string input = transform_map[input_name];
        std::string output = get_layer_name("deconv", deconv);
        ADD_TRANSFORM_MAP(output_name, output);
        WRITE_LAYER_NAME("deconv", deconv++);
        ADD_PARAM_STRING("input", input);
        ADD_PARAM_STRING("output", output);
        
    } else if (type == "InnerProduct") {
        int num_output = pd.get(0, 0);
        int bias_term = pd.get(1, 0);
        int weight_data_size = pd.get(2, 0);
        int int8_scale_term = pd.get(8, 0);
        int activation_type = pd.get(9, 0);
        auto activation_param = pd.get(10, std::vector<float>());
        
        ADD_PARAM("out_features", num_output);
        (bias_term) ? sub_team.addParam({"bias_term", "true"}) : sub_team.addParam({"bias_term", "false"});
        
        if (activation_type == 1) {
            ACTIVATION("Relu");
        } else if (activation_type == 2) {
            ACTIVATION("LRelu");
            sub_team.addParam({"alpha", std::to_string(activation_param[0])});
        } else if (activation_type == 3) {
            // Suppose it is relu 6
            if (activation_param[0] == 0 && activation_param[1] == 6) {
                ACTIVATION("Relu6");
            } else {
                printf("Unsupport activation!\n");
                exit(-100);
            }
        } else if (activation_type == 4) {
            ACTIVATION("Sigmoid");
        }
        
        std::string input = transform_map[input_name];
        std::string output = get_layer_name("linear", innerproduct);
        ADD_TRANSFORM_MAP(output_name, output);
        WRITE_LAYER_NAME("linear", innerproduct++);
        ADD_PARAM_STRING("input", input);
        ADD_PARAM_STRING("output", output);
        
    } else if (type == "Concat") {
        int axis = pd.get(0, 1);
        ADD_PARAM("axis", axis);
        int input_count = std::count(input_name.begin(), input_name.end(), ',') + 1;
        
        std::string input;
        std::string output;
        
        std::stringstream ss(input_name);
        for (int i = 0; i < input_count; ++i) {
            std::string name;
            getline(ss, name, ',');
            EARSE_SPACE(name);
            if (i)
                input += ", ";
            input += transform_map[name];
        }
        output = get_layer_name("concat", concat);
        ADD_TRANSFORM_MAP(output_name, output);
        WRITE_LAYER_NAME("concat", concat++);
        ADD_PARAM_STRING("input", input);
        ADD_PARAM_STRING("output", output);
        
        
    } else if (type == "ShuffleChannel") {
        team.setName("ChannelShuffle");
        int groups = pd.get(0, 1);
        ADD_PARAM("groups", groups);
        
        std::string input = transform_map[input_name];
        std::string output = get_layer_name("shuffle", shuffle);
        ADD_TRANSFORM_MAP(output_name, output);
        WRITE_LAYER_NAME("shuffle", shuffle++);
        ADD_PARAM_STRING("input", input);
        ADD_PARAM_STRING("output", output);
        
    } else if (type == "Crop") {
        int woffset = pd.get(0, 0);
        int hoffset = pd.get(1, 0);
        int coffset = pd.get(2, 0);
        int outw = pd.get(3, 0);
        int outh = pd.get(4, 0);
        int outc = pd.get(5, 0);
        int woffset2 = pd.get(6, 0);
        int hoffset2 = pd.get(7, 0);
        int coffset2 = pd.get(8, 0);
        auto start = pd.get(9, std::vector<float>());
        auto end = pd.get(10, std::vector<float>());
        auto axis = pd.get(11, std::vector<float>());
        if (woffset || hoffset || coffset || outw || outh || outc || woffset2 || hoffset2 || coffset2) {
            printf("Crop unsupports!\n");
            exit(-100);
        }
        team.setName("Crop");
        ADD_PARAM("start", (int)start[0]);
        ADD_PARAM("end", (int)end[0]);
        ADD_PARAM("axis", (int)axis[0] + 1);
        
        std::string input = transform_map[input_name];
        std::string output = get_layer_name("crop", crop);
        ADD_TRANSFORM_MAP(output_name, output);
        WRITE_LAYER_NAME("crop", crop++);
        ADD_PARAM_STRING("input", input);
        ADD_PARAM_STRING("output", output);
        
    } else if (type == "Slice") {
        int axis = pd.get(1, 0);
        auto slice_v = pd.get(0, std::vector<float>());

        std::string slice_param;
        for (int i = 0; i < slice_v.size(); ++i) {
            if (i)
                slice_param += ", ";
            int value = (slice_v[i] == -233) ? -1 : slice_v[i];
            slice_param += std::to_string(value);
        }

        axis += 1; // ncnn is CHW otter is NCHW

        team.setName("Slice");
        ADD_PARAM("axis", axis);
        sub_team.addParam({"slice", slice_param});

        std::string input = transform_map[input_name];
        ADD_PARAM_STRING("input", input);

        int output_count = std::count(output_name.begin(), output_name.end(), ',') + 1;
        std::stringstream ss(output_name);
        std::string output;
        for (int i = 0; i < output_count; ++i) {
            std::string transform = "slice_" + to_string(slice) + "_" + to_string(i);
            if (i)
                output += ", ";
            output += transform;
            std::string name;
            getline(ss, name, ',');
            EARSE_SPACE(name);
            printf("add \"%s\" with \"%s\"\n", name.c_str(), transform.c_str());
            transform_map[name] = transform;
        }
        WRITE_LAYER_NAME("slice", slice++);
        ADD_PARAM_STRING("output", output);

    } else if (type == "Sigmoid") {
        std::string input = transform_map[input_name];
        std::string output = get_layer_name("sigmoid", sigmoid);
        ADD_TRANSFORM_MAP(output_name, output);
        WRITE_LAYER_NAME("sigmoid", sigmoid++);
        ADD_PARAM_STRING("input", input);
        ADD_PARAM_STRING("output", output);
    } else if (type == "Flatten") {
        std::string input = transform_map[input_name];
        std::string output = get_layer_name("flatten", flatten);
        ADD_TRANSFORM_MAP(output_name, output);
        WRITE_LAYER_NAME("flatten", flatten++);
        ADD_PARAM_STRING("input", input);
        ADD_PARAM_STRING("output", output);
    } else if (type == "ReLU") {
        team.setName("Relu");
        printf("relu input name: %s\n", input_name.c_str());
        std::string input = transform_map[input_name];
        std::string output = get_layer_name("relu", relu);
        ADD_TRANSFORM_MAP(output_name, output);
        WRITE_LAYER_NAME("relu", relu++);
        ADD_PARAM_STRING("input", input);
        ADD_PARAM_STRING("output", output);
    }  else if (type == "Reshape") {
        int w = pd.get(0, -233);
        int h = pd.get(1, -233);
        int c = pd.get(2, -233);

        w = w == -233 ? 1 : w;
        h = h == -233 ? 1 : h;
        c = c == -233 ? 1 : c;
        std::string shape = "1, " + std::to_string(c) + ", " + std::to_string(h) + ", " + std::to_string(w);

        sub_team.addParam({"reshape", shape});

        printf("reshape input name: %s\n", input_name.c_str());

        std::string input = transform_map[input_name];
        printf("reshape get transform input name: %s\n", input.c_str());

        std::string output = get_layer_name("reshape", reshape);
        ADD_TRANSFORM_MAP(output_name, output);
        WRITE_LAYER_NAME("reshape", reshape++);
        ADD_PARAM_STRING("input", input);
        ADD_PARAM_STRING("output", output);
    } else if (type == "Permute") {
        int order_type = pd.get(0, 0);
        // order_type
        // 0 = w h
        // 1 = h w

        // order_type
        // 0 = w h c
        // 1 = h w c
        // 2 = w c h
        // 3 = c w h
        // 4 = h c w
        // 5 = c h w

        // order_type
        // 0 = w h d c
        // 1 = h w d c
        // 2 = w d h c
        // 3 = d w h c
        // 4 = h d w c
        // 5 = d h w c
        // 6 = w h c d
        // 7 = h w c d
        // 8 = w c h d
        // 9 = c w h d
        //10 = h c w d
        //11 = c h w d
        //12 = w d c h
        //13 = d w c h
        //14 = w c d h
        //15 = c w d h
        //16 = d c w h
        //17 = c d w h
        //18 = h d c w
        //19 = d h c w
        //20 = h c d w
        //21 = c h d w
        //22 = d c h w
        //23 = c d h w

        std::string input = transform_map[input_name];
        std::string output = get_layer_name("permute", permute);
        ADD_TRANSFORM_MAP(output_name, output);
        WRITE_LAYER_NAME("permute", permute++);
        ADD_PARAM_STRING("input", input);
        ADD_PARAM_STRING("output", output);
    } else if (type == "Interp") {
        team.setName("Upsample");
        int resize_type = pd.get(0, 0);
        float scale_height = pd.get(1, 1.f);
        float scale_width = pd.get(2, 1.f);
        int output_height = pd.get(3, 0);
        int output_width = pd.get(4, 0);
        int align_corner = pd.get(6, 0);
        if (scale_height != scale_width) {
            printf("Interp unsupport!\n");
            exit(-100);
        }
        ADD_PARAM("stride", scale_width);
        (align_corner) ? sub_team.addParam({"align_corner", "true"}) : sub_team.addParam({"align_corner", "false"});
        if (resize_type == 1) {
            sub_team.addParam({"upsample_mode", "nearest"});
        } else if (resize_type == 2) {
            sub_team.addParam({"upsample_mode", "bilinear"});
        } else if (resize_type == 3) {
            sub_team.addParam({"upsample_mode", "bicubic"});
        }
        
        int input_count = std::count(input_name.begin(), input_name.end(), ',') + 1;
        std::string input;
        std::string output;
        
        std::stringstream ss(input_name);
        for (int i = 0; i < input_count; ++i) {
            std::string name;
            getline(ss, name, ',');
            EARSE_SPACE(name);
            if (i)
                input += ", ";
            input += transform_map[name];
        }
        output = get_layer_name("upsample", upsample);
        ADD_TRANSFORM_MAP(output_name, output);
        WRITE_LAYER_NAME("upsample", upsample++);
        ADD_PARAM_STRING("input", input);
        ADD_PARAM_STRING("output", output);
    } else if (type == "BinaryOp") {
        enum OperationType {
            Operation_ADD = 0,
            Operation_SUB = 1,
            Operation_MUL = 2,
            Operation_DIV = 3,
            Operation_MAX = 4,
            Operation_MIN = 5,
            Operation_POW = 6,
            Operation_RSUB = 7,
            Operation_RDIV = 8
        };
        
        int type = pd.get(0, 0);
        if (type == Operation_ADD) {
            team.setName("ShortCut");
        } else {
            printf("BinaryOp unsupport!\n");
            exit(-100);
        }
        
        int input_count = std::count(input_name.begin(), input_name.end(), ',') + 1;
        std::string input;
        std::string output;
        std::stringstream ss(input_name);
        for (int i = 0; i < input_count; ++i) {
            std::string name;
            getline(ss, name, ',');
            EARSE_SPACE(name);
            if (i)
                input += ", ";
            input += transform_map[name];
        }
        output = get_layer_name("shortcut", shortcut);
        ADD_TRANSFORM_MAP(output_name, output);
        WRITE_LAYER_NAME("shortcut", shortcut++);
        ADD_PARAM_STRING("input", input);
        ADD_PARAM_STRING("output", output);
        
    } else if (type == "Eltwise") {
        int op_type = pd.get(0, 0);
        auto coeffs = pd.get(1, std::vector<float>());
        
        enum OperationType {
            Operation_PROD = 0,
            Operation_SUM = 1,
            Operation_MAX = 2
        };
        
        if (op_type == Operation_SUM) {
            team.setName("ShortCut");
        } else {
            printf("BinaryOp unsupport!\n");
            exit(-100);
        }
        
        int input_count = std::count(input_name.begin(), input_name.end(), ',') + 1;
        std::string input;
        std::string output;
        std::stringstream ss(input_name);
        for (int i = 0; i < input_count; ++i) {
            std::string name;
            getline(ss, name, ',');
            EARSE_SPACE(name);
            if (i)
                input += ", ";
            input += transform_map[name];
        }
        output = get_layer_name("shortcut", shortcut);
        ADD_TRANSFORM_MAP(output_name, output);
        WRITE_LAYER_NAME("shortcut", shortcut++);
        ADD_PARAM_STRING("input", input);
        ADD_PARAM_STRING("output", output);
        
    } else if (type == "Yolov3DetectionOutput") {
        int num_class = pd.get(0, 80);
        int num_box = pd.get(1, 5);
        float confidence_threshold = pd.get(2, 0.01f);
        float num_threshold = pd.get(3, 0.45f);
        auto biases = pd.get(4, std::vector<float>());
        auto mask = pd.get(5, std::vector<float>());
        auto anchors_scale = pd.get(6, std::vector<float>());
        
        printf("%d %d %g %g\n", num_class, num_box, confidence_threshold, num_threshold);
        
        for (auto i : biases) {
            printf("%f ", i);
        } printf("\n");
        for (auto i : mask) {
            printf("%f ", i);
        } printf("\n");
        for (auto i : anchors_scale) {
            printf("%f ", i);
        } printf("\n");
        
        ADD_PARAM("num_class", num_class);
        ADD_PARAM("num_box", num_box);
        std::string anchor;
        for (int i = 0; i < biases.size(); i += 2) {
            if (i)
                anchor += " ";
            anchor += to_string((int)biases[i]) + "," + to_string((int)biases[i + 1]);
        }
        sub_team.addParam({"anchor", anchor});
        
        std::string input = transform_map[input_name];
        std::string output = get_layer_name("yolo", yolo);
        ADD_TRANSFORM_MAP(output_name, output);
        WRITE_LAYER_NAME("yolo", yolo++);
        ADD_PARAM_STRING("input", input);
        ADD_PARAM_STRING("output", output);
    }
    
    if (!sub_team.idle()) {
        team.addPartner(sub_team);
    }
}

}   // end namespace otter
