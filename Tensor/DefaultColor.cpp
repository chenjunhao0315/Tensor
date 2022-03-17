//
//  DefaultColor.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/17.
//

#include "DefaultColor.hpp"

namespace otter {
namespace cv {

Color getDefaultColor(int color) {
    switch (color) {
        case MAROON:
            return { 128,   0,   0}; break;
        case DARK_RED:
            return { 139,   0,   0}; break;
        case BROWN:
            return { 165,  42,  42}; break;
        case FIREBRICK:
            return { 178,  34,  34}; break;
        case CRIMSON:
            return { 220,  20,  60}; break;
        case RED:
            return { 255,   0,   0}; break;
        case TOMATO:
            return { 255,  99,  71}; break;
        case CORAL:
            return { 255, 127,  80}; break;
        case INDIAN_RED:
            return { 205,  92,  92}; break;
        case LIGHT_CORAL:
            return { 240, 128, 128}; break;
        case DARK_SALMON:
            return { 233, 150, 122}; break;
        case SALMON:
            return { 250, 128, 114}; break;
        case LIGHT_SALMON:
            return { 255, 160, 122}; break;
        case ORANGE_RED:
            return { 255,  69,   0}; break;
        case DARK_ORANGE:
            return { 255, 140,   0}; break;
        case ORANGE:
            return { 255, 165,   0}; break;
        case GOLD:
            return { 255, 215,   0}; break;
        case DARK_GOLDEN_ROD:
            return { 184, 134,  11}; break;
        case GOLDEN_ROD:
            return { 218, 165,  32}; break;
        case PALE_GOLDEN_ROD:
            return { 238, 232, 170}; break;
        case DARK_KHAKI:
            return { 189, 183, 107}; break;
        case KHAKI:
            return { 240, 230, 140}; break;
        case OLIVE:
            return { 128, 128,   0}; break;
        case YELLOW:
            return { 255, 255,   0}; break;
        case YELLOW_GREEN:
            return { 154, 205,  50}; break;
        case DARK_OLIVE_GREEN:
            return {  85, 107,  47}; break;
        case OLIVE_DRAB:
            return { 107, 142,  35}; break;
        case LAWN_GREEN:
            return { 124, 252,   0}; break;
        case CHARTREUSE:
            return { 127, 255,   0}; break;
        case GREEN_YELLOW:
            return { 173, 255,  47}; break;
        case DARK_GREEN:
            return {   0, 100,   0}; break;
        case GREEN:
            return {   0, 128,   0}; break;
        case FOREST_GREEN:
            return {  34, 139,  34}; break;
        case LIME:
            return {   0, 255,   0}; break;
        case LIME_GREEN:
            return {  50, 205,  50}; break;
        case LIGHT_GREEN:
            return { 144, 238, 144}; break;
        case PALE_GREEN:
            return { 152, 251, 152}; break;
        case DARK_SEA_GREEN:
            return { 143, 188, 143}; break;
        case MEDIUM_SPRING_GREEN:
            return {   0, 250, 154}; break;
        case SPRING_GREEN:
            return {   0, 255, 127}; break;
        case SEA_GREEN:
            return {  46, 139,  87}; break;
        case MEDIUM_AQUA_MARINE:
            return { 102, 205, 170}; break;
        case MEDIUM_SEA_GREEN:
            return {  60, 179, 113}; break;
        case LIGHT_SEA_GREEN:
            return {  32, 178, 170}; break;
        case DARK_SLATE_GRAY:
            return {  47,  79,  79}; break;
        case TEAL:
            return {   0, 128, 128}; break;
        case DARK_CYAN:
            return {   0, 139, 139}; break;
        case AQUA:
            return {   0, 255, 255}; break;
        case CYAN:
            return {   0, 255, 255}; break;
        case LIGHT_CYAN:
            return { 224, 255, 255}; break;
        case DARK_TURQUOISE:
            return {   0, 206, 209}; break;
        case TURQUOISE:
            return {  64, 224, 208}; break;
        case MEDIUM_TURQUOISE:
            return {  72, 209, 204}; break;
        case PALE_TURQUOISE:
            return { 175, 238, 238}; break;
        case AQUA_MARINE:
            return { 127, 255, 212}; break;
        case POWDER_BLUE:
            return { 176, 224, 230}; break;
        case CADET_BLUE:
            return {  95, 158, 160}; break;
        case STEEL_BLUE:
            return {  70, 130, 180}; break;
        case CORN_FLOWER_BLUE:
            return { 100, 149, 237}; break;
        case DEEP_SKY_BLUE:
            return {   0, 191, 255}; break;
        case DODGER_BLUE:
            return {  30, 144, 255}; break;
        case LIGHT_BLUE:
            return { 173, 216, 230}; break;
        case SKY_BLUE:
            return { 135, 206, 235}; break;
        case LIGHT_SKY_BLUE:
            return { 135, 206, 250}; break;
        case MIDNIGHT_BLUE:
            return {  25,  25, 112}; break;
        case NAVY:
            return {   0,   0, 128}; break;
        case DARK_BLUE:
            return {   0,   0, 139}; break;
        case MEDIUM_BLUE:
            return {   0,   0, 205}; break;
        case BLUE:
            return {   0,   0, 255}; break;
        case ROYAL_BLUE:
            return {  65, 105, 225}; break;
        case BLUE_VIOLET:
            return { 138,  43, 226}; break;
        case INDIGO:
            return {  75,   0, 130}; break;
        case DARK_SLATE_BLUE:
            return {  72,  61, 139}; break;
        case SLATE_BLUE:
            return { 106,  90, 205}; break;
        case MEDIUM_SLATE_BLUE:
            return { 123, 104, 238}; break;
        case MEDIUM_PURPLE:
            return { 147, 112, 219}; break;
        case DARK_MAGENTA:
            return { 139,   0, 139}; break;
        case DARK_VIOLET:
            return { 148,   0, 211}; break;
        case DARK_ORCHID:
            return { 153,  50, 204}; break;
        case MEDIUM_ORCHID:
            return { 186,  85, 211}; break;
        case PURPLE:
            return { 128,   0, 128}; break;
        case THISTLE:
            return { 216, 191, 216}; break;
        case PLUM:
            return { 221, 160, 221}; break;
        case VIOLET:
            return { 238, 130, 238}; break;
        case MAGENTA:
            return { 255,   0, 255}; break;
        case FUCHSIA:
            return { 255,   0, 255}; break;
        case ORCHID:
            return { 218, 112, 214}; break;
        case MEDIUM_VIOLET_RED:
            return { 199,  21, 133}; break;
        case PALE_VIOLET_RED:
            return { 219, 112, 147}; break;
        case DEEP_PINK:
            return { 255,  20, 147}; break;
        case HOT_PINK:
            return { 255, 105, 180}; break;
        case LIGHT_PINK:
            return { 255, 182, 193}; break;
        case PINK:
            return { 255, 192, 203}; break;
        case ANTIQUE_WHITE:
            return { 250, 235, 215}; break;
        case BEIGE:
            return { 245, 245, 220}; break;
        case BISQUE:
            return { 255, 228, 196}; break;
        case BLANCHED_ALMOND:
            return { 255, 235, 205}; break;
        case WHEAT:
            return { 245, 222, 179}; break;
        case CORN_SILK:
            return { 255, 248, 220}; break;
        case LEMON_CHIFFON:
            return { 255, 250, 205}; break;
        case LIGHT_GOLDEN_ROD_YELLOW:
            return { 250, 250, 210}; break;
        case LIGHT_YELLOW:
            return { 255, 255, 224}; break;
        case SADDLE_BROWN:
            return { 139,  69,  19}; break;
        case SIENNA:
            return { 160,  82,  45}; break;
        case CHOCOLATE:
            return { 210, 105,  30}; break;
        case PERU:
            return { 205, 133,  63}; break;
        case SANDY_BROWN:
            return { 244, 164,  96}; break;
        case BURLY_WOOD:
            return { 222, 184, 135}; break;
        case TAN:
            return { 210, 180, 140}; break;
        case ROSY_BROWN:
            return { 188, 143, 143}; break;
        case MOCCASIN:
            return { 255, 228, 181}; break;
        case NAVAJO_WHITE:
            return { 255, 222, 173}; break;
        case PEACH_PUFF:
            return { 255, 218, 185}; break;
        case MISTY_ROSE:
            return { 255, 228, 225}; break;
        case LAVENDER_BLUSH:
            return { 255, 240, 245}; break;
        case LINEN:
            return { 250, 240, 230}; break;
        case OLD_LACE:
            return { 253, 245, 230}; break;
        case PAPAYA_WHIP:
            return { 255, 239, 213}; break;
        case SEA_SHELL:
            return { 255, 245, 238}; break;
        case MINT_CREAM:
            return { 245, 255, 250}; break;
        case SLATE_GRAY:
            return { 112, 128, 144}; break;
        case LIGHT_SLATE_GRAY:
            return { 119, 136, 153}; break;
        case LIGHT_STEEL_BLUE:
            return { 176, 196, 222}; break;
        case LAVENDER:
            return { 230, 230, 250}; break;
        case FLORAL_WHITE:
            return { 255, 250, 240}; break;
        case ALICE_BLUE:
            return { 240, 248, 255}; break;
        case GHOST_WHITE:
            return { 248, 248, 255}; break;
        case HONEYDEW:
            return { 240, 255, 240}; break;
        case IVORY:
            return { 255, 255, 240}; break;
        case AZURE:
            return { 240, 255, 255}; break;
        case SNOW:
            return { 255, 250, 250}; break;
        case BLACK:
            return {   0,   0,   0}; break;
        case DIM_GRAY:
            return { 105, 105, 105}; break;
        case GRAY:
            return { 128, 128, 128}; break;
        case DARK_GRAY:
            return { 169, 169, 169}; break;
        case SILVER:
            return { 192, 192, 192}; break;
        case LIGHT_GRAY:
            return { 211, 211, 211}; break;
        case GAINSBORO:
            return { 220, 220, 220}; break;
        case WHITE_SMOKE:
            return { 245, 245, 245}; break;
        case WHITE:
            return { 255, 255, 255}; break;
        default:
            return { 255, 255, 255};
    }
}

}   // end namespace cv
}   // end namespace otter
