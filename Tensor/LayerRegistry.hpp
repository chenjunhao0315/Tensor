//
//  LayerRegistry.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/20.
//

#ifndef LayerRegistry_hpp
#define LayerRegistry_hpp

#include "Layer.hpp"

#include <map>

namespace otter {

class LayerRegistry {
public:
    typedef Layer* (*Creator)();
    typedef std::map<std::string, Creator> CreatorRegistry;

    static CreatorRegistry &Registry() {
        static auto *g_registry_ = new CreatorRegistry();
        return *g_registry_;
    }

    static void AddCreator(const std::string &type, Creator creator) {
        CreatorRegistry &registry = Registry();
        if (registry.count(type) == 1) {
            std::cout << "Layer type " << type << " already registered."<< std::endl;
        }
        registry[type] = creator;
    }

    static Layer* CreateLayer(std::string type) {
        CreatorRegistry &registry = Registry();
        if (registry.count(type) == 0) {
            std::cout << "Unknown layer type: " << type << " (known layer types: " << TypeListString() << ")" << std::endl;
            exit(100);
        }
        return registry[type]();
    }

    static std::vector<std::string> TypeList() {
        CreatorRegistry &registry = Registry();
        std::vector<std::string> types;
        for (typename CreatorRegistry::iterator iter = registry.begin();
             iter != registry.end(); ++iter) {
            types.push_back(iter->first);
        }
        return types;
    }

private:
    LayerRegistry() {}

    static std::string TypeListString() {
        std::vector<std::string> types = TypeList();
        std::string types_str;
        for (auto iter = types.begin();
             iter != types.end(); ++iter) {
            if (iter != types.begin()) {
                types_str += ", ";
            }
            types_str += *iter;
        }
        return types_str;
    }
};

class LayerRegister {
public:
    LayerRegister(const std::string &type, Layer* (*creator)()) {
        LayerRegistry::AddCreator(type, creator);
    }
};

#define REGISTER_LAYER_CREATOR(type, creator)   \
    static LayerRegister g_creator_##type(#type, creator)

#define REGISTER_LAYER_CLASS(type)  \
    Layer* Creator_##type##Layer() {    \
        return new type##Layer();  \
    }   \
    REGISTER_LAYER_CREATOR(type, Creator_##type##Layer)

}

#endif /* LayerRegistry_hpp */
