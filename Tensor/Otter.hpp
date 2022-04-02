//
//  Otter.hpp
//  Otter
//
//  Created by 陳均豪 on 2022/03/10.
//

#ifndef Otter_hpp
#define Otter_hpp

#include <stdio.h>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>

namespace otter {
namespace core{

#define EARSE_CHARACTER(str, c) str.erase(std::remove_if(str.begin(), str.end(), [](unsigned char x) { return x == c; }), str.end());

#define EARSE_SPACE(str) EARSE_CHARACTER(str, ' ')

#define WRITE_SPACE(file, n) for (int i = n; i--; ) fprintf(file, " ");

struct Param {
    Param(std::string type_, std::string info_) : type(type_), info(info_) {}
    std::string type;
    std::string info;
};

class Otter {
public:
    Otter(std::string name = "");
    
    void setName(std::string name);
    
    void addParam(Param param);
    
    void addPartner(Otter partner);
    
    bool parseBlueprint(std::fstream &blueprint);
    
    void saveBlueprint(FILE *project, int format = 0);
    
    std::string getName() const;
    
    std::vector<Param> getParams() const;
    
    std::vector<Otter> getPartners() const;
    
    Otter getPartner(int index) const;
    
    bool idle() const;
    
private:
    std::string name_;
    std::vector<Otter> partners;
    std::vector<Param> params;
};

class OtterLeader {
public:
    OtterLeader(std::string project_name = "");
    
    void setProjectName(std::string project_name);
    
    std::string getProjectName() const;
    
    void addTeam(Otter team);
    
    void addParam(Param param);
    
    std::vector<Otter> getTemas() const;
    
    Otter getTeam(int index) const;
    
    std::string getTeamName(int index) const;
    
    std::vector<Param> getParams() const;
    
    std::vector<Param> getTeamParams(int index) const;
    
    Param getParam(int index) const;
    
    size_t params_size() const;
    
    size_t teams_size() const;
    
    bool readProject(const char *project_path);
    
    bool saveProject(const char *project_path);
    
private:
    std::string project_name;
    std::vector<Otter> teams;
    std::vector<Param> params;
};


}   // end namespace core
}   // end namespace otter

#endif /* Otter_hpp */
