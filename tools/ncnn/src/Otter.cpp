//
//  Otter.cpp
//  Otter
//
//  Created by 陳均豪 on 2022/03/10.
//

#include "Otter.hpp"

namespace otter {
namespace core {

std::string parse_arg(std::fstream &f) {
    std::string arg;
    f >> arg;
    if (arg[0] == '"') {
        if (arg[arg.size() - 1] != '"') {
            std::string append;
            std::getline(f, append, '"');
            arg.append(append);
        }
    }
    return arg;
}

Param parse_param(std::fstream &f) {
    size_t mark;
    std::string type;
    std::string arg;
    f >> type;
    if (f.eof()) return Param("End", type);
    if ((mark = type.find(':')) != std::string::npos) {
        if (mark == type.size() - 1) {
            type = type.substr(0, mark);
            arg = parse_arg(f);
        } else {
            arg = type.substr(mark + 1);
            type = type.substr(0, mark);
        }
    } else if ((mark = type.find('{')) != std::string::npos) {
        type = type.substr(0, mark);
        EARSE_SPACE(type);
        return Param("Partner", type);
    } else if ((mark = type.find('}')) != std::string::npos) {
        return Param("End", type);
    } else if (type[0] == '#') {
        getline(f, type, '\n');
        return Param("Comment", type);
    } else if (type[0] == '$') {
        return Param("End", "End of otter syntax");
    } else {
        std::string find_colon;
        f >> find_colon;
        if ((mark = find_colon.find('{')) != std::string::npos) {
            EARSE_SPACE(type);
            return Param("Partner", type);
        } else if ((mark = find_colon.find(':')) == std::string::npos) {
            fprintf(stderr, "[Param] Syntax error!\n");
        } else {
            if (mark == find_colon.size() - 1) {
                arg = parse_arg(f);
            } else {
                arg = find_colon.substr(mark + 1);
            }
        }
    }
    
    EARSE_SPACE(type);
    EARSE_CHARACTER(arg, '"');
    
    return Param(type, arg);
}

Otter::Otter(std::string name) : name_(name) {
}

void Otter::setName(std::string name) {
    name_ = name;
}

std::string Otter::getName() const {
    return name_;
}

std::vector<Param> Otter::getParams() const {
    return params;
}

std::vector<Otter> Otter::getPartners() const {
    return partners;
}

Otter Otter::getPartner(int index) const {
    return partners[index];
}

bool Otter::idle() const {
    return params.empty();
}

void Otter::addPartner(Otter partner) {
    partners.push_back(partner);
}

void Otter::addParam(Param param) {
    params.push_back(param);
}

bool Otter::parseBlueprint(std::fstream &blueprint) {
    Param element = parse_param(blueprint);
    while (element.type != "End") {
        if (element.type == "Partner") {
            Otter team_leader(element.info);
            team_leader.parseBlueprint(blueprint);
            partners.push_back(team_leader);
        } else if (element.type == "Comment") {
            // skip
        } else {
            params.push_back(Param(element.type, element.info));
        }
        element = parse_param(blueprint);
    }
    return true;
}

void Otter::saveBlueprint(FILE *project, int format) {
    WRITE_SPACE(project, format);
    fprintf(project, "%s {\n", name_.c_str());
    
    for (int i = 0; i < params.size(); ++i) {
        WRITE_SPACE(project, (format + 4));
        if (params[i].info.find(' ') != std::string::npos)
            fprintf(project, "%s: \"%s\"\n", params[i].type.c_str(), params[i].info.c_str());
        else
            fprintf(project, "%s: %s\n", params[i].type.c_str(), params[i].info.c_str());
    }
    
    for (int i = 0; i < partners.size(); ++i) {
        partners[i].saveBlueprint(project, format + 4);
    }
    
    WRITE_SPACE(project, format);
    fprintf(project, "}\n");
}

OtterLeader::OtterLeader(std::string project_name_) : project_name(project_name_) {}

void OtterLeader::setProjectName(std::string project_name_) {
    project_name = project_name_;
}

std::string OtterLeader::getProjectName() const {
    return project_name;
}

void OtterLeader::addTeam(Otter team) {
    teams.push_back(team);
}

void OtterLeader::addParam(Param param) {
    params.push_back(param);
}

std::vector<Otter> OtterLeader::getTemas() const {
    return teams;
}

Otter OtterLeader::getTeam(int index) const {
    return teams[index];
}

std::string OtterLeader::getTeamName(int index) const {
    return teams[index].getName();
}

std::vector<Param> OtterLeader::getParams() const {
    return params;
}

Param OtterLeader::getParam(int index) const {
    return params[index];
}

size_t OtterLeader::teams_size() const {
    return teams.size();
}

size_t OtterLeader::params_size() const {
    return params.size();
}

bool OtterLeader::readProject(const char *project_path) {
    std::fstream project;
    project.open(project_path);
    if (!project.is_open()) {
        fprintf(stderr, "[OtterLeader] Open file fail!\n");
        return false;
    }
    
    Param project_title = parse_param(project);
    if (project_title.type == "name")
        project_name = project_title.info;
    else
        fprintf(stderr, "[OtterLeader] Syntax error!\n");
    
    Param segment = parse_param(project);
    while (segment.type != "End") {
        if (segment.type == "Partner") {
            Otter team_leader(segment.info);
            team_leader.parseBlueprint(project);
            teams.push_back(team_leader);
        } else if (segment.type == "Comment") {
            // skip
        } else {
            params.push_back(segment);
        }
        segment = parse_param(project);
    }
    
    project.close();
    return true;
}

bool OtterLeader::saveProject(const char *project_path) {
    FILE *project = fopen(project_path, "w");
    if (!project) return false;
    
    fprintf(project, "name: \"%s\"\n", project_name.c_str());
    for (int i = 0; i < params.size(); ++i) {
        if (params[i].info.find(' ') == std::string::npos)
            fprintf(project, "%s: %s\n", params[i].type.c_str(), params[i].info.c_str());
        else
            fprintf(project, "%s: \"%s\"\n", params[i].type.c_str(), params[i].info.c_str());
    }
    
    for (int i = 0; i < teams.size(); ++i) {
        teams[i].saveBlueprint(project);
    }
    
    fclose(project);
    return true;
}


}   // end namespace core
}   // end namespace otter
