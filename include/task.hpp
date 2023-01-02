#ifndef TASK_H_
#define TASK_H_

#include "utils.hpp"

class task {
private:
    std::string template_name;
    size_t layer_num;
    std::vector <std::pair<std::string, std::string>> layer_meta_params_string;
    std::vector <std::vector<std::string>> layer_profiling_result;

public:

    // Future Custom Task
    explicit task();

    task(const std::string& __template_name, const char* __config, const char* __profiling) {
        layer_num = 0;
        template_name = __template_name;
        if (task_mapping.find(template_name) == task_mapping.end()) {
            task_mapping[template_name] = 0;
        }
        task_mapping[template_name]++;
        layer_meta_params_string.clear();
        std::ifstream fin(__config, std::ios::in);
        char buffer[LAYER_PARAM_LEN_MAX];
        std::regex filter(",");
        while (!fin.eof()) {
            fin.getline(buffer, LAYER_PARAM_LEN_MAX);
            std::string buffer_str = (std::string) buffer;
            layer_num++;
            std::vector <std::string> _layer_info(std::sregex_token_iterator(buffer_str.begin(), buffer_str.end(), filter, -1), std::sregex_token_iterator());
            layer_meta_params_string.push_back(std::pair<std::string, std::string>(_layer_info[0], _layer_info[1]));
        }
        fin.close();
        std::ifstream fin_prof(__profiling, std::ios::in);
        while (!fin_prof.eof()) {
            fin_prof.getline(buffer, LAYER_PARAM_LEN_MAX);
            std::string buffer_str = (std::string) buffer;
            std::vector <std::string> _layer_core_perf(std::sregex_token_iterator(buffer_str.begin(), buffer_str.end(), filter, -1), std::sregex_token_iterator());
            layer_profiling_result.push_back(_layer_core_perf);
            
        }
        fin_prof.close();
    }

    friend class request;
};

#endif