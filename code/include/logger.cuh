#ifndef __LOGGER__
#define __LOGGER__
#include <string>

#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"

class Logger {
 public:
  Logger(std::string log_file_path = "Log.txt") {
    const plog::util::nchar *path = log_file_path.c_str();
    plog::init(plog::debug, path);
  }
};

#endif