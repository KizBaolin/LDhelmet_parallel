// Copyright (C) 2012  Andrew H. Chan, Paul A. Jenkins, Yun S. Song
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// <http://www.gnu.org/licenses/>
//
// email: andrewhc@eecs.berkeley.edu

#ifndef LDHELMET_TABLE_GEN_OUTPUT_WRITER_H_
#define LDHELMET_TABLE_GEN_OUTPUT_WRITER_H_

#include <stddef.h>
#include <stdint.h>

#include <string>
#include <vector>

#include "common/conf.h"
#include "common/vector_definitions.h"

class InputConfBinaryWriter {
 public:
  InputConfBinaryWriter(std::string file_name,
                        std::vector<Conf> const &conf_list,
                        std::vector<size_t> const &degree_seps);

  void Write(uint32_t degree, Vec8 const &table);

  std::vector<Conf> const &conf_list_;
  std::vector<size_t> const &degree_seps_;
  FILE *fp_;
};

#endif  // LDHELMET_TABLE_GEN_OUTPUT_WRITER_H_
