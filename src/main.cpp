//
// Created by Leo on 2019/12/29.
//

#include <iostream>
#include <string>

#include <boost/program_options.hpp>

#include "byte_bpe.h"

namespace po = boost::program_options;

int main(int argc, char *argv[]) {

  std::string filename{};
  std::string save_to{};
  std::string load_from{};
  int vocab_size{ 320 };

  po::options_description desc("Allowed options");
  desc.add_options()
      ("help", "produce help message")
      ("learn", po::value<std::string>(&filename), "file name")
      ("vocab", po::value<int>(&vocab_size)->default_value(320), "vocab size")
      ("save", po::value<std::string>(&save_to), "file to save to")
      ("load", po::value<std::string>(&load_from), "file to load from");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);

  if (vm.count("help")) {
    std::cout << desc << "\n";
    return 1;
  }

  po::notify(vm);

  bytebpe::ByteBPE bpe{};
  if (vm.count("load"))
    bpe.load_from_file(load_from);
  if (vm.count("learn"))
    bpe.learn(filename, vocab_size);
  if (vm.count("save"))
    bpe.save_to_file(save_to);

  return 0;
}