#pragma once

#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <boost/functional/hash.hpp>
#include <pybind11/pybind11.h>

namespace bytebpe {

namespace py = pybind11;

enum class SymbolType {
  INTERNAL = 0,
  FINAL = 1,
  BYTE_PAIR = 2
};

using std::pair;
using std::vector;
using std::unordered_map;
using std::set;
using std::unordered_set;
using std::make_pair;

using BytePair = pair<int, int>;
using SymbolValue = std::variant<BytePair, int>;
template<typename V>
using BytePairMap = unordered_map<BytePair, V, boost::hash<BytePair>>;

class ByteBPE {
private:


  static int count_line(const std::string& filename);
  template<typename Key, typename Hash>
  static void increment_counter(unordered_map<Key, int, Hash>& counter, Key key, int value);
  void create_reverse_symbol_mapping();
  void generate_symbol_strings();
  static vector<int> substitute_byte_pair(const vector<int>& token_vector, BytePair bp, int symbol);
  void clear();

public:
  vector<pair<SymbolValue, SymbolType>> symbol_mapping{};
  unordered_map<pair<SymbolValue, SymbolType>, int, boost::hash<pair<SymbolValue, SymbolType>>> bp_to_symbol;
  vector<std::string> symbol_to_string{};

  ByteBPE() = default;
  ByteBPE(const ByteBPE&) = delete;
  ByteBPE& operator=(const ByteBPE&) = delete;

  void learn(const std::string& filename, const int& vocab_size);
  void save_to_file(const std::string& filename);
  void load_from_file(const std::string& filename, bool overwrite = false);
  vector<int> encode_line(const std::string& line);
  vector<int> encode_token(const std::string& token);
  std::string decode(const vector<int>& token_ids);
  py::bytes py_decode(const vector<int>& token_ids) { return py::bytes(decode(token_ids)); };

};

} // namespace bytebpe
