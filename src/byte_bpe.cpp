#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <boost/format.hpp>
#include <boost/functional/hash.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/heap/fibonacci_heap.hpp>
#include <boost/timer/progress_display.hpp>

#include "byte_bpe.h"

namespace bytebpe {

void ByteBPE::learn(const std::string& filename, const int& vocab_size) {
  using boost::heap::fibonacci_heap;
  using boost::timer::progress_display;

  clear();

  int n_line = count_line(filename);
  std::cout << "Learning BPE on: " << filename;  // for debug
  std::cout << " (" << n_line << " lines)" << '\n';  // for debug

  std::ifstream file(filename);
  if (!file.good())
      throw std::invalid_argument{ boost::str(boost::format("unable to open file %s") % filename)};

  // (1) build token count
  std::cout << "indexing tokens...";
  unordered_map<std::string, int> token_counter{};
  unordered_set<char> unique_internal_bytes{};
  unordered_set<char> unique_final_bytes{};
  progress_display token_index_progress{ static_cast<unsigned long>(n_line) };
  std::string line{};

  while (std::getline(file, line)) {
    std::string token{};
    std::istringstream token_stream{line};

    while(std::getline(token_stream, token, ' ')) {
      increment_counter(token_counter, token, 1);

      for (size_t i = 0; i + 1 < token.size(); ++i) {
        unique_internal_bytes.insert(token[i]);
      }
      unique_final_bytes.insert(token.back());
    }

    ++token_index_progress;
  }
  file.close();

  // build the base symbol set (all internal and final bytes)
  set<char> ordered_internal_bytes{unique_internal_bytes.begin(), unique_internal_bytes.end()};
  set<char> ordered_final_bytes{unique_final_bytes.begin(), unique_final_bytes.end()};

  unordered_map<pair<int, SymbolType>, int, boost::hash<pair<int, SymbolType>>> byte_to_symbol{};
  // add all internal bytes
  for (auto internal_byte : ordered_internal_bytes) {
    const int symbol = symbol_mapping.size();
    pair<SymbolValue, SymbolType> byte{internal_byte, SymbolType::INTERNAL};
    symbol_mapping.emplace_back(byte);
    byte_to_symbol[{internal_byte, SymbolType::INTERNAL}] = symbol;
  }
  // add all final bytes
  for (auto final_byte: ordered_final_bytes) {
    const int symbol = symbol_mapping.size();
    pair<SymbolValue, SymbolType> byte{final_byte, SymbolType::FINAL};
    symbol_mapping.emplace_back(byte);
    byte_to_symbol[{final_byte, SymbolType::FINAL}] = symbol;
  }


  // (2) build byte pair index:
  //     - counter BP->int
  //     - reverse index counter BP->counter(token_index)
  //     - token index -> token vector
  std::cout << "indexing byte pairs...";
  progress_display bp_index_progress{token_counter.size() };

  BytePairMap<int> bp_counter{};
  BytePairMap<unordered_map<int, int>> bp_to_token_index_counter{};
  vector<pair<vector<int>, int>> token_index_to_token_vec_and_freq{};

  for(const auto& token_cnt_pair : token_counter) {
    const std::string& token = token_cnt_pair.first;
    const auto cnt = token_cnt_pair.second;

    vector<int> token_vec{};
    for (size_t i = 0; i + 1 < token.size(); ++i) {
      token_vec.emplace_back(byte_to_symbol[{token[i], SymbolType::INTERNAL}]);
    }
    token_vec.emplace_back(byte_to_symbol[{token.back(), SymbolType::FINAL}]);

    const int token_index = token_index_to_token_vec_and_freq.size();
    token_index_to_token_vec_and_freq.emplace_back(make_pair(token_vec, cnt));

    for (size_t i = 0; i + 1 < token_vec.size(); ++i) {
      BytePair bp{token_vec[i], token_vec[i + 1]};

      increment_counter(bp_counter, bp, cnt);

      if (bp_to_token_index_counter.find(bp) == bp_to_token_index_counter.end())
        bp_to_token_index_counter[bp] = {{token_index, 1}};
      else
        increment_counter(bp_to_token_index_counter[bp], token_index, 1);
    }
    ++bp_index_progress;
  }

  // (3) building BPE frequency heap
  std::cout << "building heap...";
  progress_display build_heap_progress{bp_counter.size() };

  fibonacci_heap<pair<int, BytePair>> bp_heap{};
  using handle_t = fibonacci_heap<pair<int, BytePair>>::handle_type;
  BytePairMap<handle_t> bp_heap_handles{};

  for (const auto& [byte_pair, freq]: bp_counter) {
    const handle_t handle = bp_heap.push(make_pair(freq, byte_pair));
    bp_heap_handles[byte_pair] = handle;
    ++build_heap_progress;
  }

  // (4) creating new symbols
  std::cout << "creating new symbols...";
  const int n_base_vocab = symbol_mapping.size();
  progress_display create_symbol_progress{ static_cast<unsigned long>(vocab_size - n_base_vocab) };

  for (int new_symbol = n_base_vocab; new_symbol < vocab_size; ++new_symbol) {
    const auto [top_bp_freq, top_bp] = bp_heap.top();
    bp_heap.pop();
    bp_heap_handles.erase(top_bp);
    symbol_mapping.emplace_back(make_pair(top_bp, SymbolType::BYTE_PAIR));

    // update affected token vectors and record by how much the byte pairs' freq changed (to update the heap altogether later)
    BytePairMap<int> bp_freq_delta{};
    const auto token_index_counter = bp_to_token_index_counter[top_bp];
    for (const auto token_index_count : token_index_counter) {
      const int token_index = token_index_count.first;
      const auto [token_vector, token_freq] = token_index_to_token_vec_and_freq[token_index];

      vector<int> new_token_vector = substitute_byte_pair(token_vector, top_bp, new_symbol);

      // increase new byte pair freqs
      for (size_t i = 0; i + 1 < new_token_vector.size(); ++i) {
        BytePair bp{new_token_vector[i], new_token_vector[i+1]};
        increment_counter(bp_freq_delta, bp, token_freq);

        if (bp_to_token_index_counter.find(bp) == bp_to_token_index_counter.end())
          bp_to_token_index_counter[bp] = {{token_index, 1}};
        else
          increment_counter(bp_to_token_index_counter[bp], token_index, 1);
      }

      // decrease old byte pair freqs
      for (size_t i = 0; i + 1 < token_vector.size(); ++i) {
        BytePair bp{token_vector[i], token_vector[i+1]};
        increment_counter(bp_freq_delta, bp, -token_freq);

        assert(bp_to_token_index_counter.find(bp) != bp_to_token_index_counter.end()); // old byte pair should be maintained
        increment_counter(bp_to_token_index_counter[bp], token_index, -1);
      }

      token_index_to_token_vec_and_freq[token_index] = {new_token_vector, token_freq};
    }

    assert(bp_freq_delta[top_bp] + bp_counter[top_bp] == 0);
    bp_counter.erase(top_bp);
    bp_freq_delta.erase(top_bp);

    // execute the update in frequencies in both heap and map (counter)
    for (const auto [bp, freq_delta] : bp_freq_delta)
      if (freq_delta != 0) {
        if (bp_counter.find(bp) == bp_counter.end()) {
          assert(freq_delta > 0);
          bp_counter[bp] = freq_delta;
          handle_t handle = bp_heap.push({freq_delta, bp});
          bp_heap_handles[bp] = handle;
        } else {
          bp_counter[bp] += freq_delta;
          bp_heap.update(bp_heap_handles[bp], {bp_counter[bp], bp});
        }
      }

    ++create_symbol_progress;
  }

  create_reverse_symbol_mapping();
  generate_symbol_strings();
}

int ByteBPE::count_line(const std::string &filename) {
  std::ifstream file(filename);
  if (!file.good())
    throw std::invalid_argument{ boost::str(boost::format("unable to open file %s") % filename)};

  std::string line{};
  int n_line{0};

  while (std::getline(file, line)) {
    ++n_line;
  }

  file.close();
  return n_line;
}

template<typename Key, typename Hash>
void ByteBPE::increment_counter(unordered_map<Key, int, Hash>& counter, Key key, int value) {
  if (value == 0)
    throw std::invalid_argument{ "Cannot increment counter by 0." };

  if (counter.find(key) == counter.end())
    counter[key] = value;
  else {
    counter[key] += value;
    if (value < 0 && counter[key] == 0)
      counter.erase(key);
  }
}

void ByteBPE::save_to_file(const std::string &filename) {
  std::ofstream file_out(filename);
  if (!file_out.good())
    throw std::invalid_argument{ boost::str(boost::format("unable to open file %s") % filename) };

  for (const auto [symbol_value, symbol_type] : symbol_mapping) {
    switch (symbol_type) {
      case SymbolType::INTERNAL:
      case SymbolType::FINAL: {
        file_out << std::get<int>(symbol_value) << " " << static_cast<int>(symbol_type) << '\n';
        break;
      }
      case SymbolType::BYTE_PAIR: {
        auto [symbol1, symbol2] = std::get<BytePair>(symbol_value);
        file_out << symbol1 << " " << symbol2 << " " << static_cast<int>(symbol_type) << '\n';
        break;
      }
    }
  }

  file_out.close();
}

void ByteBPE::load_from_file(const std::string& filename, const bool overwrite) {
  std::ifstream file_in(filename);
  if (!file_in.good())
    throw std::invalid_argument{ boost::str(boost::format("unable to open file %s") % filename)};

  if (!symbol_mapping.empty() && !overwrite)
    throw std::invalid_argument{
      "Trying to load into a learned/loaded BPE object without specifying overwrite = true; "
      "Recommend to save learned symbols with ``save_to_file`` before overwriting." };

  clear();

  std::string line{};
  int symbol_id{0};
  while (std::getline(file_in, line)) {
    std::istringstream token_stream{line};
    vector<int> tokens{};
    int token;
    while (token_stream >> token)
      tokens.emplace_back(token);

    SymbolType symbol_type;
    SymbolValue symbol_value;
    if (tokens.size() == 2) {
      symbol_type = static_cast<SymbolType>(tokens[1]);
      if (symbol_type != SymbolType::INTERNAL && symbol_type != SymbolType::FINAL)
        throw std::invalid_argument{ "Malformatted bpe file" };
      symbol_value = tokens[0];
    } else if (tokens.size() == 3) {
      symbol_type = static_cast<SymbolType>(tokens[2]);
      if (symbol_type != SymbolType::BYTE_PAIR || tokens[0] >= symbol_id || tokens[1] >= symbol_id) // enforce acyclic
        throw std::invalid_argument{ "Malformatted bpe file" };
      symbol_value = BytePair{ tokens[0], tokens[1] };
    } else {
      throw std::invalid_argument{ "Malformatted bpe file" };
    }
    symbol_mapping.emplace_back(pair<SymbolValue, SymbolType>{symbol_value, symbol_type});
    ++symbol_id;
  }

  create_reverse_symbol_mapping();
  generate_symbol_strings();
}

vector<int> ByteBPE::encode_line(const std::string &line) {
  std::istringstream token_stream{line};
  std::string token;
  vector<int> encoded_line{};

  while (getline(token_stream, token, ' ')) {
    auto encoded_token = encode_token(token);
    encoded_line.insert(encoded_line.end(), encoded_token.begin(), encoded_token.end());
  }

  return encoded_line;
}

vector<int> ByteBPE::encode_token(const std::string &token) {
  vector<int> token_symbols{};

  for (size_t i = 0; i + 1 < token.size(); ++i)
    token_symbols.emplace_back(bp_to_symbol.at(make_pair(token[i], SymbolType::INTERNAL)));
  token_symbols.emplace_back(bp_to_symbol.at(make_pair(token.back(), SymbolType::FINAL)));

  while (token_symbols.size() > 1) {
    BytePair min_bp;
    int min_bp_symbol = std::numeric_limits<int>::max();

    for (size_t i = 0; i + 1 < token_symbols.size(); ++i) {
      auto bp = make_pair(make_pair(token_symbols[i], token_symbols[i+1]), SymbolType::BYTE_PAIR);
      if (bp_to_symbol.find(bp) != bp_to_symbol.end() && bp_to_symbol[bp] < min_bp_symbol) {
        min_bp_symbol = bp_to_symbol[bp];
        min_bp = bp.first;
      }
    }
    if (min_bp_symbol == std::numeric_limits<int>::max())
      break;

    token_symbols = substitute_byte_pair(token_symbols, min_bp, min_bp_symbol);
  }

  return token_symbols;
}

std::string ByteBPE::decode(const vector<int>& token_ids) {
  std::string decoded{};
  for (const int symbol : token_ids) {
    decoded += symbol_to_string[symbol];
  }

  return decoded;
}

void ByteBPE::create_reverse_symbol_mapping() {
  int symbol_id = 0;
  for (const auto& [symbol_value, symbol_type] : symbol_mapping) {
    bp_to_symbol[make_pair(symbol_value, symbol_type)] = symbol_id;
    ++symbol_id;
  }
}

vector<int> ByteBPE::substitute_byte_pair(const vector<int>& token_vector, BytePair bp, int symbol) {
  vector<int> new_token_vector{};
  for (size_t i = 0; i < token_vector.size(); ++i) {
    if (i + 1 < token_vector.size() && BytePair{token_vector[i], token_vector[i+1]} == bp) {
      new_token_vector.emplace_back(symbol);
      ++i;
    } else
      new_token_vector.emplace_back(token_vector[i]);
  }

  return new_token_vector;
}

void ByteBPE::clear() {
  symbol_mapping.clear();
  bp_to_symbol.clear();
  symbol_to_string.clear();
}

void ByteBPE::generate_symbol_strings() {
  assert(symbol_to_string.empty());
  using boost::numeric_cast;

  for (const auto [value, type] : symbol_mapping) {
    switch (type) {
      case SymbolType::INTERNAL: {
        symbol_to_string.emplace_back(std::string{numeric_cast<char>(std::get<int>(value))});
        break;
      }
      case SymbolType::FINAL: {
        symbol_to_string.emplace_back(std::string{numeric_cast<char>(std::get<int>(value))} + ' ');
        break;
      }
      case SymbolType::BYTE_PAIR: {
        const auto [s1, s2] = std::get<BytePair>(value);
        symbol_to_string.emplace_back(symbol_to_string[s1] + symbol_to_string[s2]);
        break;
      }
    }
  }
}

} // namespace bytebpe
