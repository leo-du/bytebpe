//
// Created by Leo on 2019/12/29.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "byte_bpe.h"

namespace py = pybind11;

PYBIND11_MODULE(bytebpe, m) {
  py::class_<bytebpe::ByteBPE>(m, "ByteBPE")
    .def(py::init<>())
    .def("learn", &bytebpe::ByteBPE::learn,
        "learn BPE from the given filename with the vocab_size",
        py::arg("filename"), py::arg("vocab_size"))
    .def("save_to_file", &bytebpe::ByteBPE::save_to_file,
        "save learned BPE symbol mappings to file",
        py::arg("filename"))
    .def("load_from_file", &bytebpe::ByteBPE::load_from_file,
        "load learned BPE symbol mappings from file",
        py::arg("filename"), py::arg("overwrite") = false)
    .def("encode_line", &bytebpe::ByteBPE::encode_line,
        "encode line with the learned BPE",
        py::arg("line"))
    .def("encode_token", &bytebpe::ByteBPE::encode_token,
        "encode token with the learned BPE",
        py::arg("token"))
    .def("decode", &bytebpe::ByteBPE::py_decode,
        "decode token ids into a python bytes object",
        py::arg("token_ids"));
}
