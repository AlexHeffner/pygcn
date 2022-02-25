#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <vector>

std::vector<std::vector<double>> matrix_multi(std::vector<std::vector<double>> a, std::vector<std::vector<double>> b, std::vector<std::vector<double>> c);

namespace py = pybind11;

PYBIND11_MODULE(pybind_11_example, mod) {
    mod.def("matrix_multi_cpp", &matrix_multi, "multily matrixes.");
}