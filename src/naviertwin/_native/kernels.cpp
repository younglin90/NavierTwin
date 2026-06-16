#include <cmath>
#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

using Array2D = py::array_t<double, py::array::c_style | py::array::forcecast>;

using ArrayD = py::array_t<double, py::array::c_style | py::array::forcecast>;

static void check_same_2d(const Array2D& u, const Array2D& v) {
    if (u.ndim() != 2 || v.ndim() != 2) {
        throw std::invalid_argument("2D arrays expected");
    }
    if (u.shape(0) != v.shape(0) || u.shape(1) != v.shape(1)) {
        throw std::invalid_argument("u and v must have the same shape");
    }
    if (u.shape(0) < 2 || u.shape(1) < 2) {
        throw std::invalid_argument("each array axis must have at least 2 points");
    }
}

static inline double grad_x(const double* a, py::ssize_t ny, py::ssize_t nx, py::ssize_t i, py::ssize_t j, double dx) {
    const py::ssize_t row = i * nx;
    if (j == 0) {
        return (a[row + 1] - a[row]) / dx;
    }
    if (j == nx - 1) {
        return (a[row + j] - a[row + j - 1]) / dx;
    }
    return (a[row + j + 1] - a[row + j - 1]) / (2.0 * dx);
}

static inline double grad_y(const double* a, py::ssize_t ny, py::ssize_t nx, py::ssize_t i, py::ssize_t j, double dy) {
    if (i == 0) {
        return (a[nx + j] - a[j]) / dy;
    }
    if (i == ny - 1) {
        return (a[i * nx + j] - a[(i - 1) * nx + j]) / dy;
    }
    return (a[(i + 1) * nx + j] - a[(i - 1) * nx + j]) / (2.0 * dy);
}

static py::array_t<double> field_j_2d(Array2D u, Array2D v, double dx, double dy) {
    check_same_2d(u, v);
    if (dx == 0.0 || dy == 0.0) {
        throw std::invalid_argument("dx and dy must be non-zero");
    }

    const auto ny = u.shape(0);
    const auto nx = u.shape(1);
    auto out = py::array_t<double>({ny, nx, static_cast<py::ssize_t>(2), static_cast<py::ssize_t>(2)});
    const double* up = u.data();
    const double* vp = v.data();
    double* op = out.mutable_data();

    for (py::ssize_t i = 0; i < ny; ++i) {
        for (py::ssize_t j = 0; j < nx; ++j) {
            const auto base = (((i * nx + j) * 2) * 2);
            op[base + 0] = grad_x(up, ny, nx, i, j, dx);
            op[base + 1] = grad_y(up, ny, nx, i, j, dy);
            op[base + 2] = grad_x(vp, ny, nx, i, j, dx);
            op[base + 3] = grad_y(vp, ny, nx, i, j, dy);
        }
    }
    return out;
}

static py::array_t<double> lambda2_2d(Array2D u, Array2D v, double dx, double dy) {
    check_same_2d(u, v);
    if (dx == 0.0 || dy == 0.0) {
        throw std::invalid_argument("dx and dy must be non-zero");
    }

    const auto ny = u.shape(0);
    const auto nx = u.shape(1);
    auto out = py::array_t<double>({ny, nx});
    const double* up = u.data();
    const double* vp = v.data();
    double* op = out.mutable_data();

    for (py::ssize_t i = 0; i < ny; ++i) {
        for (py::ssize_t j = 0; j < nx; ++j) {
            const double du_dx = grad_x(up, ny, nx, i, j, dx);
            const double du_dy = grad_y(up, ny, nx, i, j, dy);
            const double dv_dx = grad_x(vp, ny, nx, i, j, dx);
            const double dv_dy = grad_y(vp, ny, nx, i, j, dy);

            const double s11 = du_dx;
            const double s22 = dv_dy;
            const double s12 = 0.5 * (du_dy + dv_dx);
            const double o12 = 0.5 * (dv_dx - du_dy);

            const double m11 = s11 * s11 + s12 * s12 - o12 * o12;
            const double m22 = s22 * s22 + s12 * s12 - o12 * o12;
            const double m12 = s11 * s12 + s12 * s22;
            const double mid = 0.5 * (m11 + m22);
            const double rad = std::sqrt(0.25 * (m11 - m22) * (m11 - m22) + m12 * m12);
            op[i * nx + j] = mid - rad;
        }
    }
    return out;
}

static inline void load_grad_3x3(const double* gp, py::ssize_t n, py::ssize_t idx, bool flat9, double j[3][3]) {
    const double* row = gp + (flat9 ? idx * 9 : idx * 9);
    (void)n;
    j[0][0] = row[0];
    j[0][1] = row[1];
    j[0][2] = row[2];
    j[1][0] = row[3];
    j[1][1] = row[4];
    j[1][2] = row[5];
    j[2][0] = row[6];
    j[2][1] = row[7];
    j[2][2] = row[8];
}

static py::ssize_t grad_count(const ArrayD& grad) {
    if (grad.ndim() == 2 && grad.shape(1) == 9) {
        return grad.shape(0);
    }
    if (grad.ndim() == 3 && grad.shape(1) == 3 && grad.shape(2) == 3) {
        return grad.shape(0);
    }
    throw std::invalid_argument("gradient must have shape (N, 9) or (N, 3, 3)");
}

static std::array<double, 3> sorted_symmetric_eigenvalues(double a00, double a01, double a02, double a11, double a12, double a22) {
    const double p1 = a01 * a01 + a02 * a02 + a12 * a12;
    if (p1 == 0.0) {
        std::array<double, 3> eig = {a00, a11, a22};
        std::sort(eig.begin(), eig.end());
        return eig;
    }

    const double q = (a00 + a11 + a22) / 3.0;
    const double b00 = a00 - q;
    const double b11 = a11 - q;
    const double b22 = a22 - q;
    const double p2 = b00 * b00 + b11 * b11 + b22 * b22 + 2.0 * p1;
    const double p = std::sqrt(p2 / 6.0);
    if (p == 0.0) {
        return {q, q, q};
    }

    const double c00 = b00 / p;
    const double c01 = a01 / p;
    const double c02 = a02 / p;
    const double c11 = b11 / p;
    const double c12 = a12 / p;
    const double c22 = b22 / p;
    const double det_c =
        c00 * (c11 * c22 - c12 * c12) -
        c01 * (c01 * c22 - c12 * c02) +
        c02 * (c01 * c12 - c11 * c02);
    double r = 0.5 * det_c;
    r = std::max(-1.0, std::min(1.0, r));

    constexpr double pi = 3.141592653589793238462643383279502884;
    const double phi = std::acos(r) / 3.0;
    const double eig1 = q + 2.0 * p * std::cos(phi);
    const double eig3 = q + 2.0 * p * std::cos(phi + (2.0 * pi / 3.0));
    const double eig2 = 3.0 * q - eig1 - eig3;
    std::array<double, 3> eig = {eig1, eig2, eig3};
    std::sort(eig.begin(), eig.end());
    return eig;
}

static py::tuple q_criterion_from_grad_3d(ArrayD grad) {
    const py::ssize_t n = grad_count(grad);
    const bool flat9 = grad.ndim() == 2;
    auto q = py::array_t<double>({n});
    auto vort = py::array_t<double>({n, static_cast<py::ssize_t>(3)});
    const double* gp = grad.data();
    double* qp = q.mutable_data();
    double* vp = vort.mutable_data();

    for (py::ssize_t idx = 0; idx < n; ++idx) {
        double j[3][3];
        load_grad_3x3(gp, n, idx, flat9, j);
        double s_norm = 0.0;
        double o_norm = 0.0;
        double omega[3][3];
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                const double s = 0.5 * (j[r][c] + j[c][r]);
                const double o = 0.5 * (j[r][c] - j[c][r]);
                omega[r][c] = o;
                s_norm += s * s;
                o_norm += o * o;
            }
        }
        qp[idx] = 0.5 * (o_norm - s_norm);
        vp[idx * 3 + 0] = omega[2][1] - omega[1][2];
        vp[idx * 3 + 1] = omega[0][2] - omega[2][0];
        vp[idx * 3 + 2] = omega[1][0] - omega[0][1];
    }
    return py::make_tuple(q, vort);
}

static py::array_t<double> lambda2_from_grad_3d(ArrayD grad) {
    const py::ssize_t n = grad_count(grad);
    const bool flat9 = grad.ndim() == 2;
    auto out = py::array_t<double>({n});
    const double* gp = grad.data();
    double* op = out.mutable_data();

    for (py::ssize_t idx = 0; idx < n; ++idx) {
        double j[3][3];
        double s[3][3];
        double o[3][3];
        load_grad_3x3(gp, n, idx, flat9, j);
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                s[r][c] = 0.5 * (j[r][c] + j[c][r]);
                o[r][c] = 0.5 * (j[r][c] - j[c][r]);
            }
        }

        double m[3][3] = {};
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                for (int k = 0; k < 3; ++k) {
                    m[r][c] += s[r][k] * s[k][c] + o[r][k] * o[k][c];
                }
            }
        }
        const auto eig = sorted_symmetric_eigenvalues(m[0][0], m[0][1], m[0][2], m[1][1], m[1][2], m[2][2]);
        op[idx] = eig[1];
    }
    return out;
}

static py::tuple decompose_j_3x3(py::array_t<double, py::array::c_style | py::array::forcecast> j) {
    if (j.ndim() != 2 || j.shape(0) != 3 || j.shape(1) != 3) {
        throw std::invalid_argument("3x3 expected");
    }
    auto s = py::array_t<double>({static_cast<py::ssize_t>(3), static_cast<py::ssize_t>(3)});
    auto w = py::array_t<double>({static_cast<py::ssize_t>(3), static_cast<py::ssize_t>(3)});
    const double* jp = j.data();
    double* sp = s.mutable_data();
    double* wp = w.mutable_data();
    for (py::ssize_t r = 0; r < 3; ++r) {
        for (py::ssize_t c = 0; c < 3; ++c) {
            const double a = jp[r * 3 + c];
            const double b = jp[c * 3 + r];
            sp[r * 3 + c] = 0.5 * (a + b);
            wp[r * 3 + c] = 0.5 * (a - b);
        }
    }
    return py::make_tuple(s, w);
}

static py::dict invariants_3x3(py::array_t<double, py::array::c_style | py::array::forcecast> j) {
    if (j.ndim() != 2 || j.shape(0) != 3 || j.shape(1) != 3) {
        throw std::invalid_argument("3x3 expected");
    }
    const double* a = j.data();
    const double trace = a[0] + a[4] + a[8];
    const double j2_trace =
        a[0] * a[0] + a[1] * a[3] + a[2] * a[6] +
        a[3] * a[1] + a[4] * a[4] + a[5] * a[7] +
        a[6] * a[2] + a[7] * a[5] + a[8] * a[8];
    const double det =
        a[0] * (a[4] * a[8] - a[5] * a[7]) -
        a[1] * (a[3] * a[8] - a[5] * a[6]) +
        a[2] * (a[3] * a[7] - a[4] * a[6]);
    const double p = -trace;
    const double q = 0.5 * (p * p - j2_trace);
    const double r = -det;
    py::dict out;
    out["P"] = p;
    out["Q"] = q;
    out["R"] = r;
    return out;
}

PYBIND11_MODULE(_kernels, m) {
    m.doc() = "C++ kernels for NavierTwin numeric hot paths";
    m.def("field_j_2d", &field_j_2d, py::arg("u"), py::arg("v"), py::arg("dx") = 1.0, py::arg("dy") = 1.0);
    m.def("lambda2_2d", &lambda2_2d, py::arg("u"), py::arg("v"), py::arg("dx") = 1.0, py::arg("dy") = 1.0);
    m.def("q_criterion_from_grad_3d", &q_criterion_from_grad_3d, py::arg("gradient"));
    m.def("lambda2_from_grad_3d", &lambda2_from_grad_3d, py::arg("gradient"));
    m.def("decompose_j_3x3", &decompose_j_3x3, py::arg("J"));
    m.def("invariants_3x3", &invariants_3x3, py::arg("J"));
}
