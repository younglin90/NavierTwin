#include <cmath>
#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

using Array2D = py::array_t<double, py::array::c_style | py::array::forcecast>;

using ArrayD = py::array_t<double, py::array::c_style | py::array::forcecast>;

using ArrayI = py::array_t<long long, py::array::c_style | py::array::forcecast>;

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

static py::array_t<double> vorticity_2d_native(Array2D u, Array2D v, double dx, double dy) {
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
            op[i * nx + j] = grad_x(vp, ny, nx, i, j, dx) - grad_y(up, ny, nx, i, j, dy);
        }
    }
    return out;
}

static py::array_t<double> q_criterion_2d_native(Array2D u, Array2D v, double dx, double dy) {
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
            const double s12 = 0.5 * (du_dy + dv_dx);
            const double o12 = 0.5 * (dv_dx - du_dy);
            const double s2 = du_dx * du_dx + dv_dy * dv_dy + 2.0 * s12 * s12;
            const double o2 = 2.0 * o12 * o12;
            op[i * nx + j] = 0.5 * (o2 - s2);
        }
    }
    return out;
}

static py::array_t<double> production_rate_2d(Array2D u, Array2D v, double dx, double dy, Array2D nu_t) {
    check_same_2d(u, v);
    if (nu_t.ndim() != 2 || nu_t.shape(0) != u.shape(0) || nu_t.shape(1) != u.shape(1)) {
        throw std::invalid_argument("nu_t must have the same shape as u and v");
    }
    if (dx == 0.0 || dy == 0.0) {
        throw std::invalid_argument("dx and dy must be non-zero");
    }

    const auto ny = u.shape(0);
    const auto nx = u.shape(1);
    auto out = py::array_t<double>({ny, nx});
    const double* up = u.data();
    const double* vp = v.data();
    const double* ntp = nu_t.data();
    double* op = out.mutable_data();

    for (py::ssize_t i = 0; i < ny; ++i) {
        for (py::ssize_t j = 0; j < nx; ++j) {
            const double dudx = grad_x(up, ny, nx, i, j, dx);
            const double dudy = grad_y(up, ny, nx, i, j, dy);
            const double dvdx = grad_x(vp, ny, nx, i, j, dx);
            const double dvdy = grad_y(vp, ny, nx, i, j, dy);
            const double e12 = 0.5 * (dudy + dvdx);
            op[i * nx + j] = 2.0 * ntp[i * nx + j] * (dudx * dudx + dvdy * dvdy + 2.0 * e12 * e12);
        }
    }
    return out;
}

static py::array_t<double> entropy_generation_2d_native(
    Array2D u, Array2D v, Array2D t, double dx, double dy, double mu, double k
) {
    check_same_2d(u, v);
    if (t.ndim() != 2 || t.shape(0) != u.shape(0) || t.shape(1) != u.shape(1)) {
        throw std::invalid_argument("u, v, and T must have the same 2D shape");
    }
    if (dx == 0.0 || dy == 0.0) {
        throw std::invalid_argument("dx and dy must be non-zero");
    }

    const auto ny = u.shape(0);
    const auto nx = u.shape(1);
    auto out = py::array_t<double>({ny, nx});
    const double* up = u.data();
    const double* vp = v.data();
    const double* tp = t.data();
    double* op = out.mutable_data();

    for (py::ssize_t i = 0; i < ny; ++i) {
        for (py::ssize_t j = 0; j < nx; ++j) {
            const py::ssize_t index = i * nx + j;
            const double temp = tp[index];
            if (temp <= 0.0) {
                throw std::invalid_argument("T must be positive");
            }
            const double dtdx = grad_x(tp, ny, nx, i, j, dx);
            const double dtdy = grad_y(tp, ny, nx, i, j, dy);
            const double dudx = grad_x(up, ny, nx, i, j, dx);
            const double dudy = grad_y(up, ny, nx, i, j, dy);
            const double dvdx = grad_x(vp, ny, nx, i, j, dx);
            const double dvdy = grad_y(vp, ny, nx, i, j, dy);
            const double e12 = 0.5 * (dudy + dvdx);
            const double div = dudx + dvdy;
            const double phi =
                2.0 * (dudx * dudx + dvdy * dvdy + 2.0 * e12 * e12) -
                (2.0 / 3.0) * div * div;
            op[index] = (k / (temp * temp)) * (dtdx * dtdx + dtdy * dtdy) + (mu / temp) * phi;
        }
    }
    return out;
}

static void check_same_3d(const ArrayD& u, const ArrayD& v, const ArrayD& w) {
    if (u.ndim() != 3 || v.ndim() != 3 || w.ndim() != 3) {
        throw std::invalid_argument("3D arrays expected");
    }
    if (
        u.shape(0) != v.shape(0) || u.shape(0) != w.shape(0) ||
        u.shape(1) != v.shape(1) || u.shape(1) != w.shape(1) ||
        u.shape(2) != v.shape(2) || u.shape(2) != w.shape(2)
    ) {
        throw std::invalid_argument("u, v, and w must have the same shape");
    }
    if (u.shape(0) < 2 || u.shape(1) < 2 || u.shape(2) < 2) {
        throw std::invalid_argument("each array axis must have at least 2 points");
    }
}

static inline py::ssize_t idx3(py::ssize_t z, py::ssize_t y, py::ssize_t x, py::ssize_t ny, py::ssize_t nx) {
    return (z * ny + y) * nx + x;
}

static inline double grad3_axis0(
    const double* a, py::ssize_t nz, py::ssize_t ny, py::ssize_t nx,
    py::ssize_t z, py::ssize_t y, py::ssize_t x, double dz
) {
    if (z == 0) {
        return (a[idx3(1, y, x, ny, nx)] - a[idx3(0, y, x, ny, nx)]) / dz;
    }
    if (z == nz - 1) {
        return (a[idx3(z, y, x, ny, nx)] - a[idx3(z - 1, y, x, ny, nx)]) / dz;
    }
    return (a[idx3(z + 1, y, x, ny, nx)] - a[idx3(z - 1, y, x, ny, nx)]) / (2.0 * dz);
}

static inline double grad3_axis1(
    const double* a, py::ssize_t nz, py::ssize_t ny, py::ssize_t nx,
    py::ssize_t z, py::ssize_t y, py::ssize_t x, double dy
) {
    (void)nz;
    if (y == 0) {
        return (a[idx3(z, 1, x, ny, nx)] - a[idx3(z, 0, x, ny, nx)]) / dy;
    }
    if (y == ny - 1) {
        return (a[idx3(z, y, x, ny, nx)] - a[idx3(z, y - 1, x, ny, nx)]) / dy;
    }
    return (a[idx3(z, y + 1, x, ny, nx)] - a[idx3(z, y - 1, x, ny, nx)]) / (2.0 * dy);
}

static inline double grad3_axis2(
    const double* a, py::ssize_t nz, py::ssize_t ny, py::ssize_t nx,
    py::ssize_t z, py::ssize_t y, py::ssize_t x, double dx
) {
    (void)nz;
    if (x == 0) {
        return (a[idx3(z, y, 1, ny, nx)] - a[idx3(z, y, 0, ny, nx)]) / dx;
    }
    if (x == nx - 1) {
        return (a[idx3(z, y, x, ny, nx)] - a[idx3(z, y, x - 1, ny, nx)]) / dx;
    }
    return (a[idx3(z, y, x + 1, ny, nx)] - a[idx3(z, y, x - 1, ny, nx)]) / (2.0 * dx);
}

static py::tuple vorticity_3d_native(ArrayD u, ArrayD v, ArrayD w, double dx, double dy, double dz) {
    check_same_3d(u, v, w);
    if (dx == 0.0 || dy == 0.0 || dz == 0.0) {
        throw std::invalid_argument("dx, dy, and dz must be non-zero");
    }

    const auto nz = u.shape(0);
    const auto ny = u.shape(1);
    const auto nx = u.shape(2);
    auto wx = py::array_t<double>({nz, ny, nx});
    auto wy = py::array_t<double>({nz, ny, nx});
    auto wz = py::array_t<double>({nz, ny, nx});
    const double* up = u.data();
    const double* vp = v.data();
    const double* wp = w.data();
    double* wxp = wx.mutable_data();
    double* wyp = wy.mutable_data();
    double* wzp = wz.mutable_data();

    for (py::ssize_t z = 0; z < nz; ++z) {
        for (py::ssize_t y = 0; y < ny; ++y) {
            for (py::ssize_t x = 0; x < nx; ++x) {
                const py::ssize_t index = idx3(z, y, x, ny, nx);
                const double du_dy = grad3_axis1(up, nz, ny, nx, z, y, x, dy);
                const double du_dz = grad3_axis0(up, nz, ny, nx, z, y, x, dz);
                const double dv_dx = grad3_axis2(vp, nz, ny, nx, z, y, x, dx);
                const double dv_dz = grad3_axis0(vp, nz, ny, nx, z, y, x, dz);
                const double dw_dx = grad3_axis2(wp, nz, ny, nx, z, y, x, dx);
                const double dw_dy = grad3_axis1(wp, nz, ny, nx, z, y, x, dy);
                wxp[index] = dw_dy - dv_dz;
                wyp[index] = du_dz - dw_dx;
                wzp[index] = dv_dx - du_dy;
            }
        }
    }
    return py::make_tuple(wx, wy, wz);
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

static py::array_t<double> symmetric_eigenvalues_3x3(py::array_t<double, py::array::c_style | py::array::forcecast> j) {
    if (j.ndim() != 2 || j.shape(0) != 3 || j.shape(1) != 3) {
        throw std::invalid_argument("3x3 expected");
    }
    const double* a = j.data();
    const double a00 = a[0];
    const double a01 = 0.5 * (a[1] + a[3]);
    const double a02 = 0.5 * (a[2] + a[6]);
    const double a11 = a[4];
    const double a12 = 0.5 * (a[5] + a[7]);
    const double a22 = a[8];
    const auto eig = sorted_symmetric_eigenvalues(a00, a01, a02, a11, a12, a22);
    auto out = py::array_t<double>({static_cast<py::ssize_t>(3)});
    double* op = out.mutable_data();
    op[0] = eig[0];
    op[1] = eig[1];
    op[2] = eig[2];
    return out;
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

static void check_square_matrix(const ArrayD& a) {
    if (a.ndim() != 2 || a.shape(0) != a.shape(1)) {
        throw std::invalid_argument("square matrix expected");
    }
}

static std::vector<double> contiguous_vector(const ArrayD& x, py::ssize_t n, const char* name) {
    if (x.ndim() != 1 || x.shape(0) != n) {
        throw std::invalid_argument(std::string(name) + " must have shape (N,)");
    }
    const double* xp = x.data();
    return std::vector<double>(xp, xp + n);
}

static double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double out = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        out += a[i] * b[i];
    }
    return out;
}

static double vec_norm(const std::vector<double>& x) {
    return std::sqrt(dot(x, x));
}

static std::vector<double> matvec(const double* a, py::ssize_t n, const std::vector<double>& x) {
    std::vector<double> out(static_cast<std::size_t>(n), 0.0);
    for (py::ssize_t r = 0; r < n; ++r) {
        double acc = 0.0;
        for (py::ssize_t c = 0; c < n; ++c) {
            acc += a[r * n + c] * x[static_cast<std::size_t>(c)];
        }
        out[static_cast<std::size_t>(r)] = acc;
    }
    return out;
}

static void normalize_in_place(std::vector<double>& x) {
    const double scale = vec_norm(x) + 1e-30;
    for (double& value : x) {
        value /= scale;
    }
}

static py::array_t<double> vector_to_numpy(const std::vector<double>& x) {
    auto out = py::array_t<double>({static_cast<py::ssize_t>(x.size())});
    double* op = out.mutable_data();
    std::copy(x.begin(), x.end(), op);
    return out;
}

static std::vector<double> solve_linear_system(std::vector<double> m, std::vector<double> rhs, py::ssize_t n) {
    for (py::ssize_t k = 0; k < n; ++k) {
        py::ssize_t pivot = k;
        double pivot_abs = std::abs(m[k * n + k]);
        for (py::ssize_t r = k + 1; r < n; ++r) {
            const double candidate = std::abs(m[r * n + k]);
            if (candidate > pivot_abs) {
                pivot_abs = candidate;
                pivot = r;
            }
        }
        if (pivot_abs == 0.0) {
            throw std::invalid_argument("singular matrix in inverse_power");
        }
        if (pivot != k) {
            for (py::ssize_t c = 0; c < n; ++c) {
                std::swap(m[k * n + c], m[pivot * n + c]);
            }
            std::swap(rhs[static_cast<std::size_t>(k)], rhs[static_cast<std::size_t>(pivot)]);
        }
        for (py::ssize_t r = k + 1; r < n; ++r) {
            const double factor = m[r * n + k] / m[k * n + k];
            m[r * n + k] = 0.0;
            for (py::ssize_t c = k + 1; c < n; ++c) {
                m[r * n + c] -= factor * m[k * n + c];
            }
            rhs[static_cast<std::size_t>(r)] -= factor * rhs[static_cast<std::size_t>(k)];
        }
    }

    std::vector<double> x(static_cast<std::size_t>(n), 0.0);
    for (py::ssize_t r = n - 1; r >= 0; --r) {
        double acc = rhs[static_cast<std::size_t>(r)];
        for (py::ssize_t c = r + 1; c < n; ++c) {
            acc -= m[r * n + c] * x[static_cast<std::size_t>(c)];
        }
        x[static_cast<std::size_t>(r)] = acc / m[r * n + r];
        if (r == 0) {
            break;
        }
    }
    return x;
}

static py::array_t<double> solve_square_native(ArrayD a, ArrayD b) {
    check_square_matrix(a);
    const py::ssize_t n = a.shape(0);
    if (b.ndim() != 1 && b.ndim() != 2) {
        throw std::invalid_argument("b must have shape (N,) or (N, K)");
    }
    if (b.shape(0) != n) {
        throw std::invalid_argument("A and b dimensions do not match");
    }

    const double* ap = a.data();
    std::vector<double> matrix(ap, ap + n * n);
    const double* bp = b.data();
    if (b.ndim() == 1) {
        std::vector<double> rhs(bp, bp + n);
        std::vector<double> x = solve_linear_system(matrix, rhs, n);
        return vector_to_numpy(x);
    }

    const py::ssize_t cols = b.shape(1);
    auto out = py::array_t<double>({n, cols});
    double* op = out.mutable_data();
    for (py::ssize_t col = 0; col < cols; ++col) {
        std::vector<double> rhs(static_cast<std::size_t>(n), 0.0);
        for (py::ssize_t row = 0; row < n; ++row) {
            rhs[static_cast<std::size_t>(row)] = bp[row * cols + col];
        }
        std::vector<double> x = solve_linear_system(matrix, rhs, n);
        for (py::ssize_t row = 0; row < n; ++row) {
            op[row * cols + col] = x[static_cast<std::size_t>(row)];
        }
    }
    return out;
}

static double determinant_square(const double* a, py::ssize_t d) {
    if (d == 2) {
        return a[0] * a[3] - a[1] * a[2];
    }
    if (d == 3) {
        return
            a[0] * (a[4] * a[8] - a[5] * a[7]) -
            a[1] * (a[3] * a[8] - a[5] * a[6]) +
            a[2] * (a[3] * a[7] - a[4] * a[6]);
    }

    std::vector<double> m(a, a + d * d);
    double det = 1.0;
    int sign = 1;
    for (py::ssize_t k = 0; k < d; ++k) {
        py::ssize_t pivot = k;
        double pivot_abs = std::abs(m[k * d + k]);
        for (py::ssize_t r = k + 1; r < d; ++r) {
            const double candidate = std::abs(m[r * d + k]);
            if (candidate > pivot_abs) {
                pivot_abs = candidate;
                pivot = r;
            }
        }
        if (pivot_abs == 0.0) {
            return 0.0;
        }
        if (pivot != k) {
            for (py::ssize_t c = 0; c < d; ++c) {
                std::swap(m[k * d + c], m[pivot * d + c]);
            }
            sign = -sign;
        }
        const double pivot_value = m[k * d + k];
        det *= pivot_value;
        for (py::ssize_t r = k + 1; r < d; ++r) {
            const double factor = m[r * d + k] / pivot_value;
            for (py::ssize_t c = k + 1; c < d; ++c) {
                m[r * d + c] -= factor * m[k * d + c];
            }
        }
    }
    return sign * det;
}

static py::array_t<double> determinant_batch(ArrayD a) {
    if (a.ndim() < 2 || a.shape(a.ndim() - 1) != a.shape(a.ndim() - 2)) {
        throw std::invalid_argument("array of square matrices expected");
    }
    const py::ssize_t d = a.shape(a.ndim() - 1);
    std::vector<py::ssize_t> out_shape;
    py::ssize_t count = 1;
    for (py::ssize_t axis = 0; axis < a.ndim() - 2; ++axis) {
        out_shape.push_back(a.shape(axis));
        count *= a.shape(axis);
    }
    if (out_shape.empty()) {
        out_shape.push_back(1);
    }

    auto out = py::array_t<double>(out_shape);
    const double* ap = a.data();
    double* op = out.mutable_data();
    const py::ssize_t stride = d * d;
    for (py::ssize_t i = 0; i < count; ++i) {
        op[i] = determinant_square(ap + i * stride, d);
    }
    return out;
}

static py::tuple power_iteration_native(ArrayD a, int n_iter, ArrayD x0, double tol) {
    check_square_matrix(a);
    const py::ssize_t n = a.shape(0);
    std::vector<double> x = contiguous_vector(x0, n, "x0");
    normalize_in_place(x);
    const double* ap = a.data();
    double lam_prev = 0.0;
    double lam = 0.0;
    for (int iter = 0; iter < n_iter; ++iter) {
        std::vector<double> y = matvec(ap, n, x);
        normalize_in_place(y);
        x = std::move(y);
        lam = dot(x, matvec(ap, n, x));
        if (std::abs(lam - lam_prev) < tol * std::max(1.0, std::abs(lam))) {
            break;
        }
        lam_prev = lam;
    }
    return py::make_tuple(lam, vector_to_numpy(x));
}

static py::tuple inverse_power_native(ArrayD a, double shift, int n_iter, ArrayD x0) {
    check_square_matrix(a);
    const py::ssize_t n = a.shape(0);
    const double* ap = a.data();
    std::vector<double> shifted(static_cast<std::size_t>(n * n), 0.0);
    for (py::ssize_t r = 0; r < n; ++r) {
        for (py::ssize_t c = 0; c < n; ++c) {
            shifted[static_cast<std::size_t>(r * n + c)] = ap[r * n + c] - (r == c ? shift : 0.0);
        }
    }

    std::vector<double> x = contiguous_vector(x0, n, "x0");
    normalize_in_place(x);
    for (int iter = 0; iter < n_iter; ++iter) {
        x = solve_linear_system(shifted, x, n);
        normalize_in_place(x);
    }
    const double lam = dot(x, matvec(ap, n, x));
    return py::make_tuple(lam, vector_to_numpy(x));
}

static py::tuple pcg_jacobi_native(ArrayD a, ArrayD b, ArrayD x0, int max_iter, double tol) {
    check_square_matrix(a);
    const py::ssize_t n = a.shape(0);
    std::vector<double> rhs = contiguous_vector(b, n, "b");
    std::vector<double> x = contiguous_vector(x0, n, "x0");
    const double* ap = a.data();
    std::vector<double> inv_d(static_cast<std::size_t>(n), 0.0);
    for (py::ssize_t i = 0; i < n; ++i) {
        const double d = ap[i * n + i];
        if (d == 0.0) {
            throw std::invalid_argument("zero diagonal");
        }
        inv_d[static_cast<std::size_t>(i)] = 1.0 / d;
    }

    std::vector<double> ax = matvec(ap, n, x);
    std::vector<double> r(static_cast<std::size_t>(n), 0.0);
    std::vector<double> z(static_cast<std::size_t>(n), 0.0);
    for (py::ssize_t i = 0; i < n; ++i) {
        r[static_cast<std::size_t>(i)] = rhs[static_cast<std::size_t>(i)] - ax[static_cast<std::size_t>(i)];
        z[static_cast<std::size_t>(i)] = inv_d[static_cast<std::size_t>(i)] * r[static_cast<std::size_t>(i)];
    }
    std::vector<double> p = z;
    double rz = dot(r, z);
    int iter_count = max_iter;
    bool converged = false;
    double residual = vec_norm(r);

    for (int iter = 0; iter < max_iter; ++iter) {
        std::vector<double> ap_vec = matvec(ap, n, p);
        const double alpha = rz / (dot(p, ap_vec) + 1e-30);
        for (py::ssize_t i = 0; i < n; ++i) {
            x[static_cast<std::size_t>(i)] += alpha * p[static_cast<std::size_t>(i)];
            r[static_cast<std::size_t>(i)] -= alpha * ap_vec[static_cast<std::size_t>(i)];
        }
        residual = vec_norm(r);
        if (residual < tol) {
            iter_count = iter + 1;
            converged = true;
            break;
        }
        for (py::ssize_t i = 0; i < n; ++i) {
            z[static_cast<std::size_t>(i)] = inv_d[static_cast<std::size_t>(i)] * r[static_cast<std::size_t>(i)];
        }
        const double rz_new = dot(r, z);
        const double beta = rz_new / rz;
        for (py::ssize_t i = 0; i < n; ++i) {
            p[static_cast<std::size_t>(i)] = z[static_cast<std::size_t>(i)] + beta * p[static_cast<std::size_t>(i)];
        }
        rz = rz_new;
    }

    py::dict info;
    info["iters"] = iter_count;
    info["residual"] = residual;
    info["converged"] = converged;
    return py::make_tuple(vector_to_numpy(x), info);
}

static py::tuple bicgstab_dense_native(ArrayD a, ArrayD b, ArrayD x0, int max_iter, double tol) {
    check_square_matrix(a);
    const py::ssize_t n = a.shape(0);
    std::vector<double> rhs = contiguous_vector(b, n, "b");
    std::vector<double> x = contiguous_vector(x0, n, "x0");
    const double* ap = a.data();

    std::vector<double> ax = matvec(ap, n, x);
    std::vector<double> r(static_cast<std::size_t>(n), 0.0);
    for (py::ssize_t i = 0; i < n; ++i) {
        r[static_cast<std::size_t>(i)] = rhs[static_cast<std::size_t>(i)] - ax[static_cast<std::size_t>(i)];
    }
    std::vector<double> r_hat = r;
    double rho_prev = 1.0;
    double alpha = 1.0;
    double omega = 1.0;
    std::vector<double> v(static_cast<std::size_t>(n), 0.0);
    std::vector<double> p(static_cast<std::size_t>(n), 0.0);
    int iter_count = max_iter;
    bool converged = false;
    double residual = vec_norm(r);

    for (int iter = 0; iter < max_iter; ++iter) {
        const double rho = dot(r_hat, r);
        if (std::abs(rho) < 1e-30) {
            break;
        }
        const double beta = (rho / rho_prev) * (alpha / (omega + 1e-30));
        for (py::ssize_t i = 0; i < n; ++i) {
            p[static_cast<std::size_t>(i)] =
                r[static_cast<std::size_t>(i)] +
                beta * (p[static_cast<std::size_t>(i)] - omega * v[static_cast<std::size_t>(i)]);
        }
        std::vector<double> y = p;
        v = matvec(ap, n, y);
        alpha = rho / (dot(r_hat, v) + 1e-30);
        std::vector<double> s(static_cast<std::size_t>(n), 0.0);
        for (py::ssize_t i = 0; i < n; ++i) {
            s[static_cast<std::size_t>(i)] = r[static_cast<std::size_t>(i)] - alpha * v[static_cast<std::size_t>(i)];
        }
        if (vec_norm(s) < tol) {
            for (py::ssize_t i = 0; i < n; ++i) {
                x[static_cast<std::size_t>(i)] += alpha * y[static_cast<std::size_t>(i)];
            }
            std::vector<double> new_ax = matvec(ap, n, x);
            for (py::ssize_t i = 0; i < n; ++i) {
                r[static_cast<std::size_t>(i)] = rhs[static_cast<std::size_t>(i)] - new_ax[static_cast<std::size_t>(i)];
            }
            residual = vec_norm(r);
            iter_count = iter + 1;
            converged = true;
            break;
        }
        std::vector<double> z = s;
        std::vector<double> t = matvec(ap, n, z);
        omega = dot(t, s) / (dot(t, t) + 1e-30);
        for (py::ssize_t i = 0; i < n; ++i) {
            x[static_cast<std::size_t>(i)] += alpha * y[static_cast<std::size_t>(i)] + omega * z[static_cast<std::size_t>(i)];
            r[static_cast<std::size_t>(i)] = s[static_cast<std::size_t>(i)] - omega * t[static_cast<std::size_t>(i)];
        }
        residual = vec_norm(r);
        if (residual < tol) {
            iter_count = iter + 1;
            converged = true;
            break;
        }
        rho_prev = rho;
    }

    if (!converged) {
        std::vector<double> new_ax = matvec(ap, n, x);
        for (py::ssize_t i = 0; i < n; ++i) {
            r[static_cast<std::size_t>(i)] = rhs[static_cast<std::size_t>(i)] - new_ax[static_cast<std::size_t>(i)];
        }
        residual = vec_norm(r);
    }

    py::dict info;
    info["iters"] = iter_count;
    info["residual"] = residual;
    info["converged"] = converged;
    return py::make_tuple(vector_to_numpy(x), info);
}

static py::array_t<double> thomas_solve_native(ArrayD a, ArrayD b, ArrayD c, ArrayD d) {
    if (d.ndim() != 1) {
        throw std::invalid_argument("d must have shape (N,)");
    }
    const py::ssize_t n = d.shape(0);
    std::vector<double> aa = contiguous_vector(a, n, "a");
    std::vector<double> bb = contiguous_vector(b, n, "b");
    std::vector<double> cc = contiguous_vector(c, n, "c");
    std::vector<double> dd = contiguous_vector(d, n, "d");
    for (py::ssize_t i = 1; i < n; ++i) {
        const double m = aa[static_cast<std::size_t>(i)] / bb[static_cast<std::size_t>(i - 1)];
        bb[static_cast<std::size_t>(i)] -= m * cc[static_cast<std::size_t>(i - 1)];
        dd[static_cast<std::size_t>(i)] -= m * dd[static_cast<std::size_t>(i - 1)];
    }
    std::vector<double> x(static_cast<std::size_t>(n), 0.0);
    x[static_cast<std::size_t>(n - 1)] = dd[static_cast<std::size_t>(n - 1)] / bb[static_cast<std::size_t>(n - 1)];
    for (py::ssize_t i = n - 2; i >= 0; --i) {
        x[static_cast<std::size_t>(i)] =
            (dd[static_cast<std::size_t>(i)] - cc[static_cast<std::size_t>(i)] * x[static_cast<std::size_t>(i + 1)]) /
            bb[static_cast<std::size_t>(i)];
        if (i == 0) {
            break;
        }
    }
    return vector_to_numpy(x);
}

static double dense_residual_norm(const double* a, py::ssize_t n, const std::vector<double>& x, const std::vector<double>& b) {
    std::vector<double> ax = matvec(a, n, x);
    double total = 0.0;
    for (py::ssize_t i = 0; i < n; ++i) {
        const double ri = ax[static_cast<std::size_t>(i)] - b[static_cast<std::size_t>(i)];
        total += ri * ri;
    }
    return std::sqrt(total);
}

static py::tuple jacobi_dense_native(ArrayD a, ArrayD b, ArrayD x0, int max_iter, double tol) {
    check_square_matrix(a);
    const py::ssize_t n = a.shape(0);
    std::vector<double> rhs = contiguous_vector(b, n, "b");
    std::vector<double> x = contiguous_vector(x0, n, "x0");
    const double* ap = a.data();
    std::vector<double> d(static_cast<std::size_t>(n), 0.0);
    for (py::ssize_t i = 0; i < n; ++i) {
        d[static_cast<std::size_t>(i)] = ap[i * n + i];
        if (d[static_cast<std::size_t>(i)] == 0.0) {
            throw std::invalid_argument("zero diagonal");
        }
    }
    std::vector<double> x_new(static_cast<std::size_t>(n), 0.0);
    bool converged = false;
    int iter_count = max_iter;
    double residual = dense_residual_norm(ap, n, x, rhs);
    for (int iter = 0; iter < max_iter; ++iter) {
        for (py::ssize_t row = 0; row < n; ++row) {
            double offdiag = 0.0;
            for (py::ssize_t col = 0; col < n; ++col) {
                if (col != row) {
                    offdiag += ap[row * n + col] * x[static_cast<std::size_t>(col)];
                }
            }
            x_new[static_cast<std::size_t>(row)] = (rhs[static_cast<std::size_t>(row)] - offdiag) / d[static_cast<std::size_t>(row)];
        }
        x = x_new;
        residual = dense_residual_norm(ap, n, x, rhs);
        if (residual < tol) {
            converged = true;
            iter_count = iter + 1;
            break;
        }
    }
    py::dict info;
    info["iters"] = iter_count;
    info["residual"] = residual;
    info["converged"] = converged;
    return py::make_tuple(vector_to_numpy(x), info);
}

static py::tuple gauss_seidel_dense_native(ArrayD a, ArrayD b, ArrayD x0, int max_iter, double tol) {
    check_square_matrix(a);
    const py::ssize_t n = a.shape(0);
    std::vector<double> rhs = contiguous_vector(b, n, "b");
    std::vector<double> x = contiguous_vector(x0, n, "x0");
    const double* ap = a.data();
    bool converged = false;
    int iter_count = max_iter;
    double residual = dense_residual_norm(ap, n, x, rhs);
    for (int iter = 0; iter < max_iter; ++iter) {
        for (py::ssize_t row = 0; row < n; ++row) {
            double s = rhs[static_cast<std::size_t>(row)];
            for (py::ssize_t col = 0; col < row; ++col) {
                s -= ap[row * n + col] * x[static_cast<std::size_t>(col)];
            }
            for (py::ssize_t col = row + 1; col < n; ++col) {
                s -= ap[row * n + col] * x[static_cast<std::size_t>(col)];
            }
            x[static_cast<std::size_t>(row)] = s / ap[row * n + row];
        }
        residual = dense_residual_norm(ap, n, x, rhs);
        if (residual < tol) {
            converged = true;
            iter_count = iter + 1;
            break;
        }
    }
    py::dict info;
    info["iters"] = iter_count;
    info["residual"] = residual;
    info["converged"] = converged;
    return py::make_tuple(vector_to_numpy(x), info);
}

static py::tuple conjugate_gradient_dense_native(ArrayD a, ArrayD b, ArrayD x0, int max_iter, double tol) {
    check_square_matrix(a);
    const py::ssize_t n = a.shape(0);
    std::vector<double> rhs = contiguous_vector(b, n, "b");
    std::vector<double> x = contiguous_vector(x0, n, "x0");
    const double* ap = a.data();
    std::vector<double> ax = matvec(ap, n, x);
    std::vector<double> r(static_cast<std::size_t>(n), 0.0);
    for (py::ssize_t i = 0; i < n; ++i) {
        r[static_cast<std::size_t>(i)] = rhs[static_cast<std::size_t>(i)] - ax[static_cast<std::size_t>(i)];
    }
    std::vector<double> p = r;
    double rs_old = dot(r, r);
    double residual = std::sqrt(rs_old);
    bool converged = false;
    int iter_count = max_iter;
    for (int iter = 0; iter < max_iter; ++iter) {
        std::vector<double> ap_vec = matvec(ap, n, p);
        const double alpha = rs_old / (dot(p, ap_vec) + 1e-30);
        for (py::ssize_t i = 0; i < n; ++i) {
            x[static_cast<std::size_t>(i)] += alpha * p[static_cast<std::size_t>(i)];
            r[static_cast<std::size_t>(i)] -= alpha * ap_vec[static_cast<std::size_t>(i)];
        }
        const double rs_new = dot(r, r);
        residual = std::sqrt(rs_new);
        if (residual < tol) {
            converged = true;
            iter_count = iter + 1;
            break;
        }
        const double beta = rs_new / rs_old;
        for (py::ssize_t i = 0; i < n; ++i) {
            p[static_cast<std::size_t>(i)] = r[static_cast<std::size_t>(i)] + beta * p[static_cast<std::size_t>(i)];
        }
        rs_old = rs_new;
    }
    py::dict info;
    info["iters"] = iter_count;
    info["residual"] = residual;
    info["converged"] = converged;
    return py::make_tuple(vector_to_numpy(x), info);
}

static py::tuple arnoldi_native(ArrayD a, ArrayD b, int k_in) {
    check_square_matrix(a);
    const py::ssize_t n = a.shape(0);
    const int k = std::min(k_in, static_cast<int>(n));
    std::vector<double> q0 = contiguous_vector(b, n, "b");
    normalize_in_place(q0);
    auto q_arr = py::array_t<double>({n, static_cast<py::ssize_t>(k + 1)});
    auto h_arr = py::array_t<double>({static_cast<py::ssize_t>(k + 1), static_cast<py::ssize_t>(k)});
    double* qp = q_arr.mutable_data();
    double* hp = h_arr.mutable_data();
    std::fill(qp, qp + n * (k + 1), 0.0);
    std::fill(hp, hp + (k + 1) * k, 0.0);
    for (py::ssize_t row = 0; row < n; ++row) {
        qp[row * (k + 1)] = q0[static_cast<std::size_t>(row)];
    }

    const double* ap = a.data();
    int cols = k + 1;
    int h_cols = k;
    for (int j = 0; j < k; ++j) {
        std::vector<double> qj(static_cast<std::size_t>(n), 0.0);
        for (py::ssize_t row = 0; row < n; ++row) {
            qj[static_cast<std::size_t>(row)] = qp[row * cols + j];
        }
        std::vector<double> v = matvec(ap, n, qj);
        for (int i = 0; i <= j; ++i) {
            std::vector<double> qi(static_cast<std::size_t>(n), 0.0);
            for (py::ssize_t row = 0; row < n; ++row) {
                qi[static_cast<std::size_t>(row)] = qp[row * cols + i];
            }
            const double hij = dot(qi, v);
            hp[i * h_cols + j] = hij;
            for (py::ssize_t row = 0; row < n; ++row) {
                v[static_cast<std::size_t>(row)] -= hij * qi[static_cast<std::size_t>(row)];
            }
        }
        const double h_next = vec_norm(v);
        hp[(j + 1) * h_cols + j] = h_next;
        if (h_next < 1e-14) {
            const py::ssize_t q_cols = j + 1;
            auto q_small = py::array_t<double>({n, q_cols});
            auto h_small = py::array_t<double>({q_cols, q_cols});
            double* qsp = q_small.mutable_data();
            double* hsp = h_small.mutable_data();
            for (py::ssize_t row = 0; row < n; ++row) {
                for (py::ssize_t col = 0; col < q_cols; ++col) {
                    qsp[row * q_cols + col] = qp[row * cols + col];
                }
            }
            for (py::ssize_t row = 0; row < q_cols; ++row) {
                for (py::ssize_t col = 0; col < q_cols; ++col) {
                    hsp[row * q_cols + col] = hp[row * h_cols + col];
                }
            }
            return py::make_tuple(q_small, h_small);
        }
        for (py::ssize_t row = 0; row < n; ++row) {
            qp[row * cols + j + 1] = v[static_cast<std::size_t>(row)] / h_next;
        }
    }
    return py::make_tuple(q_arr, h_arr);
}

static py::array_t<double> dtw_matrix_native(ArrayD a, ArrayD b) {
    if (a.ndim() != 1 || b.ndim() != 1) {
        throw std::invalid_argument("1D arrays expected");
    }
    const py::ssize_t n = a.shape(0);
    const py::ssize_t m = b.shape(0);
    auto out = py::array_t<double>({n + 1, m + 1});
    const double* ap = a.data();
    const double* bp = b.data();
    double* dp = out.mutable_data();
    const double inf = std::numeric_limits<double>::infinity();
    std::fill(dp, dp + (n + 1) * (m + 1), inf);
    dp[0] = 0.0;
    const py::ssize_t cols = m + 1;
    for (py::ssize_t i = 1; i <= n; ++i) {
        for (py::ssize_t j = 1; j <= m; ++j) {
            const double cost = std::abs(ap[i - 1] - bp[j - 1]);
            const double prev = std::min({dp[(i - 1) * cols + j], dp[i * cols + j - 1], dp[(i - 1) * cols + j - 1]});
            dp[i * cols + j] = cost + prev;
        }
    }
    return out;
}

static double dtw_distance_native(ArrayD a, ArrayD b, py::object window) {
    if (a.ndim() != 1 || b.ndim() != 1) {
        throw std::invalid_argument("1D arrays expected");
    }
    const py::ssize_t n = a.shape(0);
    const py::ssize_t m = b.shape(0);
    py::ssize_t w = std::max(n, m);
    if (!window.is_none()) {
        w = window.cast<py::ssize_t>();
    }
    std::vector<double> d(static_cast<std::size_t>((n + 1) * (m + 1)), std::numeric_limits<double>::infinity());
    const double* ap = a.data();
    const double* bp = b.data();
    const py::ssize_t cols = m + 1;
    d[0] = 0.0;
    for (py::ssize_t i = 1; i <= n; ++i) {
        const py::ssize_t jlo = std::max(static_cast<py::ssize_t>(1), i - w);
        const py::ssize_t jhi = std::min(m, i + w);
        for (py::ssize_t j = jlo; j <= jhi; ++j) {
            const double cost = std::abs(ap[i - 1] - bp[j - 1]);
            const double prev = std::min({d[static_cast<std::size_t>((i - 1) * cols + j)], d[static_cast<std::size_t>(i * cols + j - 1)], d[static_cast<std::size_t>((i - 1) * cols + j - 1)]});
            d[static_cast<std::size_t>(i * cols + j)] = cost + prev;
        }
    }
    return d[static_cast<std::size_t>(n * cols + m)];
}

static py::array_t<long long> dbscan_native(ArrayD points, double eps, int min_samples) {
    if (points.ndim() != 2) {
        throw std::invalid_argument("points must have shape (N, D)");
    }
    const py::ssize_t n = points.shape(0);
    const py::ssize_t dim = points.shape(1);
    const double* xp = points.data();
    const double eps2 = eps * eps;
    std::vector<std::vector<int>> neighbors(static_cast<std::size_t>(n));
    for (py::ssize_t i = 0; i < n; ++i) {
        for (py::ssize_t j = 0; j < n; ++j) {
            double dist2 = 0.0;
            for (py::ssize_t c = 0; c < dim; ++c) {
                const double diff = xp[i * dim + c] - xp[j * dim + c];
                dist2 += diff * diff;
            }
            if (dist2 <= eps2) {
                neighbors[static_cast<std::size_t>(i)].push_back(static_cast<int>(j));
            }
        }
    }

    auto labels = py::array_t<long long>({n});
    long long* lp = labels.mutable_data();
    std::fill(lp, lp + n, -1);
    long long cluster = 0;
    for (py::ssize_t i = 0; i < n; ++i) {
        if (lp[i] != -1) {
            continue;
        }
        if (static_cast<int>(neighbors[static_cast<std::size_t>(i)].size()) < min_samples) {
            continue;
        }
        lp[i] = cluster;
        std::vector<int> seeds = neighbors[static_cast<std::size_t>(i)];
        while (!seeds.empty()) {
            const int j = seeds.back();
            seeds.pop_back();
            if (lp[j] == -1) {
                lp[j] = cluster;
            } else if (lp[j] != cluster) {
                continue;
            }
            if (static_cast<int>(neighbors[static_cast<std::size_t>(j)].size()) >= min_samples) {
                for (const int candidate : neighbors[static_cast<std::size_t>(j)]) {
                    if (lp[candidate] == -1) {
                        seeds.push_back(candidate);
                    }
                }
            }
        }
        cluster += 1;
    }
    return labels;
}

static int cusum_detect_native(ArrayD x, double threshold, double mean, double sigma, double k) {
    if (x.ndim() != 1) {
        throw std::invalid_argument("x must have shape (N,)");
    }
    const double* xp = x.data();
    double s_pos = 0.0;
    double s_neg = 0.0;
    for (py::ssize_t i = 0; i < x.shape(0); ++i) {
        const double z = (xp[i] - mean) / sigma;
        s_pos = std::max(0.0, s_pos + z - k);
        s_neg = std::min(0.0, s_neg + z + k);
        if (s_pos > threshold || -s_neg > threshold) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

static py::dict rom_energy_spectrum_native(ArrayD singular_values) {
    if (singular_values.ndim() != 1) {
        throw std::invalid_argument("singular_values must have shape (N,)");
    }
    const double* sv = singular_values.data();
    const py::ssize_t n = singular_values.shape(0);
    double total = 1e-30;
    for (py::ssize_t i = 0; i < n; ++i) {
        total += sv[i] * sv[i];
    }
    py::dict out;
    out["total"] = 1.0;
    const int ks[5] = {1, 3, 5, 10, 20};
    for (const int k : ks) {
        if (k <= n) {
            double partial = 0.0;
            for (int i = 0; i < k; ++i) {
                partial += sv[i] * sv[i];
            }
            out[py::str(std::string("top") + std::to_string(k))] = partial / total;
        }
    }
    return out;
}

static py::array_t<double> radial_energy_sum(ArrayD k_values, ArrayD energy, ArrayD edges) {
    if (k_values.ndim() != 2 || energy.ndim() != 2 || edges.ndim() != 1) {
        throw std::invalid_argument("K and energy must be 2D, edges must be 1D");
    }
    if (k_values.shape(0) != energy.shape(0) || k_values.shape(1) != energy.shape(1)) {
        throw std::invalid_argument("K and energy shapes must match");
    }
    const py::ssize_t bins = edges.shape(0) - 1;
    auto out = py::array_t<double>({bins});
    const double* kp = k_values.data();
    const double* ep = energy.data();
    const double* edp = edges.data();
    double* op = out.mutable_data();
    std::fill(op, op + bins, 0.0);
    const py::ssize_t total = k_values.shape(0) * k_values.shape(1);
    for (py::ssize_t idx = 0; idx < total; ++idx) {
        const double kval = kp[idx];
        const double eval = ep[idx];
        for (py::ssize_t bin = 0; bin < bins; ++bin) {
            if (kval >= edp[bin] && kval < edp[bin + 1]) {
                op[bin] += eval;
                break;
            }
        }
    }
    return out;
}

static py::list dominant_frequencies_from_power(ArrayD freqs, ArrayD power, int top_k) {
    if (freqs.ndim() != 1 || power.ndim() != 1 || freqs.shape(0) != power.shape(0)) {
        throw std::invalid_argument("freqs and power must be matching 1D arrays");
    }
    const py::ssize_t n = freqs.shape(0);
    const double* fp = freqs.data();
    const double* pp = power.data();
    std::vector<py::ssize_t> idx(static_cast<std::size_t>(n));
    for (py::ssize_t i = 0; i < n; ++i) {
        idx[static_cast<std::size_t>(i)] = i;
    }
    std::sort(idx.begin(), idx.end(), [pp](py::ssize_t lhs, py::ssize_t rhs) {
        return pp[lhs] > pp[rhs];
    });
    const py::ssize_t count = std::min(static_cast<py::ssize_t>(top_k), n);
    py::list out;
    for (py::ssize_t i = 0; i < count; ++i) {
        const py::ssize_t j = idx[static_cast<std::size_t>(i)];
        out.append(py::make_tuple(fp[j], pp[j]));
    }
    return out;
}

static py::array_t<double> deposit_cic_1d(ArrayD x, ArrayD weights, int n_grid, double dx, double x0) {
    if (x.ndim() != 1 || weights.ndim() != 1 || x.shape(0) != weights.shape(0)) {
        throw std::invalid_argument("x and weights must be matching 1D arrays");
    }
    if (n_grid < 0) {
        throw std::invalid_argument("n_grid must be non-negative");
    }
    if (dx == 0.0) {
        throw std::invalid_argument("dx must be non-zero");
    }
    auto out = py::array_t<double>({static_cast<py::ssize_t>(n_grid)});
    double* op = out.mutable_data();
    std::fill(op, op + n_grid, 0.0);

    const double* xp = x.data();
    const double* wp = weights.data();
    const py::ssize_t n = x.shape(0);
    for (py::ssize_t k = 0; k < n; ++k) {
        const double pos = (xp[k] - x0) / dx;
        const auto i = static_cast<py::ssize_t>(std::floor(pos));
        const double f = pos - static_cast<double>(i);
        if (0 <= i && i < n_grid) {
            op[i] += wp[k] * (1.0 - f);
        }
        if (0 <= i + 1 && i + 1 < n_grid) {
            op[i + 1] += wp[k] * f;
        }
    }
    return out;
}

static inline double cubic_spline_1d_value(double r, double h) {
    const double q = std::abs(r) / h;
    const double sigma = 2.0 / 3.0 / h;
    if (q < 1.0) {
        return sigma * (1.0 - 1.5 * q * q + 0.75 * q * q * q);
    }
    if (q < 2.0) {
        const double d = 2.0 - q;
        return sigma * 0.25 * d * d * d;
    }
    return 0.0;
}

static py::array_t<double> sph_density_1d(ArrayD positions, ArrayD masses, double h) {
    if (positions.ndim() != 1 || masses.ndim() != 1 || positions.shape(0) != masses.shape(0)) {
        throw std::invalid_argument("positions and masses must be matching 1D arrays");
    }
    if (h == 0.0) {
        throw std::invalid_argument("h must be non-zero");
    }
    const py::ssize_t n = positions.shape(0);
    auto out = py::array_t<double>({n});
    const double* pp = positions.data();
    const double* mp = masses.data();
    double* op = out.mutable_data();
    for (py::ssize_t i = 0; i < n; ++i) {
        double rho = 0.0;
        for (py::ssize_t j = 0; j < n; ++j) {
            rho += mp[j] * cubic_spline_1d_value(pp[j] - pp[i], h);
        }
        op[i] = rho;
    }
    return out;
}

static double reaction_rate_native(double k, py::sequence concentrations, py::sequence orders) {
    const auto n = py::len(concentrations);
    if (py::len(orders) != n) {
        throw py::value_error("zip() argument 2 is shorter or longer than argument 1");
    }
    double rate = k;
    for (py::size_t i = 0; i < n; ++i) {
        const double c = py::cast<double>(concentrations[i]);
        const double order = py::cast<double>(orders[i]);
        rate *= std::pow(std::max(c, 0.0), order);
    }
    return rate;
}

static py::array_t<double> vof_step_1d(ArrayD alpha, ArrayD u, double dt, double dx) {
    if (alpha.ndim() != 1 || u.ndim() != 1 || alpha.shape(0) != u.shape(0)) {
        throw std::invalid_argument("alpha and u must be matching 1D arrays");
    }
    if (dx == 0.0) {
        throw std::invalid_argument("dx must be non-zero");
    }
    const py::ssize_t n = alpha.shape(0);
    auto out = py::array_t<double>({n});
    const double* ap = alpha.data();
    const double* up = u.data();
    double* op = out.mutable_data();
    for (py::ssize_t i = 0; i < n; ++i) {
        op[i] = ap[i];
    }
    const double c = dt / dx;
    for (py::ssize_t i = 1; i < n - 1; ++i) {
        if (up[i] >= 0.0) {
            op[i] = ap[i] - c * up[i] * (ap[i] - ap[i - 1]);
        } else {
            op[i] = ap[i] - c * up[i] * (ap[i + 1] - ap[i]);
        }
    }
    for (py::ssize_t i = 0; i < n; ++i) {
        op[i] = std::min(1.0, std::max(0.0, op[i]));
    }
    return out;
}

static py::array_t<double> levelset_advect_step_1d(ArrayD phi, ArrayD u, double dt, double dx) {
    if (phi.ndim() != 1 || u.ndim() != 1 || phi.shape(0) != u.shape(0)) {
        throw std::invalid_argument("phi and u must be matching 1D arrays");
    }
    if (dx == 0.0) {
        throw std::invalid_argument("dx must be non-zero");
    }
    const py::ssize_t n = phi.shape(0);
    auto out = py::array_t<double>({n});
    const double* pp = phi.data();
    const double* up = u.data();
    double* op = out.mutable_data();
    for (py::ssize_t i = 0; i < n; ++i) {
        op[i] = pp[i];
    }
    const double c = dt / dx;
    for (py::ssize_t i = 1; i < n - 1; ++i) {
        if (up[i] >= 0.0) {
            op[i] = pp[i] - c * up[i] * (pp[i] - pp[i - 1]);
        } else {
            op[i] = pp[i] - c * up[i] * (pp[i + 1] - pp[i]);
        }
    }
    return out;
}

static inline double jensen_wake_velocity(double v0, double x, double r, double a, double k) {
    if (x <= 0.0) {
        return v0;
    }
    const double denom = 1.0 + k * x / r;
    const double deficit = 2.0 * a / (denom * denom);
    return v0 * (1.0 - deficit);
}

static py::list jensen_farm_velocity(double v0, py::sequence distances, double r, double a, double k) {
    py::list out;
    out.append(v0);
    double current = v0;
    const auto n = py::len(distances);
    for (py::size_t i = 0; i < n; ++i) {
        current = jensen_wake_velocity(current, py::cast<double>(distances[i]), r, a, k);
        out.append(current);
    }
    return out;
}

static py::array_t<double> conservative_remap_1d(ArrayD x_old_edges, ArrayD u_old, ArrayD x_new_edges) {
    if (x_old_edges.ndim() != 1 || u_old.ndim() != 1 || x_new_edges.ndim() != 1) {
        throw std::invalid_argument("x_old_edges, u_old, and x_new_edges must be 1D arrays");
    }
    if (x_old_edges.shape(0) != u_old.shape(0) + 1) {
        throw std::invalid_argument("u_old length must be one less than x_old_edges length");
    }
    const py::ssize_t n_new = x_new_edges.shape(0) - 1;
    auto out = py::array_t<double>({n_new});
    const double* xo = x_old_edges.data();
    const double* uo = u_old.data();
    const double* xn = x_new_edges.data();
    double* op = out.mutable_data();
    const py::ssize_t n_old = u_old.shape(0);
    for (py::ssize_t i = 0; i < n_new; ++i) {
        const double a = xn[i];
        const double b = xn[i + 1];
        double total = 0.0;
        for (py::ssize_t j = 0; j < n_old; ++j) {
            const double ov_a = std::max(a, xo[j]);
            const double ov_b = std::min(b, xo[j + 1]);
            if (ov_b > ov_a) {
                total += (ov_b - ov_a) * uo[j];
            }
        }
        op[i] = total / (b - a);
    }
    return out;
}

static py::tuple fd_heat_1d_evolve(ArrayD u0, int n_steps, double coef, double dt) {
    if (u0.ndim() != 1) {
        throw std::invalid_argument("u0 must be a 1D array");
    }
    if (n_steps < 0) {
        throw std::invalid_argument("n_steps must be non-negative");
    }
    const py::ssize_t nx = u0.shape(0);
    const py::ssize_t nt = static_cast<py::ssize_t>(n_steps) + 1;
    auto t = py::array_t<double>({nt});
    auto U = py::array_t<double>({nx, nt});
    double* tp = t.mutable_data();
    double* Up = U.mutable_data();
    std::vector<double> u(static_cast<std::size_t>(nx));
    std::vector<double> u_new(static_cast<std::size_t>(nx));
    const double* u0p = u0.data();
    for (py::ssize_t i = 0; i < nx; ++i) {
        u[static_cast<std::size_t>(i)] = u0p[i];
        Up[i * nt] = u0p[i];
    }
    tp[0] = 0.0;
    for (int k = 0; k < n_steps; ++k) {
        u_new = u;
        for (py::ssize_t i = 1; i < nx - 1; ++i) {
            u_new[static_cast<std::size_t>(i)] = u[static_cast<std::size_t>(i)]
                + coef * (u[static_cast<std::size_t>(i + 1)] - 2.0 * u[static_cast<std::size_t>(i)] + u[static_cast<std::size_t>(i - 1)]);
        }
        if (nx > 0) {
            u_new[0] = 0.0;
            u_new[static_cast<std::size_t>(nx - 1)] = 0.0;
        }
        u.swap(u_new);
        const py::ssize_t col = static_cast<py::ssize_t>(k) + 1;
        for (py::ssize_t i = 0; i < nx; ++i) {
            Up[i * nt + col] = u[static_cast<std::size_t>(i)];
        }
        tp[col] = static_cast<double>(k + 1) * dt;
    }
    return py::make_tuple(t, U);
}

static py::tuple fd_burgers_1d_evolve(ArrayD u0, int n_steps, double dt, double dx, double nu) {
    if (u0.ndim() != 1) {
        throw std::invalid_argument("u0 must be a 1D array");
    }
    if (n_steps < 0) {
        throw std::invalid_argument("n_steps must be non-negative");
    }
    if (dx == 0.0) {
        throw std::invalid_argument("dx must be non-zero");
    }
    const py::ssize_t nx = u0.shape(0);
    const py::ssize_t nt = static_cast<py::ssize_t>(n_steps) + 1;
    auto t = py::array_t<double>({nt});
    auto U = py::array_t<double>({nx, nt});
    double* tp = t.mutable_data();
    double* Up = U.mutable_data();
    std::vector<double> u(static_cast<std::size_t>(nx));
    std::vector<double> u_new(static_cast<std::size_t>(nx));
    const double* u0p = u0.data();
    for (py::ssize_t i = 0; i < nx; ++i) {
        u[static_cast<std::size_t>(i)] = u0p[i];
        Up[i * nt] = u0p[i];
    }
    tp[0] = 0.0;
    for (int k = 0; k < n_steps; ++k) {
        u_new = u;
        for (py::ssize_t i = 1; i < nx - 1; ++i) {
            const double du = (u[static_cast<std::size_t>(i + 1)] - u[static_cast<std::size_t>(i - 1)]) / (2.0 * dx);
            const double d2u = (
                u[static_cast<std::size_t>(i + 1)] - 2.0 * u[static_cast<std::size_t>(i)] + u[static_cast<std::size_t>(i - 1)]
            ) / (dx * dx);
            u_new[static_cast<std::size_t>(i)] = u[static_cast<std::size_t>(i)] + dt * (-u[static_cast<std::size_t>(i)] * du + nu * d2u);
        }
        if (nx > 0) {
            u_new[0] = 0.0;
            u_new[static_cast<std::size_t>(nx - 1)] = 0.0;
        }
        u.swap(u_new);
        const py::ssize_t col = static_cast<py::ssize_t>(k) + 1;
        for (py::ssize_t i = 0; i < nx; ++i) {
            Up[i * nt + col] = u[static_cast<std::size_t>(i)];
        }
        tp[col] = static_cast<double>(k + 1) * dt;
    }
    return py::make_tuple(t, U);
}

static py::tuple poisson_2d_jacobi_native(Array2D f, double dx, double dy, int max_iter, double tol) {
    if (dx == 0.0 || dy == 0.0) {
        throw std::invalid_argument("dx and dy must be non-zero");
    }
    if (max_iter < 0) {
        throw std::invalid_argument("max_iter must be non-negative");
    }
    const py::ssize_t nx = f.shape(0);
    const py::ssize_t ny = f.shape(1);
    auto out = py::array_t<double>({nx, ny});
    double* op = out.mutable_data();
    const double* fp = f.data();
    const py::ssize_t total = nx * ny;
    std::vector<double> p(static_cast<std::size_t>(total), 0.0);
    std::vector<double> p_new(static_cast<std::size_t>(total), 0.0);
    const double inv_dx2 = 1.0 / (dx * dx);
    const double inv_dy2 = 1.0 / (dy * dy);
    const double denom = 2.0 * (inv_dx2 + inv_dy2);
    double err = 0.0;
    for (int it = 0; it < max_iter; ++it) {
        p_new = p;
        for (py::ssize_t i = 1; i < nx - 1; ++i) {
            for (py::ssize_t j = 1; j < ny - 1; ++j) {
                const py::ssize_t idx = i * ny + j;
                p_new[static_cast<std::size_t>(idx)] = (
                    (p[static_cast<std::size_t>((i + 1) * ny + j)] + p[static_cast<std::size_t>((i - 1) * ny + j)]) * inv_dx2
                    + (p[static_cast<std::size_t>(i * ny + j + 1)] + p[static_cast<std::size_t>(i * ny + j - 1)]) * inv_dy2
                    - fp[idx]
                ) / denom;
            }
        }
        for (py::ssize_t i = 0; i < nx; ++i) {
            p_new[static_cast<std::size_t>(i * ny)] = 0.0;
            p_new[static_cast<std::size_t>(i * ny + ny - 1)] = 0.0;
        }
        for (py::ssize_t j = 0; j < ny; ++j) {
            p_new[static_cast<std::size_t>(j)] = 0.0;
            p_new[static_cast<std::size_t>((nx - 1) * ny + j)] = 0.0;
        }
        err = 0.0;
        for (py::ssize_t idx = 0; idx < total; ++idx) {
            err = std::max(err, std::abs(p_new[static_cast<std::size_t>(idx)] - p[static_cast<std::size_t>(idx)]));
        }
        p.swap(p_new);
        if (err < tol) {
            for (py::ssize_t idx = 0; idx < total; ++idx) {
                op[idx] = p[static_cast<std::size_t>(idx)];
            }
            py::dict info;
            info["iters"] = it + 1;
            info["err"] = err;
            info["converged"] = true;
            return py::make_tuple(out, info);
        }
    }
    for (py::ssize_t idx = 0; idx < total; ++idx) {
        op[idx] = p[static_cast<std::size_t>(idx)];
    }
    py::dict info;
    info["iters"] = max_iter;
    info["err"] = err;
    info["converged"] = false;
    return py::make_tuple(out, info);
}

static py::array_t<double> levelset_reinit_1d(ArrayD phi, double dx, int n_iter) {
    if (phi.ndim() != 1) {
        throw std::invalid_argument("phi must be a 1D array");
    }
    if (dx == 0.0) {
        throw std::invalid_argument("dx must be non-zero");
    }
    if (n_iter < 0) {
        throw std::invalid_argument("n_iter must be non-negative");
    }
    const py::ssize_t n = phi.shape(0);
    const double* phip = phi.data();
    std::vector<double> p(static_cast<std::size_t>(n));
    std::vector<double> p_new(static_cast<std::size_t>(n));
    std::vector<double> s0(static_cast<std::size_t>(n));
    for (py::ssize_t i = 0; i < n; ++i) {
        p[static_cast<std::size_t>(i)] = phip[i];
        s0[static_cast<std::size_t>(i)] = (phip[i] > 0.0) ? 1.0 : ((phip[i] < 0.0) ? -1.0 : 0.0);
    }
    const double dt = 0.3 * dx;
    for (int iter = 0; iter < n_iter; ++iter) {
        for (py::ssize_t i = 0; i < n; ++i) {
            const double dxp = (i == n - 1) ? 0.0 : (p[static_cast<std::size_t>(i + 1)] - p[static_cast<std::size_t>(i)]) / dx;
            const double dxm = (i == 0) ? 0.0 : (p[static_cast<std::size_t>(i)] - p[static_cast<std::size_t>(i - 1)]) / dx;
            double gp;
            if (s0[static_cast<std::size_t>(i)] > 0.0) {
                const double a = std::max(dxm, 0.0);
                const double b = std::min(dxp, 0.0);
                gp = std::max(a * a, b * b);
            } else {
                const double a = std::max(dxp, 0.0);
                const double b = std::min(dxm, 0.0);
                gp = std::max(a * a, b * b);
            }
            const double grad = std::sqrt(gp);
            p_new[static_cast<std::size_t>(i)] = p[static_cast<std::size_t>(i)] - dt * s0[static_cast<std::size_t>(i)] * (grad - 1.0);
        }
        p.swap(p_new);
    }
    auto out = py::array_t<double>({n});
    double* op = out.mutable_data();
    for (py::ssize_t i = 0; i < n; ++i) {
        op[i] = p[static_cast<std::size_t>(i)];
    }
    return out;
}

static py::array_t<double> fast_march_1d(ArrayD phi, double dx) {
    if (phi.ndim() != 1) {
        throw std::invalid_argument("phi must be a 1D array");
    }
    const py::ssize_t n = phi.shape(0);
    const double* pp = phi.data();
    std::vector<double> iface_locs;
    iface_locs.reserve(static_cast<std::size_t>(n));
    for (py::ssize_t i = 0; i < n - 1; ++i) {
        if (pp[i] == 0.0) {
            iface_locs.push_back(static_cast<double>(i) * dx);
        } else if (pp[i] * pp[i + 1] < 0.0) {
            const double t = pp[i] / (pp[i] - pp[i + 1]);
            iface_locs.push_back((static_cast<double>(i) + t) * dx);
        }
    }
    auto out = py::array_t<double>({n});
    double* op = out.mutable_data();
    if (iface_locs.empty()) {
        for (py::ssize_t i = 0; i < n; ++i) {
            op[i] = pp[i];
        }
        return out;
    }
    for (py::ssize_t i = 0; i < n; ++i) {
        const double x = static_cast<double>(i) * dx;
        double d = std::numeric_limits<double>::infinity();
        for (const double loc : iface_locs) {
            d = std::min(d, std::abs(x - loc));
        }
        const double sign = (pp[i] > 0.0) ? 1.0 : ((pp[i] < 0.0) ? -1.0 : 0.0);
        op[i] = d * sign;
    }
    return out;
}

static py::tuple mesh_refine_by_gradient(ArrayD x_in, ArrayD f_in, double threshold, int max_passes) {
    if (x_in.ndim() != 1 || f_in.ndim() != 1 || x_in.shape(0) != f_in.shape(0)) {
        throw std::invalid_argument("x and f must be matching 1D arrays");
    }
    if (max_passes < 0) {
        throw std::invalid_argument("max_passes must be non-negative");
    }
    std::vector<double> x(x_in.data(), x_in.data() + x_in.shape(0));
    std::vector<double> f(f_in.data(), f_in.data() + f_in.shape(0));
    for (int pass = 0; pass < max_passes; ++pass) {
        const std::size_t n = x.size();
        bool any = false;
        std::vector<char> mask(n > 0 ? n - 1 : 0, 0);
        for (std::size_t i = 0; i + 1 < n; ++i) {
            if (std::abs(f[i + 1] - f[i]) > threshold) {
                mask[i] = 1;
                any = true;
            }
        }
        if (!any) {
            break;
        }
        std::vector<double> new_x;
        std::vector<double> new_f;
        new_x.reserve(n * 2);
        new_f.reserve(n * 2);
        for (std::size_t i = 0; i + 1 < n; ++i) {
            new_x.push_back(x[i]);
            new_f.push_back(f[i]);
            if (mask[i]) {
                new_x.push_back(0.5 * (x[i] + x[i + 1]));
                new_f.push_back(0.5 * (f[i] + f[i + 1]));
            }
        }
        if (n > 0) {
            new_x.push_back(x.back());
            new_f.push_back(f.back());
        }
        x.swap(new_x);
        f.swap(new_f);
    }
    auto x_out = py::array_t<double>({static_cast<py::ssize_t>(x.size())});
    auto f_out = py::array_t<double>({static_cast<py::ssize_t>(f.size())});
    std::copy(x.begin(), x.end(), x_out.mutable_data());
    std::copy(f.begin(), f.end(), f_out.mutable_data());
    return py::make_tuple(x_out, f_out);
}

static py::tuple mesh_coarsen_by_tolerance(ArrayD x, ArrayD f, double tol) {
    if (x.ndim() != 1 || f.ndim() != 1 || x.shape(0) != f.shape(0)) {
        throw std::invalid_argument("x and f must be matching 1D arrays");
    }
    const py::ssize_t n = x.shape(0);
    const double* xp = x.data();
    const double* fp = f.data();
    std::vector<double> x_keep;
    std::vector<double> f_keep;
    x_keep.reserve(static_cast<std::size_t>(n));
    f_keep.reserve(static_cast<std::size_t>(n));
    for (py::ssize_t i = 0; i < n; ++i) {
        bool keep = (i == 0 || i == n - 1);
        if (!keep) {
            const double d2 = fp[i - 1] - 2.0 * fp[i] + fp[i + 1];
            keep = std::abs(d2) >= tol;
        }
        if (keep) {
            x_keep.push_back(xp[i]);
            f_keep.push_back(fp[i]);
        }
    }
    auto x_out = py::array_t<double>({static_cast<py::ssize_t>(x_keep.size())});
    auto f_out = py::array_t<double>({static_cast<py::ssize_t>(f_keep.size())});
    std::copy(x_keep.begin(), x_keep.end(), x_out.mutable_data());
    std::copy(f_keep.begin(), f_keep.end(), f_out.mutable_data());
    return py::make_tuple(x_out, f_out);
}

static py::array_t<double> laplacian_smooth_native(ArrayD verts, ArrayI edges, int n_iter, double alpha, ArrayI fixed) {
    if (verts.ndim() != 2) {
        throw std::invalid_argument("verts must be a 2D array");
    }
    if (edges.ndim() != 2 || edges.shape(1) != 2) {
        throw std::invalid_argument("edges must be a 2D array with shape (n_edges, 2)");
    }
    if (fixed.ndim() != 1) {
        throw std::invalid_argument("fixed must be a 1D array");
    }
    if (n_iter < 0) {
        throw std::invalid_argument("n_iter must be non-negative");
    }
    const py::ssize_t n = verts.shape(0);
    const py::ssize_t dim = verts.shape(1);
    const py::ssize_t n_edges = edges.shape(0);
    std::vector<std::vector<py::ssize_t>> adj(static_cast<std::size_t>(n));
    const long long* ep = edges.data();
    for (py::ssize_t e = 0; e < n_edges; ++e) {
        const auto a = static_cast<py::ssize_t>(ep[2 * e]);
        const auto b = static_cast<py::ssize_t>(ep[2 * e + 1]);
        if (a < 0 || b < 0 || a >= n || b >= n) {
            throw std::out_of_range("edge index out of range");
        }
        adj[static_cast<std::size_t>(a)].push_back(b);
        adj[static_cast<std::size_t>(b)].push_back(a);
    }
    std::vector<char> is_fixed(static_cast<std::size_t>(n), 0);
    const long long* fp = fixed.data();
    for (py::ssize_t i = 0; i < fixed.shape(0); ++i) {
        const auto idx = static_cast<py::ssize_t>(fp[i]);
        if (idx >= 0 && idx < n) {
            is_fixed[static_cast<std::size_t>(idx)] = 1;
        }
    }
    const py::ssize_t total = n * dim;
    std::vector<double> v(verts.data(), verts.data() + total);
    std::vector<double> v_new(static_cast<std::size_t>(total));
    for (int iter = 0; iter < n_iter; ++iter) {
        v_new = v;
        for (py::ssize_t i = 0; i < n; ++i) {
            const auto& neighbors = adj[static_cast<std::size_t>(i)];
            if (is_fixed[static_cast<std::size_t>(i)] || neighbors.empty()) {
                continue;
            }
            for (py::ssize_t d = 0; d < dim; ++d) {
                double mean = 0.0;
                for (const py::ssize_t j : neighbors) {
                    mean += v[static_cast<std::size_t>(j * dim + d)];
                }
                mean /= static_cast<double>(neighbors.size());
                const py::ssize_t idx = i * dim + d;
                v_new[static_cast<std::size_t>(idx)] = (1.0 - alpha) * v[static_cast<std::size_t>(idx)] + alpha * mean;
            }
        }
        v.swap(v_new);
    }
    auto out = py::array_t<double>({n, dim});
    std::copy(v.begin(), v.end(), out.mutable_data());
    return out;
}

static inline double dist2d(const double* p, py::ssize_t dim, py::ssize_t a, py::ssize_t b) {
    const double dx = p[a * dim] - p[b * dim];
    const double dy = p[a * dim + 1] - p[b * dim + 1];
    return std::sqrt(dx * dx + dy * dy);
}

static inline double triangle_aspect_from_points(const double* p, py::ssize_t dim, py::ssize_t ia, py::ssize_t ib, py::ssize_t ic) {
    const double a = dist2d(p, dim, ib, ic);
    const double b = dist2d(p, dim, ia, ic);
    const double c = dist2d(p, dim, ia, ib);
    const double s = 0.5 * (a + b + c);
    const double area_arg = std::max(s * (s - a) * (s - b) * (s - c), 1e-30);
    const double area = std::max(std::sqrt(area_arg), 1e-30);
    const double R = (a * b * c) / (4.0 * area);
    const double r = area / s;
    return R / (2.0 * r);
}

static inline double triangle_angle_deg(const double* p, py::ssize_t dim, py::ssize_t ip, py::ssize_t iq, py::ssize_t ir) {
    const double ux = p[iq * dim] - p[ip * dim];
    const double uy = p[iq * dim + 1] - p[ip * dim + 1];
    const double vx = p[ir * dim] - p[ip * dim];
    const double vy = p[ir * dim + 1] - p[ip * dim + 1];
    const double un = std::sqrt(ux * ux + uy * uy);
    const double vn = std::sqrt(vx * vx + vy * vy);
    double cosv = (ux * vx + uy * vy) / (un * vn + 1e-30);
    cosv = std::min(1.0, std::max(-1.0, cosv));
    return std::acos(cosv) * 180.0 / 3.141592653589793238462643383279502884;
}

static py::dict mesh_quality_report_native(ArrayD points, ArrayI simplices) {
    if (points.ndim() != 2 || points.shape(1) < 2) {
        throw std::invalid_argument("points must have shape (n, >=2)");
    }
    if (simplices.ndim() != 2 || simplices.shape(1) != 3) {
        throw std::invalid_argument("simplices must have shape (n, 3)");
    }
    const py::ssize_t dim = points.shape(1);
    const py::ssize_t n_tri = simplices.shape(0);
    const double* pp = points.data();
    const long long* sp = simplices.data();
    double sum_aspect = 0.0;
    double max_aspect = -std::numeric_limits<double>::infinity();
    double sum_skew = 0.0;
    double max_skew = -std::numeric_limits<double>::infinity();
    double min_angle = std::numeric_limits<double>::infinity();
    for (py::ssize_t t = 0; t < n_tri; ++t) {
        const auto ia = static_cast<py::ssize_t>(sp[3 * t]);
        const auto ib = static_cast<py::ssize_t>(sp[3 * t + 1]);
        const auto ic = static_cast<py::ssize_t>(sp[3 * t + 2]);
        const double aspect = triangle_aspect_from_points(pp, dim, ia, ib, ic);
        const double angle = std::min({
            triangle_angle_deg(pp, dim, ia, ib, ic),
            triangle_angle_deg(pp, dim, ib, ia, ic),
            triangle_angle_deg(pp, dim, ic, ia, ib),
        });
        const double skew = std::max(0.0, (60.0 - angle) / 60.0);
        sum_aspect += aspect;
        max_aspect = std::max(max_aspect, aspect);
        sum_skew += skew;
        max_skew = std::max(max_skew, skew);
        min_angle = std::min(min_angle, angle);
    }
    py::dict out;
    out["mean_aspect"] = sum_aspect / static_cast<double>(n_tri);
    out["max_aspect"] = max_aspect;
    out["mean_skew"] = sum_skew / static_cast<double>(n_tri);
    out["max_skew"] = max_skew;
    out["min_angle_deg"] = min_angle;
    return out;
}

static py::dict order_table_native(ArrayD h, ArrayD err) {
    if (h.ndim() != 1 || err.ndim() != 1 || h.shape(0) != err.shape(0)) {
        throw std::invalid_argument("h and err must be matching 1D arrays");
    }
    const py::ssize_t n = h.shape(0);
    const double* hp = h.data();
    const double* ep = err.data();
    py::list h_list;
    py::list err_list;
    py::list p_pair;
    for (py::ssize_t i = 0; i < n; ++i) {
        h_list.append(hp[i]);
        err_list.append(ep[i]);
    }
    for (py::ssize_t i = 0; i < n - 1; ++i) {
        const double p = std::log(ep[i] / ep[i + 1]) / std::log(hp[i] / hp[i + 1]);
        p_pair.append(p);
    }
    py::dict out;
    out["h"] = h_list;
    out["err"] = err_list;
    out["p_pair"] = p_pair;
    return out;
}

static inline double cubic_spline_grad_1d_value(double r, double h) {
    const double q = std::abs(r) / h;
    const double sigma = 2.0 / 3.0 / h;
    const double sign = (r > 0.0) ? 1.0 : ((r < 0.0) ? -1.0 : 0.0);
    if (q < 1.0) {
        return sigma * sign * (-3.0 * q + 2.25 * q * q) / h;
    }
    if (q < 2.0) {
        const double d = 2.0 - q;
        return sigma * sign * -0.75 * d * d / h;
    }
    return 0.0;
}

static py::array_t<double> sph_acceleration_1d(ArrayD x, ArrayD m, ArrayD rho, ArrayD p, double h) {
    if (x.ndim() != 1 || m.ndim() != 1 || rho.ndim() != 1 || p.ndim() != 1) {
        throw std::invalid_argument("x, m, rho, and p must be 1D arrays");
    }
    const py::ssize_t n = x.shape(0);
    if (m.shape(0) != n || rho.shape(0) != n || p.shape(0) != n) {
        throw std::invalid_argument("x, m, rho, and p must have matching lengths");
    }
    const double* xp = x.data();
    const double* mp = m.data();
    const double* rp = rho.data();
    const double* pp = p.data();
    auto out = py::array_t<double>({n});
    double* op = out.mutable_data();
    for (py::ssize_t i = 0; i < n; ++i) {
        double acc = 0.0;
        const double pi_over_rhoi2 = pp[i] / (rp[i] * rp[i]);
        for (py::ssize_t j = 0; j < n; ++j) {
            const double dW = cubic_spline_grad_1d_value(xp[i] - xp[j], h);
            acc += mp[j] * (pi_over_rhoi2 + pp[j] / (rp[j] * rp[j])) * dW;
        }
        op[i] = -acc;
    }
    return out;
}

static constexpr int D2Q9_E[9][2] = {
    {0, 0}, {1, 0}, {0, 1}, {-1, 0}, {0, -1}, {1, 1}, {-1, 1}, {-1, -1}, {1, -1}
};

static constexpr double D2Q9_W[9] = {
    4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
};

static py::array_t<double> lbm_equilibrium(Array2D rho, py::array_t<double, py::array::c_style | py::array::forcecast> u) {
    if (u.ndim() != 3 || u.shape(0) != 2 || u.shape(1) != rho.shape(0) || u.shape(2) != rho.shape(1)) {
        throw std::invalid_argument("u must have shape (2, X, Y) matching rho");
    }
    const py::ssize_t nx = rho.shape(0);
    const py::ssize_t ny = rho.shape(1);
    auto out = py::array_t<double>({static_cast<py::ssize_t>(9), nx, ny});
    const double* rp = rho.data();
    const double* up = u.data();
    double* op = out.mutable_data();
    const py::ssize_t plane = nx * ny;
    for (int k = 0; k < 9; ++k) {
        for (py::ssize_t i = 0; i < nx; ++i) {
            for (py::ssize_t j = 0; j < ny; ++j) {
                const py::ssize_t idx = i * ny + j;
                const double ux = up[idx];
                const double uy = up[plane + idx];
                const double u_sq = ux * ux + uy * uy;
                const double eu = D2Q9_E[k][0] * ux + D2Q9_E[k][1] * uy;
                op[k * plane + idx] = D2Q9_W[k] * rp[idx] * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u_sq);
            }
        }
    }
    return out;
}

static py::array_t<double> lbm_step(ArrayD f, double omega) {
    if (f.ndim() != 3 || f.shape(0) != 9) {
        throw std::invalid_argument("f must have shape (9, X, Y)");
    }
    const py::ssize_t nx = f.shape(1);
    const py::ssize_t ny = f.shape(2);
    const py::ssize_t plane = nx * ny;
    const double* fp = f.data();
    std::vector<double> rho(static_cast<std::size_t>(plane), 0.0);
    std::vector<double> ux(static_cast<std::size_t>(plane), 0.0);
    std::vector<double> uy(static_cast<std::size_t>(plane), 0.0);
    std::vector<double> collided(static_cast<std::size_t>(9 * plane), 0.0);
    for (int k = 0; k < 9; ++k) {
        for (py::ssize_t idx = 0; idx < plane; ++idx) {
            const double val = fp[k * plane + idx];
            rho[static_cast<std::size_t>(idx)] += val;
            ux[static_cast<std::size_t>(idx)] += D2Q9_E[k][0] * val;
            uy[static_cast<std::size_t>(idx)] += D2Q9_E[k][1] * val;
        }
    }
    for (py::ssize_t idx = 0; idx < plane; ++idx) {
        const double denom = std::max(rho[static_cast<std::size_t>(idx)], 1e-30);
        ux[static_cast<std::size_t>(idx)] /= denom;
        uy[static_cast<std::size_t>(idx)] /= denom;
    }
    for (int k = 0; k < 9; ++k) {
        for (py::ssize_t idx = 0; idx < plane; ++idx) {
            const double u_sq = ux[static_cast<std::size_t>(idx)] * ux[static_cast<std::size_t>(idx)]
                + uy[static_cast<std::size_t>(idx)] * uy[static_cast<std::size_t>(idx)];
            const double eu = D2Q9_E[k][0] * ux[static_cast<std::size_t>(idx)] + D2Q9_E[k][1] * uy[static_cast<std::size_t>(idx)];
            const double feq = D2Q9_W[k] * rho[static_cast<std::size_t>(idx)] * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u_sq);
            collided[static_cast<std::size_t>(k * plane + idx)] = fp[k * plane + idx] - omega * (fp[k * plane + idx] - feq);
        }
    }
    auto out = py::array_t<double>({static_cast<py::ssize_t>(9), nx, ny});
    double* op = out.mutable_data();
    std::fill(op, op + 9 * plane, 0.0);
    for (int k = 0; k < 9; ++k) {
        const py::ssize_t sx = D2Q9_E[k][0];
        const py::ssize_t sy = D2Q9_E[k][1];
        for (py::ssize_t i = 0; i < nx; ++i) {
            for (py::ssize_t j = 0; j < ny; ++j) {
                const py::ssize_t dst_i = (i + sx + nx) % nx;
                const py::ssize_t dst_j = (j + sy + ny) % ny;
                op[k * plane + dst_i * ny + dst_j] = collided[static_cast<std::size_t>(k * plane + i * ny + j)];
            }
        }
    }
    return out;
}

static bool is_monotone_decreasing_native(py::sequence errs, double atol) {
    const auto n = py::len(errs);
    if (n < 2) {
        return true;
    }
    double prev = py::cast<double>(errs[0]);
    for (py::size_t i = 1; i < n; ++i) {
        const double cur = py::cast<double>(errs[i]);
        if (cur > prev + atol) {
            return false;
        }
        prev = cur;
    }
    return true;
}

static py::list convergence_ratio_native(py::sequence errs) {
    const auto n = py::len(errs);
    py::list out;
    if (n < 2) {
        return out;
    }
    double prev = py::cast<double>(errs[0]);
    for (py::size_t i = 1; i < n; ++i) {
        const double cur = py::cast<double>(errs[i]);
        out.append(cur / std::max(std::abs(prev), 1e-30));
        prev = cur;
    }
    return out;
}

static double combined_disc_uncertainty_native(py::sequence uncs) {
    const auto n = py::len(uncs);
    double total = 0.0;
    for (py::size_t i = 0; i < n; ++i) {
        const double u = py::cast<double>(uncs[i]);
        total += u * u;
    }
    return std::sqrt(total);
}

static double friction_colebrook_native(double re, double eps_over_d, int n_iter) {
    double f = 0.02;
    for (int i = 0; i < n_iter; ++i) {
        const double rhs = -2.0 * std::log10(eps_over_d / 3.7 + 2.51 / (re * std::sqrt(f) + 1e-30));
        const double f_new = 1.0 / (rhs * rhs);
        if (std::abs(f_new - f) < 1e-10) {
            return f_new;
        }
        f = f_new;
    }
    return f;
}

static double delta99_scan(ArrayD y, ArrayD u, double target) {
    if (y.ndim() != 1 || u.ndim() != 1 || y.shape(0) != u.shape(0)) {
        throw std::invalid_argument("y and u must be matching 1D arrays");
    }
    const py::ssize_t n = u.shape(0);
    if (n == 0) {
        throw std::invalid_argument("y and u must be non-empty");
    }
    const double* yp = y.data();
    const double* up = u.data();
    for (py::ssize_t i = 0; i < n - 1; ++i) {
        if ((up[i] <= target && target <= up[i + 1]) || (up[i] >= target && target >= up[i + 1])) {
            if (up[i + 1] == up[i]) {
                return yp[i];
            }
            const double frac = (target - up[i]) / (up[i + 1] - up[i]);
            return yp[i] + frac * (yp[i + 1] - yp[i]);
        }
    }
    return yp[n - 1];
}

static py::array_t<double> scale_to_bounds_native(ArrayD unit, Array2D bounds) {
    if (unit.ndim() != 2 || bounds.shape(1) != 2 || bounds.shape(0) != unit.shape(1)) {
        throw std::invalid_argument("unit must be (n, d), bounds must be (d, 2)");
    }
    const py::ssize_t n = unit.shape(0);
    const py::ssize_t dim = unit.shape(1);
    const double* up = unit.data();
    const double* bp = bounds.data();
    auto out = py::array_t<double>({n, dim});
    double* op = out.mutable_data();
    for (py::ssize_t i = 0; i < n; ++i) {
        for (py::ssize_t d = 0; d < dim; ++d) {
            const double lo = bp[d * 2];
            const double hi = bp[d * 2 + 1];
            op[i * dim + d] = lo + up[i * dim + d] * (hi - lo);
        }
    }
    return out;
}

static py::array_t<double> regular_grid_points(Array2D bounds, int per) {
    if (bounds.shape(1) != 2) {
        throw std::invalid_argument("bounds must have shape (d, 2)");
    }
    if (per < 1) {
        throw std::invalid_argument("per must be positive");
    }
    const py::ssize_t dim = bounds.shape(0);
    py::ssize_t total = 1;
    for (py::ssize_t d = 0; d < dim; ++d) {
        total *= per;
    }
    const double* bp = bounds.data();
    auto out = py::array_t<double>({total, dim});
    double* op = out.mutable_data();
    for (py::ssize_t row = 0; row < total; ++row) {
        py::ssize_t div = total;
        for (py::ssize_t d = 0; d < dim; ++d) {
            div /= per;
            const py::ssize_t idx = (row / div) % per;
            const double lo = bp[d * 2];
            const double hi = bp[d * 2 + 1];
            const double value = (per == 1) ? lo : lo + (hi - lo) * static_cast<double>(idx) / static_cast<double>(per - 1);
            op[row * dim + d] = value;
        }
    }
    return out;
}

static py::array_t<double> norm_cdf(ArrayD x) {
    const py::ssize_t n = x.size();
    auto out = py::array_t<double>({n});
    const double* xp = x.data();
    double* op = out.mutable_data();
    const double inv_sqrt2 = 0.707106781186547524400844362104849039;
    for (py::ssize_t i = 0; i < n; ++i) {
        op[i] = 0.5 * (1.0 + std::erf(xp[i] * inv_sqrt2));
    }
    return out;
}

static py::array_t<long long> greedy_batch_acquisition_native(
    ArrayD acq, Array2D candidates, int batch_size, double min_distance, bool use_distance
) {
    if (acq.ndim() != 1 || candidates.shape(0) != acq.shape(0)) {
        throw std::invalid_argument("acq must be 1D and candidates must have matching rows");
    }
    if (batch_size <= 0) {
        throw std::invalid_argument("batch_size must be positive");
    }
    const py::ssize_t n = acq.shape(0);
    const py::ssize_t dim = candidates.shape(1);
    const double* ap = acq.data();
    const double* cp = candidates.data();
    std::vector<py::ssize_t> order(static_cast<std::size_t>(n));
    for (py::ssize_t i = 0; i < n; ++i) {
        order[static_cast<std::size_t>(i)] = i;
    }
    std::sort(order.begin(), order.end(), [ap](py::ssize_t lhs, py::ssize_t rhs) {
        return ap[lhs] > ap[rhs];
    });
    std::vector<long long> selected;
    selected.reserve(static_cast<std::size_t>(batch_size));
    for (const py::ssize_t idx : order) {
        if (static_cast<int>(selected.size()) >= batch_size) {
            break;
        }
        if (use_distance && !selected.empty()) {
            double min_dist2 = std::numeric_limits<double>::infinity();
            for (const long long sel : selected) {
                double dist2 = 0.0;
                for (py::ssize_t d = 0; d < dim; ++d) {
                    const double diff = cp[idx * dim + d] - cp[static_cast<py::ssize_t>(sel) * dim + d];
                    dist2 += diff * diff;
                }
                min_dist2 = std::min(min_dist2, dist2);
            }
            if (std::sqrt(min_dist2) < min_distance) {
                continue;
            }
        }
        selected.push_back(static_cast<long long>(idx));
    }
    auto out = py::array_t<long long>({static_cast<py::ssize_t>(selected.size())});
    std::copy(selected.begin(), selected.end(), out.mutable_data());
    return out;
}

static inline double tet_volume_from_vertices(const double* v, const int ids[4], py::ssize_t dim) {
    const double ax = v[ids[1] * dim] - v[ids[0] * dim];
    const double ay = v[ids[1] * dim + 1] - v[ids[0] * dim + 1];
    const double az = v[ids[1] * dim + 2] - v[ids[0] * dim + 2];
    const double bx = v[ids[2] * dim] - v[ids[0] * dim];
    const double by = v[ids[2] * dim + 1] - v[ids[0] * dim + 1];
    const double bz = v[ids[2] * dim + 2] - v[ids[0] * dim + 2];
    const double cx = v[ids[3] * dim] - v[ids[0] * dim];
    const double cy = v[ids[3] * dim + 1] - v[ids[0] * dim + 1];
    const double cz = v[ids[3] * dim + 2] - v[ids[0] * dim + 2];
    const double cross_x = by * cz - bz * cy;
    const double cross_y = bz * cx - bx * cz;
    const double cross_z = bx * cy - by * cx;
    const double triple = ax * cross_x + ay * cross_y + az * cross_z;
    return std::abs(triple) / 6.0;
}

static double hex_volume_native(Array2D vertices) {
    if (vertices.shape(0) != 8 || vertices.shape(1) != 3) {
        throw std::invalid_argument("vertices must have shape (8, 3)");
    }
    static constexpr int tets[5][4] = {
        {0, 1, 3, 4},
        {1, 2, 3, 6},
        {1, 3, 4, 6},
        {3, 4, 6, 7},
        {1, 4, 5, 6},
    };
    double volume = 0.0;
    for (const auto& tet : tets) {
        volume += tet_volume_from_vertices(vertices.data(), tet, 3);
    }
    return volume;
}

static py::tuple trigger_average_accum(ArrayD signal, ArrayI valid_indices, int half_window, bool return_std) {
    if (signal.ndim() < 1) {
        throw std::invalid_argument("signal must have at least one dimension");
    }
    if (valid_indices.ndim() != 1) {
        throw std::invalid_argument("valid_indices must be 1D");
    }
    const py::ssize_t n_t = signal.shape(0);
    const py::ssize_t win_len = 2 * static_cast<py::ssize_t>(half_window) + 1;
    py::ssize_t frame_size = 1;
    std::vector<py::ssize_t> out_shape;
    out_shape.push_back(win_len);
    for (py::ssize_t axis = 1; axis < signal.ndim(); ++axis) {
        out_shape.push_back(signal.shape(axis));
        frame_size *= signal.shape(axis);
    }
    auto mean = py::array_t<double>(out_shape);
    double* mp = mean.mutable_data();
    const py::ssize_t out_size = win_len * frame_size;
    std::fill(mp, mp + out_size, 0.0);
    auto std_arr = py::array_t<double>(out_shape);
    double* sp = std_arr.mutable_data();
    if (return_std) {
        std::fill(sp, sp + out_size, 0.0);
    }
    const double* sig = signal.data();
    const long long* idxp = valid_indices.data();
    const py::ssize_t count = valid_indices.shape(0);
    for (py::ssize_t idx_i = 0; idx_i < count; ++idx_i) {
        const py::ssize_t center = static_cast<py::ssize_t>(idxp[idx_i]);
        if (center - half_window < 0 || center + half_window >= n_t) {
            continue;
        }
        for (py::ssize_t w = 0; w < win_len; ++w) {
            const py::ssize_t src_frame = center - half_window + w;
            for (py::ssize_t q = 0; q < frame_size; ++q) {
                const double value = sig[src_frame * frame_size + q];
                const py::ssize_t out_idx = w * frame_size + q;
                mp[out_idx] += value;
                if (return_std) {
                    sp[out_idx] += value * value;
                }
            }
        }
    }
    if (count == 0) {
        if (return_std) {
            return py::make_tuple(mean, std_arr, 0);
        }
        return py::make_tuple(mean, 0);
    }
    for (py::ssize_t i = 0; i < out_size; ++i) {
        mp[i] /= static_cast<double>(count);
    }
    if (return_std) {
        for (py::ssize_t i = 0; i < out_size; ++i) {
            const double var = sp[i] / static_cast<double>(count) - mp[i] * mp[i];
            sp[i] = std::sqrt(std::max(var, 0.0));
        }
        return py::make_tuple(mean, std_arr, static_cast<int>(count));
    }
    return py::make_tuple(mean, static_cast<int>(count));
}

static py::dict quadrant_split_native(ArrayD up, ArrayD vp, double hole) {
    if (up.ndim() != 1 || vp.ndim() != 1 || up.shape(0) != vp.shape(0)) {
        throw std::invalid_argument("up and vp must be matching 1D arrays");
    }
    if (hole < 0.0) {
        throw std::invalid_argument("hole must be non-negative");
    }
    const py::ssize_t n = up.shape(0);
    const double* upv = up.data();
    const double* vpv = vp.data();
    double sum_u2 = 0.0;
    double sum_v2 = 0.0;
    for (py::ssize_t i = 0; i < n; ++i) {
        sum_u2 += upv[i] * upv[i];
        sum_v2 += vpv[i] * vpv[i];
    }
    const double denom_n = static_cast<double>(std::max<py::ssize_t>(n, 1));
    const double u_rms = std::sqrt(sum_u2 / denom_n) + 1e-30;
    const double v_rms = std::sqrt(sum_v2 / denom_n) + 1e-30;
    const double threshold = hole * u_rms * v_rms;
    std::array<int, 5> counts = {0, 0, 0, 0, 0};
    std::array<double, 5> sums = {0.0, 0.0, 0.0, 0.0, 0.0};
    for (py::ssize_t i = 0; i < n; ++i) {
        const double uv = upv[i] * vpv[i];
        const bool in_hole = std::abs(uv) < threshold;
        int bucket = in_hole ? 4 : -1;
        if (!in_hole) {
            if (upv[i] > 0.0 && vpv[i] > 0.0) {
                bucket = 0;
            } else if (upv[i] < 0.0 && vpv[i] > 0.0) {
                bucket = 1;
            } else if (upv[i] < 0.0 && vpv[i] < 0.0) {
                bucket = 2;
            } else if (upv[i] > 0.0 && vpv[i] < 0.0) {
                bucket = 3;
            }
        }
        if (bucket >= 0) {
            counts[static_cast<std::size_t>(bucket)] += 1;
            sums[static_cast<std::size_t>(bucket)] += uv;
        }
    }
    const char* names[5] = {"Q1", "Q2", "Q3", "Q4", "hole"};
    py::dict out;
    for (int b = 0; b < 5; ++b) {
        py::dict item;
        item["count"] = counts[static_cast<std::size_t>(b)];
        item["fraction"] = static_cast<double>(counts[static_cast<std::size_t>(b)]) / denom_n;
        item["mean_uv"] = counts[static_cast<std::size_t>(b)] > 0 ? sums[static_cast<std::size_t>(b)] / counts[static_cast<std::size_t>(b)] : 0.0;
        item["contribution"] = sums[static_cast<std::size_t>(b)] / denom_n;
        out[names[b]] = item;
    }
    return out;
}

static double kolmogorov_pvalue(double d, double n) {
    if (d <= 0.0) {
        return 1.0;
    }
    const double s = std::sqrt(n);
    const double en = s + 0.12 + 0.11 / s;
    const double lam = en * d;
    if (lam < 0.18) {
        return 1.0;
    }
    double sum_p = 0.0;
    double sign = 1.0;
    for (int j = 1; j <= 100; ++j) {
        const double term = sign * std::exp(-2.0 * static_cast<double>(j * j) * lam * lam);
        sum_p += term;
        if (std::abs(term) < 1e-9 * std::abs(sum_p)) {
            break;
        }
        sign = -sign;
    }
    return std::min(1.0, std::max(0.0, 2.0 * sum_p));
}

static int number_peaks_native(ArrayD x, int support) {
    if (x.ndim() != 1) {
        throw std::invalid_argument("x must be a 1D array");
    }
    if (support <= 0) {
        return static_cast<int>(x.shape(0) - 2 * static_cast<py::ssize_t>(support));
    }
    const py::ssize_t n = x.shape(0);
    if (n < 2 * static_cast<py::ssize_t>(support) + 1) {
        return 0;
    }
    const double* xp = x.data();
    int count = 0;
    for (py::ssize_t i = support; i < n - support; ++i) {
        bool is_peak = true;
        for (int k = 1; k <= support; ++k) {
            if (xp[i] <= xp[i - k] || xp[i] <= xp[i + k]) {
                is_peak = false;
                break;
            }
        }
        if (is_peak) {
            count += 1;
        }
    }
    return count;
}

static double jackknife_mean_var_native(ArrayD data) {
    if (data.ndim() != 1) {
        throw std::invalid_argument("data must be a 1D array");
    }
    const py::ssize_t n = data.shape(0);
    if (n == 0) {
        throw std::invalid_argument("data must not be empty");
    }
    if (n == 1) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double* xp = data.data();
    double total = 0.0;
    for (py::ssize_t i = 0; i < n; ++i) {
        total += xp[i];
    }
    std::vector<double> theta(static_cast<std::size_t>(n));
    double theta_dot = 0.0;
    const double denom = static_cast<double>(n - 1);
    for (py::ssize_t i = 0; i < n; ++i) {
        const double value = (total - xp[i]) / denom;
        theta[static_cast<std::size_t>(i)] = value;
        theta_dot += value;
    }
    theta_dot /= static_cast<double>(n);
    double ss = 0.0;
    for (double value : theta) {
        const double diff = value - theta_dot;
        ss += diff * diff;
    }
    return static_cast<double>(n - 1) / static_cast<double>(n) * ss;
}

static py::array_t<double> probe_time_series_native(ArrayD snapshots, ArrayD coords, ArrayD probes, const std::string& method, int k) {
    if (snapshots.ndim() != 2 || coords.ndim() != 2 || probes.ndim() != 2) {
        throw std::invalid_argument("snapshots, coords, and probes must be 2D arrays");
    }
    const py::ssize_t n_points = snapshots.shape(0);
    const py::ssize_t n_times = snapshots.shape(1);
    if (coords.shape(0) != n_points || probes.shape(1) != coords.shape(1)) {
        throw std::invalid_argument("shape mismatch among snapshots, coords, and probes");
    }
    if (method != "nearest" && method != "idw") {
        throw std::invalid_argument("unknown probe interpolation method");
    }
    const py::ssize_t dim = coords.shape(1);
    const py::ssize_t n_probes = probes.shape(0);
    auto out = py::array_t<double>({n_probes, n_times});
    double* op = out.mutable_data();
    const double* sp = snapshots.data();
    const double* cp = coords.data();
    const double* pp = probes.data();
    const int kk = std::max(1, std::min(k, static_cast<int>(n_points)));
    std::vector<std::pair<double, py::ssize_t>> dist_idx(static_cast<std::size_t>(n_points));
    for (py::ssize_t p = 0; p < n_probes; ++p) {
        for (py::ssize_t i = 0; i < n_points; ++i) {
            double d2 = 0.0;
            for (py::ssize_t d = 0; d < dim; ++d) {
                const double diff = pp[p * dim + d] - cp[i * dim + d];
                d2 += diff * diff;
            }
            dist_idx[static_cast<std::size_t>(i)] = {d2, i};
        }
        std::partial_sort(
            dist_idx.begin(), dist_idx.begin() + kk, dist_idx.end(),
            [](const auto& a, const auto& b) {
                if (a.first == b.first) {
                    return a.second < b.second;
                }
                return a.first < b.first;
            }
        );
        for (py::ssize_t t = 0; t < n_times; ++t) {
            double value = 0.0;
            if (method == "nearest") {
                value = sp[dist_idx[0].second * n_times + t];
            } else {
                double weight_sum = 0.0;
                double weighted = 0.0;
                for (int j = 0; j < kk; ++j) {
                    const py::ssize_t src_i = dist_idx[static_cast<std::size_t>(j)].second;
                    const double weight = 1.0 / (dist_idx[static_cast<std::size_t>(j)].first + 1e-12);
                    weight_sum += weight;
                    weighted += weight * sp[src_i * n_times + t];
                }
                value = weighted / weight_sum;
            }
            op[p * n_times + t] = value;
        }
    }
    return out;
}

static double bingham_stress_native(double gamma_dot, double tau_y, double mu_p) {
    if (gamma_dot == 0.0) {
        return 0.0;
    }
    const double sign = gamma_dot > 0.0 ? 1.0 : -1.0;
    return sign * (tau_y + mu_p * std::abs(gamma_dot));
}

static double bingham_apparent_viscosity_native(double gamma_dot, double tau_y, double mu_p, double eps) {
    const double g = std::max(std::abs(gamma_dot), eps);
    return tau_y / g + mu_p;
}

static std::array<double, 3> barycentric_values_2d(const double* tri, const double* p) {
    const double v0x = tri[2] - tri[0];
    const double v0y = tri[3] - tri[1];
    const double v1x = tri[4] - tri[0];
    const double v1y = tri[5] - tri[1];
    const double v2x = p[0] - tri[0];
    const double v2y = p[1] - tri[1];
    const double d00 = v0x * v0x + v0y * v0y;
    const double d01 = v0x * v1x + v0y * v1y;
    const double d11 = v1x * v1x + v1y * v1y;
    const double d20 = v2x * v0x + v2y * v0y;
    const double d21 = v2x * v1x + v2y * v1y;
    const double denom = d00 * d11 - d01 * d01;
    if (std::abs(denom) < 1e-20) {
        throw std::invalid_argument("degenerate triangle");
    }
    const double v = (d11 * d20 - d01 * d21) / denom;
    const double w = (d00 * d21 - d01 * d20) / denom;
    return {1.0 - v - w, v, w};
}

static py::array_t<double> barycentric_2d_native(ArrayD triangle, ArrayD p) {
    if (triangle.ndim() != 2 || triangle.shape(0) != 3 || triangle.shape(1) != 2 || p.ndim() != 1 || p.shape(0) != 2) {
        throw std::invalid_argument("triangle must be (3, 2) and p must be (2,)");
    }
    const auto bc = barycentric_values_2d(triangle.data(), p.data());
    auto out = py::array_t<double>({static_cast<py::ssize_t>(3)});
    double* op = out.mutable_data();
    op[0] = bc[0];
    op[1] = bc[1];
    op[2] = bc[2];
    return out;
}

static int locate_triangle_native(ArrayD points, ArrayI simplices, ArrayD p) {
    if (points.ndim() != 2 || points.shape(1) != 2 || simplices.ndim() != 2 || simplices.shape(1) != 3 || p.ndim() != 1 || p.shape(0) != 2) {
        throw std::invalid_argument("points must be (N, 2), simplices must be (M, 3), and p must be (2,)");
    }
    const double* pts = points.data();
    const long long* sim = simplices.data();
    const py::ssize_t n_tri = simplices.shape(0);
    for (py::ssize_t i = 0; i < n_tri; ++i) {
        double tri[6];
        bool valid = true;
        for (py::ssize_t j = 0; j < 3; ++j) {
            const long long idx = sim[i * 3 + j];
            if (idx < 0 || idx >= points.shape(0)) {
                valid = false;
                break;
            }
            tri[2 * j] = pts[idx * 2];
            tri[2 * j + 1] = pts[idx * 2 + 1];
        }
        if (!valid) {
            continue;
        }
        try {
            const auto bc = barycentric_values_2d(tri, p.data());
            if (bc[0] >= -1e-10 && bc[1] >= -1e-10 && bc[2] >= -1e-10) {
                return static_cast<int>(i);
            }
        } catch (const std::invalid_argument&) {
            continue;
        }
    }
    return -1;
}

static py::array_t<double> friction_velocity_native(ArrayD wall_shear_stress, double rho) {
    if (rho <= 0.0) {
        throw std::invalid_argument("rho must be positive");
    }
    if (wall_shear_stress.ndim() == 1) {
        double tau2 = 0.0;
        const double* wp = wall_shear_stress.data();
        for (py::ssize_t j = 0; j < wall_shear_stress.shape(0); ++j) {
            tau2 += wp[j] * wp[j];
        }
        auto out = py::array_t<double>({static_cast<py::ssize_t>(1)});
        out.mutable_data()[0] = std::sqrt(std::sqrt(tau2) / rho);
        return out;
    }
    if (wall_shear_stress.ndim() != 2) {
        throw std::invalid_argument("wall_shear_stress must be a 1D vector or 2D array");
    }
    const py::ssize_t n = wall_shear_stress.shape(0);
    const py::ssize_t dim = wall_shear_stress.shape(1);
    auto out = py::array_t<double>({n});
    const double* wp = wall_shear_stress.data();
    double* op = out.mutable_data();
    for (py::ssize_t i = 0; i < n; ++i) {
        double tau2 = 0.0;
        for (py::ssize_t j = 0; j < dim; ++j) {
            const double value = wp[i * dim + j];
            tau2 += value * value;
        }
        op[i] = std::sqrt(std::sqrt(tau2) / rho);
    }
    return out;
}

static double estimate_first_cell_height_native(double y_plus_target, double Re, double rho, double U_inf, double nu) {
    const double Cf = 0.026 * std::pow(Re, -1.0 / 7.0);
    const double tau_w = Cf * 0.5 * rho * U_inf * U_inf;
    const double u_tau = std::sqrt(tau_w / rho);
    if (u_tau < 1e-16) {
        throw std::invalid_argument("friction velocity is too small");
    }
    return y_plus_target * nu / u_tau;
}

static py::tuple cht_iterate_native(ArrayD T_solid, ArrayD T_fluid, double k_s, double k_f, int n_iter) {
    if (T_solid.ndim() != 1 || T_fluid.ndim() != 1) {
        throw std::invalid_argument("T_solid and T_fluid must be 1D arrays");
    }
    if (T_solid.shape(0) < 2 || T_fluid.shape(0) < 2) {
        throw std::invalid_argument("T_solid and T_fluid must have at least two nodes");
    }
    if (k_s + k_f == 0.0) {
        throw std::invalid_argument("k_s + k_f must be non-zero");
    }
    const py::ssize_t ns = T_solid.shape(0);
    const py::ssize_t nf = T_fluid.shape(0);
    auto Ts = py::array_t<double>({ns});
    auto Tf = py::array_t<double>({nf});
    std::copy(T_solid.data(), T_solid.data() + ns, Ts.mutable_data());
    std::copy(T_fluid.data(), T_fluid.data() + nf, Tf.mutable_data());
    double* tsp = Ts.mutable_data();
    double* tfp = Tf.mutable_data();
    for (int it = 0; it < n_iter; ++it) {
        for (py::ssize_t i = 1; i < ns - 1; ++i) {
            tsp[i] = 0.5 * (tsp[i + 1] + tsp[i - 1]);
        }
        for (py::ssize_t i = 1; i < nf - 1; ++i) {
            tfp[i] = 0.5 * (tfp[i + 1] + tfp[i - 1]);
        }
        const double T_iface = (k_s * tsp[ns - 2] + k_f * tfp[1]) / (k_s + k_f);
        tsp[ns - 1] = T_iface;
        tfp[0] = T_iface;
    }
    return py::make_tuple(Ts, Tf);
}

static double battery_temperature_step_native(double T, double T_amb, double Q_gen, double h, double A, double m, double cp, double dt) {
    const double dTdt = (Q_gen - h * A * (T - T_amb)) / (m * cp);
    return T + dt * dTdt;
}

static double battery_steady_temperature_native(double T_amb, double Q_gen, double h, double A) {
    return T_amb + Q_gen / (h * A);
}

static double rayleigh_quotient_native(ArrayD a, ArrayD x0) {
    check_square_matrix(a);
    const py::ssize_t n = a.shape(0);
    std::vector<double> x = contiguous_vector(x0, n, "x");
    return dot(x, matvec(a.data(), n, x)) / (dot(x, x) + 1e-30);
}

PYBIND11_MODULE(_kernels, m) {
    m.doc() = "C++ kernels for NavierTwin numeric hot paths";
    m.def("field_j_2d", &field_j_2d, py::arg("u"), py::arg("v"), py::arg("dx") = 1.0, py::arg("dy") = 1.0);
    m.def("lambda2_2d", &lambda2_2d, py::arg("u"), py::arg("v"), py::arg("dx") = 1.0, py::arg("dy") = 1.0);
    m.def("vorticity_2d", &vorticity_2d_native, py::arg("u"), py::arg("v"), py::arg("dx") = 1.0, py::arg("dy") = 1.0);
    m.def("q_criterion_2d", &q_criterion_2d_native, py::arg("u"), py::arg("v"), py::arg("dx") = 1.0, py::arg("dy") = 1.0);
    m.def("production_rate_2d", &production_rate_2d, py::arg("u"), py::arg("v"), py::arg("dx"), py::arg("dy"), py::arg("nu_t"));
    m.def(
        "entropy_generation_2d", &entropy_generation_2d_native, py::arg("u"), py::arg("v"), py::arg("T"),
        py::arg("dx") = 1.0, py::arg("dy") = 1.0, py::arg("mu") = 1.8e-5, py::arg("k") = 0.026
    );
    m.def(
        "vorticity_3d", &vorticity_3d_native, py::arg("u"), py::arg("v"), py::arg("w"),
        py::arg("dx") = 1.0, py::arg("dy") = 1.0, py::arg("dz") = 1.0
    );
    m.def("q_criterion_from_grad_3d", &q_criterion_from_grad_3d, py::arg("gradient"));
    m.def("lambda2_from_grad_3d", &lambda2_from_grad_3d, py::arg("gradient"));
    m.def("decompose_j_3x3", &decompose_j_3x3, py::arg("J"));
    m.def("symmetric_eigenvalues_3x3", &symmetric_eigenvalues_3x3, py::arg("J"));
    m.def("invariants_3x3", &invariants_3x3, py::arg("J"));
    m.def("solve_square", &solve_square_native, py::arg("A"), py::arg("b"));
    m.def("determinant_batch", &determinant_batch, py::arg("A"));
    m.def("power_iteration", &power_iteration_native, py::arg("A"), py::arg("n_iter"), py::arg("x0"), py::arg("tol"));
    m.def("inverse_power", &inverse_power_native, py::arg("A"), py::arg("shift"), py::arg("n_iter"), py::arg("x0"));
    m.def("pcg_jacobi", &pcg_jacobi_native, py::arg("A"), py::arg("b"), py::arg("x0"), py::arg("max_iter"), py::arg("tol"));
    m.def("bicgstab_dense", &bicgstab_dense_native, py::arg("A"), py::arg("b"), py::arg("x0"), py::arg("max_iter"), py::arg("tol"));
    m.def("thomas_solve", &thomas_solve_native, py::arg("a"), py::arg("b"), py::arg("c"), py::arg("d"));
    m.def("jacobi_dense", &jacobi_dense_native, py::arg("A"), py::arg("b"), py::arg("x0"), py::arg("max_iter"), py::arg("tol"));
    m.def("gauss_seidel_dense", &gauss_seidel_dense_native, py::arg("A"), py::arg("b"), py::arg("x0"), py::arg("max_iter"), py::arg("tol"));
    m.def("conjugate_gradient_dense", &conjugate_gradient_dense_native, py::arg("A"), py::arg("b"), py::arg("x0"), py::arg("max_iter"), py::arg("tol"));
    m.def("arnoldi", &arnoldi_native, py::arg("A"), py::arg("b"), py::arg("k"));
    m.def("dtw_distance", &dtw_distance_native, py::arg("a"), py::arg("b"), py::arg("window") = py::none());
    m.def("dtw_matrix", &dtw_matrix_native, py::arg("a"), py::arg("b"));
    m.def("dbscan", &dbscan_native, py::arg("points"), py::arg("eps") = 0.5, py::arg("min_samples") = 5);
    m.def("cusum_detect", &cusum_detect_native, py::arg("x"), py::arg("threshold"), py::arg("mean"), py::arg("sigma"), py::arg("k"));
    m.def("rom_energy_spectrum", &rom_energy_spectrum_native, py::arg("singular_values"));
    m.def("radial_energy_sum", &radial_energy_sum, py::arg("K"), py::arg("energy"), py::arg("edges"));
    m.def("dominant_frequencies_from_power", &dominant_frequencies_from_power, py::arg("freqs"), py::arg("power"), py::arg("top_k"));
    m.def("deposit_cic_1d", &deposit_cic_1d, py::arg("x"), py::arg("weights"), py::arg("n_grid"), py::arg("dx") = 1.0, py::arg("x0") = 0.0);
    m.def("sph_density_1d", &sph_density_1d, py::arg("positions"), py::arg("masses"), py::arg("h") = 1.0);
    m.def("reaction_rate", &reaction_rate_native, py::arg("k"), py::arg("concentrations"), py::arg("orders"));
    m.def("vof_step_1d", &vof_step_1d, py::arg("alpha"), py::arg("u"), py::arg("dt"), py::arg("dx"));
    m.def("levelset_advect_step_1d", &levelset_advect_step_1d, py::arg("phi"), py::arg("u"), py::arg("dt"), py::arg("dx"));
    m.def("jensen_farm_velocity", &jensen_farm_velocity, py::arg("V0"), py::arg("distances"), py::arg("R"), py::arg("a") = 0.3, py::arg("k") = 0.04);
    m.def("conservative_remap_1d", &conservative_remap_1d, py::arg("x_old_edges"), py::arg("u_old"), py::arg("x_new_edges"));
    m.def("fd_heat_1d_evolve", &fd_heat_1d_evolve, py::arg("u0"), py::arg("n_steps"), py::arg("coef"), py::arg("dt"));
    m.def("fd_burgers_1d_evolve", &fd_burgers_1d_evolve, py::arg("u0"), py::arg("n_steps"), py::arg("dt"), py::arg("dx"), py::arg("nu"));
    m.def("poisson_2d_jacobi", &poisson_2d_jacobi_native, py::arg("f"), py::arg("dx") = 1.0, py::arg("dy") = 1.0, py::arg("max_iter") = 5000, py::arg("tol") = 1e-6);
    m.def("levelset_reinit_1d", &levelset_reinit_1d, py::arg("phi"), py::arg("dx") = 1.0, py::arg("n_iter") = 30);
    m.def("fast_march_1d", &fast_march_1d, py::arg("phi"), py::arg("dx") = 1.0);
    m.def("mesh_refine_by_gradient", &mesh_refine_by_gradient, py::arg("x"), py::arg("f"), py::arg("threshold") = 0.1, py::arg("max_passes") = 5);
    m.def("mesh_coarsen_by_tolerance", &mesh_coarsen_by_tolerance, py::arg("x"), py::arg("f"), py::arg("tol") = 1e-3);
    m.def("laplacian_smooth", &laplacian_smooth_native, py::arg("verts"), py::arg("edges"), py::arg("n_iter"), py::arg("alpha"), py::arg("fixed"));
    m.def("mesh_quality_report", &mesh_quality_report_native, py::arg("points"), py::arg("simplices"));
    m.def("order_table", &order_table_native, py::arg("h"), py::arg("err"));
    m.def("sph_acceleration_1d", &sph_acceleration_1d, py::arg("x"), py::arg("m"), py::arg("rho"), py::arg("p"), py::arg("h") = 1.0);
    m.def("lbm_equilibrium", &lbm_equilibrium, py::arg("rho"), py::arg("u"));
    m.def("lbm_step", &lbm_step, py::arg("f"), py::arg("omega") = 1.0);
    m.def("is_monotone_decreasing", &is_monotone_decreasing_native, py::arg("errs"), py::arg("atol") = 0.0);
    m.def("convergence_ratio", &convergence_ratio_native, py::arg("errs"));
    m.def("combined_disc_uncertainty", &combined_disc_uncertainty_native, py::arg("uncs"));
    m.def("friction_colebrook", &friction_colebrook_native, py::arg("Re"), py::arg("eps_over_D"), py::arg("n_iter") = 50);
    m.def("delta99_scan", &delta99_scan, py::arg("y"), py::arg("u"), py::arg("target"));
    m.def("scale_to_bounds", &scale_to_bounds_native, py::arg("unit"), py::arg("bounds"));
    m.def("regular_grid_points", &regular_grid_points, py::arg("bounds"), py::arg("per"));
    m.def("norm_cdf", &norm_cdf, py::arg("x"));
    m.def(
        "greedy_batch_acquisition", &greedy_batch_acquisition_native, py::arg("acq"), py::arg("candidates"),
        py::arg("batch_size"), py::arg("min_distance"), py::arg("use_distance")
    );
    m.def("hex_volume", &hex_volume_native, py::arg("vertices"));
    m.def(
        "trigger_average_accum", &trigger_average_accum, py::arg("signal"), py::arg("valid_indices"),
        py::arg("half_window"), py::arg("return_std")
    );
    m.def("quadrant_split", &quadrant_split_native, py::arg("up"), py::arg("vp"), py::arg("hole") = 0.0);
    m.def("kolmogorov_pvalue", &kolmogorov_pvalue, py::arg("D"), py::arg("n"));
    m.def("number_peaks", &number_peaks_native, py::arg("x"), py::arg("support") = 3);
    m.def("jackknife_mean_var", &jackknife_mean_var_native, py::arg("data"));
    m.def(
        "probe_time_series", &probe_time_series_native, py::arg("snapshots"), py::arg("coords"),
        py::arg("probes"), py::arg("method") = "nearest", py::arg("k") = 4
    );
    m.def("bingham_stress", &bingham_stress_native, py::arg("gamma_dot"), py::arg("tau_y"), py::arg("mu_p"));
    m.def(
        "bingham_apparent_viscosity", &bingham_apparent_viscosity_native, py::arg("gamma_dot"),
        py::arg("tau_y"), py::arg("mu_p"), py::arg("eps") = 1e-6
    );
    m.def("barycentric_2d", &barycentric_2d_native, py::arg("triangle"), py::arg("p"));
    m.def("locate_triangle", &locate_triangle_native, py::arg("points"), py::arg("simplices"), py::arg("p"));
    m.def("friction_velocity", &friction_velocity_native, py::arg("wall_shear_stress"), py::arg("rho"));
    m.def(
        "estimate_first_cell_height", &estimate_first_cell_height_native, py::arg("y_plus_target"),
        py::arg("Re"), py::arg("rho"), py::arg("U_inf"), py::arg("nu")
    );
    m.def("cht_iterate", &cht_iterate_native, py::arg("T_solid"), py::arg("T_fluid"), py::arg("k_s"), py::arg("k_f"), py::arg("n_iter"));
    m.def(
        "battery_temperature_step", &battery_temperature_step_native, py::arg("T"), py::arg("T_amb"),
        py::arg("Q_gen"), py::arg("h"), py::arg("A"), py::arg("m"), py::arg("cp"), py::arg("dt")
    );
    m.def(
        "battery_steady_temperature", &battery_steady_temperature_native, py::arg("T_amb"),
        py::arg("Q_gen"), py::arg("h"), py::arg("A")
    );
    m.def("rayleigh_quotient", &rayleigh_quotient_native, py::arg("A"), py::arg("x"));
}
