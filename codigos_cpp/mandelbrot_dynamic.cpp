// mandelbrot_dynamic.cpp
// Compilar con: g++ -O3 -fopenmp -shared -std=c++17 -o mandelbrot.dll mandelbrot_dynamic.cpp

#include <complex>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <cctype>
#include <regex>
#include <omp.h>

struct Term {
    double coeff;
    int exp_z;
    int exp_c;
};

// Elimina espacios de s
static void remove_spaces(std::string& s) {
    s.erase(std::remove_if(s.begin(), s.end(), ::isspace), s.end());
}

// Divide s en subcadenas separadas por ‘delim’
static std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> out;
    size_t start = 0, pos;
    while ((pos = s.find(delim, start)) != std::string::npos) {
        if (pos > start)
            out.emplace_back(s.substr(start, pos - start));
        start = pos + 1;
    }
    if (start < s.size())
        out.emplace_back(s.substr(start));
    return out;
}

// Parsea expresiones tipo z=z**3+c**2+2*c**3
static std::vector<Term> parse_formula(const std::string& f_in) {
    std::string f = f_in;
    remove_spaces(f);
    auto eq = f.find('=');
    std::string expr = (eq != std::string::npos ? f.substr(eq + 1) : f);

    // Normaliza signos: convierte '-' en '+-' salvo el primero
    if (!expr.empty() && expr[0] == '-') expr = "0" + expr;
    for (size_t i = 1; i < expr.size(); ++i) {
        if (expr[i] == '-') {
            expr[i] = '+';
            expr.insert(i + 1, "-");
        }
    }

    auto tokens = split(expr, '+');
    std::vector<Term> terms;
    std::regex num_re(R"(([+-]?[0-9]*\.?[0-9]+))");
    std::regex z_re(R"(z\*\*(\d+))");
    std::regex c_re(R"(c\*\*(\d+))");
    std::smatch m;

    for (auto& tok : tokens) {
        if (tok.empty()) continue;
        Term t{1.0, 0, 0};
        // coeficiente numérico
        if (std::regex_search(tok, m, num_re)) {
            t.coeff = std::stod(m[1]);
        }
        // exponente de z
        if (std::regex_search(tok, m, z_re)) {
            t.exp_z = std::stoi(m[1]);
        } else if (tok.find('z') != std::string::npos) {
            t.exp_z = 1;
        }
        // exponente de c
        if (std::regex_search(tok, m, c_re)) {
            t.exp_c = std::stoi(m[1]);
        } else if (tok.find('c') != std::string::npos) {
            t.exp_c = 1;
        }
        terms.push_back(t);
    }
    return terms;
}

extern "C" {

__declspec(dllexport)
int* mandelbrot(double xmin, double xmax,
                double ymin, double ymax,
                int width, int height,
                int max_iter,
                const char* formula_cstr)
{
    // Parseamos la fórmula una vez
    std::vector<Term> terms = parse_formula(std::string(formula_cstr));

    int* M = new int[width * height];
    if (!M) return nullptr;

    double dx = (xmax - xmin) / (width  - 1);
    double dy = (ymax - ymin) / (height - 1);

    #pragma omp parallel for collapse(2)
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            std::complex<double> z{0.0, 0.0};
            std::complex<double> c{xmin + i * dx, ymin + j * dy};
            int iter = 0;

            while (iter < max_iter && std::norm(z) <= 4.0) {
                std::complex<double> next{0.0, 0.0};
                for (auto& t : terms) {
                    next += t.coeff
                          * std::pow(z, t.exp_z)
                          * std::pow(c, t.exp_c);
                }
                z = next;
                ++iter;
            }
            M[j * width + i] = iter - 1;
        }
    }

    return M;
}

__declspec(dllexport)
void free_mandelbrot(int* M) {
    delete[] M;
}

} // extern "C"
