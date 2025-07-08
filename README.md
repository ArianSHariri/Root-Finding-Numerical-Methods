# Root-Finding Numerical Methods

This repository contains a Python implementation of numerical root-finding methods for solving \( f(x) = 0 \). The methods include Bisection, Newton-Raphson, Fixed Point Iteration, False Position, Secant, and Aitken’s Delta-Squared Acceleration. The default function is \( f(x) = x^3 - \cos(x) \) with parameters \( x_0 = 0.9 \), \( a = 0 \), \( b = 1 \), \( \epsilon = 10^{-6} \), and max iterations = 10.

## Project Overview
The goal is to compare the performance of root-finding methods in terms of convergence speed, accuracy, and computational time. The code generates:
- Iteration tables for each method and their Aitken-accelerated versions, saved in `tables/`.
- Graphs visualizing convergence, trajectories, iteration counts, error ratios, and Aitken’s improvements, saved in `graphs/`.

## Requirements
- Python 3.x
- Libraries: `numpy`, `matplotlib`, `sympy`, `tabulate`, `scipy`
Install dependencies:
```bash
pip install -r requirements.txt
```

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Root-Finding-Numerical-Methods.git
   cd Root-Finding-Numerical-Methods
   ```
2. Run the script with the default function (\( f(x) = x^3 - \cos(x) \)):
   ```bash
   python algo.py
   ```
3. Run with a different function (e.g., \( f(x) = x^4 - 2x^3 + x - 1 \)):
   ```bash
   python algo.py --function x4_minus_2x3_plus_x_minus_1
   ```
4. Override default parameters (e.g., custom \( x_0 \), tolerance):
   ```bash
   python algo.py --function x3_minus_cosx --x0 0.8 --e 1e-5
   ```

### Available Functions
Run `python algo.py --help` to see all options. Available functions:
- `x3_minus_cosx`: \( f(x) = x^3 - \cos(x) \), \( g(x) = \cos(x)^{1/3} \), default params: \( x_0 = 0.9 \), \( [a, b] = [0, 1] \)
- `x2_minus_4`: \( f(x) = x^2 - 4 \), \( g(x) = (x^2 + 4)/(2x) \), default params: \( x_0 = 1.5 \), \( [a, b] = [1, 2] \)
- `x2_minus_3`: \( f(x) = x^2 - 3 \), \( g(x) = x - (1/4)(x^2 - 3) \), default params: \( x_0 = 1.5 \), \( [a, b] = [1, 2] \)
- `x3_minus_x_minus_2`: \( f(x) = x^3 - x - 2 \), \( g(x) = (x + 2)^{1/3} \), default params: \( x_0 = 1.5 \), \( [a, b] = [1, 2] \)
- `x4_minus_2x3_plus_x_minus_1`: \( f(x) = x^4 - 2x^3 + x - 1 \), \( g(x) = (2x^3 - x + 1)^{1/4} \), default params: \( x_0 = 1.5 \), \( [a, b] = [1, 2] \)
- `exp_x_minus_3x`: \( f(x) = e^x - 3x \), \( g(x) = \ln(3x) \), default params: \( x_0 = 0.6 \), \( [a, b] = [0, 1] \)
- `sin_x_minus_x_over_2`: \( f(x) = \sin(x) - x/2 \), \( g(x) = 2\sin(x) \), default params: \( x_0 = 0.6 \), \( [a, b] = [0, 1] \)
- `exp_x_minus_x_minus_2`: \( f(x) = e^x - x - 2 \), \( g(x) = \ln(x + 2) \), default params: \( x_0 = 0.6 \), \( [a, b] = [0, 1] \)
- `sin_x_minus_exp_minus_x`: \( f(x) = \sin(x) - e^{-x} \), \( g(x) = \arcsin(e^{-x}) \), default params: \( x_0 = 0.6 \), \( [a, b] = [0, 1] \)

## Outputs
- **Tables**: Saved in `tables/` (e.g., `bisection_table.txt`, `newton_aitken_table.txt`).
- **Graphs**: Saved in `graphs/`:
  - `convergence_comparison.png`: Error vs. iteration.
  - `trajectory_comparison.png`: Path on \( f(x) \).
  - `iterations_comparison.png`: Bar chart of iterations.
  - `error_ratio_plot.png`: Convergence rate analysis.
  - `aitken_improvement.png`: Aitken’s error reduction.

### Sample Output: \( f(x) = x^3 - \cos(x) \)
**Bisection Table** (`tables/bisection_table.txt`):
```
Iteration         a         b         c      f(c)  Error |x_{n+1} - x_n|
        1  0.000000  1.000000  0.500000 -0.375000       0.250000
        2  0.500000  1.000000  0.750000  0.047852       0.125000
        3  0.500000  0.750000  0.625000 -0.143188       0.062500
...
```
**Bisection with Aitken** (`tables/bisection_aitken_table.txt`):
```
Iteration    x_hat            f(x_hat)         Error
        1  0.666667 -0.0941234567890123456  0.086233
        2  0.714286  0.0123456789012345678  0.038613
...
```

### Sample Output: \( f(x) = x^4 - 2x^3 + x - 1 \)
Run: `python algo.py --function x4_minus_2x3_plus_x_minus_1`
**Bisection Table**:
```
Iteration         a         b         c      f(c)  Error |x_{n+1} - x_n|
        1  1.000000  2.000000  1.500000 -0.437500       0.250000
        2  1.500000  2.000000  1.750000  0.316406       0.125000
        3  1.500000  1.750000  1.625000 -0.158203       0.062500
...
```
**Comparison Table**:
```
+---------------+--------------+------------------------+--------------------------+-------------------------+------------+
|    Method     | Iterations   |         Root           | Abs Error |x - x_true| |   Stopping Error        |  Time (s)  |
+---------------+--------------+------------------------+--------------------------+-------------------------+------------+
|   Bisection   |      10      | 1.7548828125000000000  | 0.0000000000000000000    | 0.0000004882812500000   |   0.001234 |
| Newton-Raphson|       4      | 1.7548776655443295730  | 0.0000000000000000000    | 0.0000000000000000000   |   0.000567 |
...
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Developed for a numerical analysis course.
- Uses Python libraries: `numpy`, `matplotlib`, `sympy`, `tabulate`, `scipy`.