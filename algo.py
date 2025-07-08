import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
import sympy as sp
import time
from scipy.optimize import fsolve
import argparse

# Define available functions
FUNCTIONS = {
    'x3_minus_cosx': {
        'f': lambda x: x**3 - np.cos(x),
        'g': lambda x: np.cos(x)**(1/3),
        'sympy': lambda x: x**3 - sp.cos(x),
        'params': {'x0': 0.9, 'a': 0, 'b': 1},
        'description': 'f(x) = x^3 - cos(x)'
    },
    'x2_minus_4': {
        'f': lambda x: x**2 - 4,
        'g': lambda x: (x**2 + 4) / (2 * x),
        'sympy': lambda x: x**2 - 4,
        'params': {'x0': 1.5, 'a': 1, 'b': 2},
        'description': 'f(x) = x^2 - 4'
    },
    'x2_minus_3': {
        'f': lambda x: x**2 - 3,
        'g': lambda x: x - (1/4)*(x**2 - 3),
        'sympy': lambda x: x**2 - 3,
        'params': {'x0': 1.5, 'a': 1, 'b': 2},
        'description': 'f(x) = x^2 - 3'
    },
    'x3_minus_x_minus_2': {
        'f': lambda x: x**3 - x - 2,
        'g': lambda x: (x + 2)**(1/3),
        'sympy': lambda x: x**3 - x - 2,
        'params': {'x0': 1.5, 'a': 1, 'b': 2},
        'description': 'f(x) = x^3 - x - 2'
    },
    'x4_minus_2x3_plus_x_minus_1': {
        'f': lambda x: x**4 - 2*x**3 + x - 1,
        'g': lambda x: (2*x**3 - x + 1)**(1/4),
        'sympy': lambda x: x**4 - 2*x**3 + x - 1,
        'params': {'x0': 1.5, 'a': 1, 'b': 2},
        'description': 'f(x) = x^4 - 2x^3 + x - 1'
    },
    'exp_x_minus_3x': {
        'f': lambda x: np.exp(x) - 3 * x,
        'g': lambda x: np.log(3 * x),
        'sympy': lambda x: sp.exp(x) - 3 * x,
        'params': {'x0': 0.6, 'a': 0, 'b': 1},
        'description': 'f(x) = e^x - 3x'
    },
    'sin_x_minus_x_over_2': {
        'f': lambda x: np.sin(x) - x/2,
        'g': lambda x: 2 * np.sin(x),
        'sympy': lambda x: sp.sin(x) - x/2,
        'params': {'x0': 0.6, 'a': 0, 'b': 1},
        'description': 'f(x) = sin(x) - x/2'
    },
    'exp_x_minus_x_minus_2': {
        'f': lambda x: np.exp(x) - x - 2,
        'g': lambda x: np.log(x + 2),
        'sympy': lambda x: sp.exp(x) - x - 2,
        'params': {'x0': 0.6, 'a': 0, 'b': 1},
        'description': 'f(x) = e^x - x - 2'
    },
    'sin_x_minus_exp_minus_x': {
        'f': lambda x: np.sin(x) - np.exp(-x),
        'g': lambda x: np.arcsin(np.exp(-x)),
        'sympy': lambda x: sp.sin(x) - sp.exp(-x),
        'params': {'x0': 0.6, 'a': 0, 'b': 1},
        'description': 'f(x) = sin(x) - e^(-x)'
    }
}

# Global functions (set dynamically)
func = None
g = None
sympy_func = None

def set_function(func_name):
    global func, g, sympy_func
    if func_name not in FUNCTIONS:
        raise ValueError(f"Function {func_name} not found. Choose from: {list(FUNCTIONS.keys())}")
    func_info = FUNCTIONS[func_name]
    func = func_info['f']
    g = func_info['g']
    x = sp.Symbol('x')
    sympy_func = func_info['sympy'](x)
    return func_info['params'], func_info['description']

# Save table to text file
def save_table(method_name, table_data, headers):
    with open(f"tables/{method_name}_table.txt", "w") as f:
        f.write(tabulate(table_data, headers=headers, tablefmt="plain", floatfmt=".6f"))

# Aitken's Delta-Squared Acceleration
def aitken_acceleration(sequence):
    if len(sequence) < 3:
        return []
    accelerated = []
    for i in range(len(sequence) - 2):
        x_n = sequence[i]
        x_n1 = sequence[i + 1]
        x_n2 = sequence[i + 2]
        denom = x_n2 - 2 * x_n1 + x_n
        if abs(denom) < 1e-10:
            continue
        x_hat = x_n - (x_n1 - x_n) ** 2 / denom
        accelerated.append(x_hat)
    return accelerated

# Bisection Method
def bisection(a, b, e, max_iterations):
    if abs(func(a)) < 1e-10:
        return a, [1], [a], [a], [b], [func(a)], [0.0]
    if abs(func(b)) < 1e-10:
        return b, [1], [b], [a], [b], [func(b)], [0.0]
    if func(a) * func(b) >= 0:
        return None, [], [], [], [], [], []
    iterations = []
    intervals_a = []
    intervals_b = []
    midpoints = []
    func_values = []
    errors = []
    iteration = 0
    while iteration < max_iterations:
        c = (a + b) / 2
        f_c = func(c)
        iterations.append(iteration + 1)
        midpoints.append(c)
        intervals_a.append(a)
        intervals_b.append(b)
        func_values.append(f_c)
        if f_c == 0.0:
            errors.append(0.0)
            break
        if func(c) * func(a) < 0:
            b = c
        else:
            a = c
        c_next = (a + b) / 2
        error = abs(c_next - c)
        errors.append(error)
        if error < e:
            break
        iteration += 1
    return c, iterations, midpoints, intervals_a, intervals_b, func_values, errors

# Newton-Raphson Method
def get_derivative(sympy_f):
    x = sp.Symbol('x')
    df = sp.diff(sympy_f, x)
    return sp.lambdify(x, df, 'numpy')

def newton(x0, e, max_iterations, sympy_f):
    derivative = get_derivative(sympy_f)
    iterations = []
    x_values = []
    func_values = []
    deriv_values = []
    errors = []
    x_n = float(x0)
    iteration = 0
    while iteration < max_iterations:
        f_n = func(x_n)
        df_n = derivative(x_n)
        if abs(df_n) < 1e-10:
            iterations.append(iteration + 1)
            x_values.append(x_n)
            func_values.append(f_n)
            deriv_values.append(df_n)
            errors.append(np.nan)
            break
        x_next = x_n - f_n / df_n
        error = abs(x_next - x_n)
        iterations.append(iteration + 1)
        x_values.append(x_n)
        func_values.append(f_n)
        deriv_values.append(df_n)
        errors.append(error)
        if error < e:
            break
        x_n = x_next
        iteration += 1
    return x_n, iterations, x_values, func_values, deriv_values, errors

# Fixed Point Iteration Method
def fixed_point(x0, e, max_iterations):
    iterations = []
    x_values = []
    g_values = []
    errors = []
    x_n = float(x0)
    iteration = 0
    while iteration < max_iterations:
        x_next = g(x_n)
        error = abs(x_next - x_n)
        iterations.append(iteration + 1)
        x_values.append(x_n)
        g_values.append(x_next)
        errors.append(error)
        if error < e:
            break
        x_n = x_next
        iteration += 1
    return x_n, iterations, x_values, g_values, errors

# False Position Method
def false_position(a, b, e, max_iterations):
    if abs(func(a)) < 1e-10:
        return a, [1], [a], [a], [b], [func(a)], [0.0]
    if abs(func(b)) < 1e-10:
        return b, [1], [b], [a], [b], [func(b)], [0.0]
    if func(a) * func(b) >= 0:
        return None, [], [], [], [], [], []
    iterations = []
    intervals_a = []
    intervals_b = []
    c_values = []
    func_c_values = []
    errors = []
    iteration = 0
    while iteration < max_iterations:
        c = a - func(a) * (b - a) / (func(b) - func(a))
        f_c = func(c)
        iterations.append(iteration + 1)
        intervals_a.append(a)
        intervals_b.append(b)
        c_values.append(c)
        func_c_values.append(f_c)
        if f_c == 0.0:
            errors.append(0.0)
            break
        if func(c) * func(a) < 0:
            b = c
        else:
            a = c
        c_next = a - func(a) * (b - a) / (func(b) - func(a))
        error = abs(c_next - c)
        errors.append(error)
        if error < e:
            break
        iteration += 1
    return c, iterations, c_values, intervals_a, intervals_b, func_c_values, errors

# Secant Method
def secant(x0, x1, e, max_iterations):
    iterations = []
    x_values = []
    func_values = []
    errors = []
    x_n_minus_1 = float(x0)
    x_n = float(x1)
    iteration = 0
    while iteration < max_iterations:
        f_n = func(x_n)
        f_n_minus_1 = func(x_n_minus_1)
        if abs(f_n - f_n_minus_1) < 1e-10:
            break
        x_next = x_n - f_n * (x_n - x_n_minus_1) / (f_n - f_n_minus_1)
        error = abs(x_next - x_n)
        iterations.append(iteration + 1)
        x_values.append(x_n)
        func_values.append(f_n)
        errors.append(error)
        if error < e:
            break
        x_n_minus_1 = x_n
        x_n = x_next
        iteration += 1
    return x_n, iterations, x_values, func_values, errors

# Secant Using Last Two Iterations
def secant_from_last_two(newton_x_values, bis_midpoints, e, max_iterations):
    if len(newton_x_values) >= 2:
        newton_x0 = newton_x_values[-2]
        newton_x1 = newton_x_values[-1]
        print("\nSecant Method with Last Two Newton-Raphson Iterations (x_0 = {:.6f}, x_1 = {:.6f}):".format(newton_x0, newton_x1))
        root_newton_sec, iter_newton_sec, x_newton_sec, func_newton_sec, errors_newton_sec = secant(newton_x0, newton_x1, e, max_iterations)
        table_data_newton_sec = [
            [it, f"{x_val:.6f}", f"{f_val:.6f}", f"{err:.6f}"]
            for it, x_val, f_val, err in zip(iter_newton_sec, x_newton_sec, func_newton_sec, errors_newton_sec)
        ]
        print(tabulate(
            table_data_newton_sec,
            headers=["Iteration", "x_n", "f(x_n)", "Error |x_{n+1} - x_n|"],
            tablefmt="grid",
            floatfmt=".6f",
            stralign="center",
            numalign="center"
        ))
        save_table("secant_newton", table_data_newton_sec, ["Iteration", "x_n", "f(x_n)", "Error |x_{n+1} - x_n|"])
        if len(errors_newton_sec) > 0 and errors_newton_sec[-1] < e:
            print(f"Stopped at iteration {iter_newton_sec[-1]} because |x_{{n+1}} - x_n| = {errors_newton_sec[-1]:.6e} < ε = {e:.6e}")
        elif len(func_newton_sec) > 0 and abs(func_newton_sec[-1]) < 1e-10:
            print(f"Stopped at iteration {iter_newton_sec[-1]} because |f(x_n)| = {abs(func_newton_sec[-1]):.6e} < 1e-10")
        else:
            print(f"Stopped at iteration {iter_newton_sec[-1]} after reaching max iterations ({max_iterations})")

    if len(bis_midpoints) >= 2:
        bis_x0 = bis_midpoints[-2]
        bis_x1 = bis_midpoints[-1]
        print("\nSecant Method with Last Two Bisection Iterations (x_0 = {:.6f}, x_1 = {:.6f}):".format(bis_x0, bis_x1))
        root_bis_sec, iter_bis_sec, x_bis_sec, func_bis_sec, errors_bis_sec = secant(bis_x0, bis_x1, e, max_iterations)
        table_data_bis_sec = [
            [it, f"{x_val:.6f}", f"{f_val:.6f}", f"{err:.6f}"]
            for it, x_val, f_val, err in zip(iter_bis_sec, x_bis_sec, func_bis_sec, errors_bis_sec)
        ]
        print(tabulate(
            table_data_bis_sec,
            headers=["Iteration", "x_n", "f(x_n)", "Error |x_{n+1} - x_n|"],
            tablefmt="grid",
            floatfmt=".6f",
            stralign="center",
            numalign="center"
        ))
        save_table("secant_bisection", table_data_bis_sec, ["Iteration", "x_n", "f(x_n)", "Error |x_{n+1} - x_n|"])
        if len(errors_bis_sec) > 0 and errors_bis_sec[-1] < e:
            print(f"Stopped at iteration {iter_bis_sec[-1]} because |x_{{n+1}} - x_n| = {errors_bis_sec[-1]:.6e} < ε = {e:.6e}")
        elif len(func_bis_sec) > 0 and abs(func_bis_sec[-1]) < 1e-10:
            print(f"Stopped at iteration {iter_bis_sec[-1]} because |f(x_n)| = {abs(func_bis_sec[-1]):.6e} < 1e-10")
        else:
            print(f"Stopped at iteration {iter_bis_sec[-1]} after reaching max iterations ({max_iterations})")

# Comparison Function
def compare_methods(x0, a, b, e, max_iterations=10):
    initial_guess = (a + b) / 2
    try:
        true_root = float(fsolve(func, initial_guess)[0])
        if abs(func(true_root)) > 1e-10:
            print("Warning: Computed true_root may be inaccurate, |f(true_root)| > 1e-10")
    except Exception as e:
        print(f"Warning: Failed to compute true_root: {e}. Using N/A for error calculations.")
        true_root = None

    results = []

    # Bisection
    print("\nBisection Method Iteration Table:")
    start_time = time.time()
    root_bis, iter_bis, c_bis, intervals_a_bis, intervals_b_bis, func_values_bis, errors_bis = bisection(a, b, e, max_iterations)
    time_bis = time.time() - start_time
    table_data_bis = [
        [it, f"{a_val:.6f}", f"{b_val:.6f}", f"{c_val:.6f}", f"{fc_val:.6f}", f"{err:.6f}"]
        for it, a_val, b_val, c_val, fc_val, err in zip(iter_bis, intervals_a_bis, intervals_b_bis, c_bis, func_values_bis, errors_bis)
    ]
    print(tabulate(
        table_data_bis,
        headers=["Iteration", "a", "b", "c", "f(c)", "Error |x_{n+1} - x_n|"],
        tablefmt="grid",
        floatfmt=".6f",
        stralign="center",
        numalign="center"
    ))
    save_table("bisection", table_data_bis, ["Iteration", "a", "b", "c", "f(c)", "Error |x_{n+1} - x_n|"])
    if len(iter_bis) > 0:
        if len(errors_bis) > 0 and errors_bis[-1] < e:
            print(f"Stopped at iteration {iter_bis[-1]} because |x_{{n+1}} - x_n| = {errors_bis[-1]:.6e} < ε = {e:.6e}")
        elif func_values_bis[-1] == 0.0:
            print(f"Stopped at iteration {iter_bis[-1]} because f(c) = 0")
        elif len(iter_bis) >= max_iterations:
            print(f"Stopped at iteration {iter_bis[-1]} after reaching max iterations ({max_iterations})")
        else:
            print(f"Stopped at iteration {iter_bis[-1]} unexpectedly")
    else:
        if root_bis is not None:
            print(f"Stopped at iteration 1 because endpoint root found at x = {root_bis:.6f}")
        else:
            print("Failed to iterate: No root in interval or invalid interval")
    error_bis = abs(root_bis - true_root) if root_bis is not None and true_root is not None else np.nan
    stop_error_bis = errors_bis[-1] if errors_bis else np.nan
    results.append(['Bisection', len(iter_bis), f"{root_bis:.19f}" if root_bis else 'N/A', f"{error_bis:.19f}" if not np.isnan(error_bis) else 'N/A', f"{stop_error_bis:.19f}" if not np.isnan(stop_error_bis) else 'N/A', time_bis])

    # Bisection with Aitken
    print("\nBisection Method with Aitken Acceleration:")
    aitken_bis = aitken_acceleration(c_bis)
    if aitken_bis:
        table_data_bis_aitken = [
            [i + 1, f"{x_hat:.6f}", f"{func(x_hat):.19f}", f"{abs(x_hat - true_root):.6f}" if true_root is not None else "N/A"]
            for i, x_hat in enumerate(aitken_bis)
        ]
        print(tabulate(
            table_data_bis_aitken,
            headers=["Iteration", "x_hat", "f(x_hat)", "Error"],
            tablefmt="grid",
            floatfmt=".6f",
            stralign="center",
            numalign="center"
        ))
        save_table("bisection_aitken", table_data_bis_aitken, ["Iteration", "x_hat", "f(x_hat)", "Error"])
    else:
        print("Not enough iterations for Aitken acceleration.")

    # Newton-Raphson
    print("\nNewton-Raphson Method Iteration Table:")
    start_time = time.time()
    root_new, iter_new, x_new, func_values_new, deriv_values_new, errors_new = newton(x0, e, max_iterations, sympy_func)
    time_new = time.time() - start_time
    table_data_new = [
        [it, f"{x_val:.6f}", f"{f_val:.6f}", f"{df_val:.6f}", f"{err:.6f}"]
        for it, x_val, f_val, df_val, err in zip(iter_new, x_new, func_values_new, deriv_values_new, errors_new)
    ]
    print(tabulate(
        table_data_new,
        headers=["Iteration", "x_n", "f(x_n)", "f'(x_n)", "Error |x_{n+1} - x_n|"],
        tablefmt="grid",
        floatfmt=".6f",
        stralign="center",
        numalign="center"
    ))
    save_table("newton", table_data_new, ["Iteration", "x_n", "f(x_n)", "f'(x_n)", "Error |x_{n+1} - x_n|"])
    if len(errors_new) > 0 and errors_new[-1] < e and not np.isnan(errors_new[-1]):
        print(f"Stopped at iteration {iter_new[-1]} because |x_{{n+1}} - x_n| = {errors_new[-1]:.6e} < ε = {e:.6e}")
    elif len(func_values_new) > 0 and abs(func_values_new[-1]) < 1e-10:
        print(f"Stopped at iteration {iter_new[-1]} because |f(x_n)| = {abs(func_values_new[-1]):.6e} < 1e-10")
    elif len(deriv_values_new) > 0 and abs(deriv_values_new[-1]) < 1e-10:
        print(f"Stopped at iteration {iter_new[-1]} because |f'(x_n)| = {abs(deriv_values_new[-1]):.6e} < 1e-10")
    else:
        print(f"Stopped at iteration {iter_new[-1]} after reaching max iterations ({max_iterations})")
    error_new = abs(root_new - true_root) if true_root is not None else np.nan
    stop_error_new = errors_new[-1] if errors_new and not np.isnan(errors_new[-1]) else np.nan
    results.append(['Newton-Raphson', len(iter_new), f"{root_new:.19f}", f"{error_new:.19f}" if not np.isnan(error_new) else 'N/A', f"{stop_error_new:.19f}" if not np.isnan(stop_error_new) else 'N/A', time_new])

    # Newton-Raphson with Aitken
    print("\nNewton-Raphson Method with Aitken Acceleration:")
    aitken_new = aitken_acceleration(x_new)
    if aitken_new:
        table_data_new_aitken = [
            [i + 1, f"{x_hat:.6f}", f"{func(x_hat):.19f}", f"{abs(x_hat - true_root):.6f}" if true_root is not None else "N/A"]
            for i, x_hat in enumerate(aitken_new)
        ]
        print(tabulate(
            table_data_new_aitken,
            headers=["Iteration", "x_hat", "f(x_hat)", "Error"],
            tablefmt="grid",
            floatfmt=".6f",
            stralign="center",
            numalign="center"
        ))
        save_table("newton_aitken", table_data_new_aitken, ["Iteration", "x_hat", "f(x_hat)", "Error"])
    else:
        print("Not enough iterations for Aitken acceleration.")

    # Fixed Point
    print("\nFixed Point Iteration Method Table:")
    start_time = time.time()
    root_fix, iter_fix, x_fix, g_values_fix, errors_fix = fixed_point(x0, e, max_iterations)
    time_fix = time.time() - start_time
    table_data_fix = [
        [it, f"{x_val:.6f}", f"{g_val:.6f}", f"{err:.6f}"]
        for it, x_val, g_val, err in zip(iter_fix, x_fix, g_values_fix, errors_fix)
    ]
    print(tabulate(
        table_data_fix,
        headers=["Iteration", "x_n", "g(x_n)", "Error |x_{n+1} - x_n|"],
        tablefmt="grid",
        floatfmt=".6f",
        stralign="center",
        numalign="center"
    ))
    save_table("fixed_point", table_data_fix, ["Iteration", "x_n", "g(x_n)", "Error |x_{n+1} - x_n|"])
    if len(errors_fix) > 0 and errors_fix[-1] < e:
        print(f"Stopped at iteration {iter_fix[-1]} because |x_{{n+1}} - x_n| = {errors_fix[-1]:.6e} < ε = {e:.6e}")
    elif len(x_fix) > 0 and abs(func(x_fix[-1])) < 1e-10:
        print(f"Stopped at iteration {iter_fix[-1]} because |f(x_n)| = {abs(func(x_fix[-1])):.6e} < 1e-10")
    else:
        print(f"Stopped at iteration {iter_fix[-1]} after reaching max iterations ({max_iterations})")
    error_fix = abs(root_fix - true_root) if true_root is not None else np.nan
    stop_error_fix = errors_fix[-1] if errors_fix else np.nan
    results.append(['Fixed Point', len(iter_fix), f"{root_fix:.19f}", f"{error_fix:.19f}" if not np.isnan(error_fix) else 'N/A', f"{stop_error_fix:.19f}" if not np.isnan(stop_error_fix) else 'N/A', time_fix])

    # Fixed Point with Aitken
    print("\nFixed Point Method with Aitken Acceleration:")
    aitken_fix = aitken_acceleration(x_fix)
    if aitken_fix:
        table_data_fix_aitken = [
            [i + 1, f"{x_hat:.6f}", f"{func(x_hat):.19f}", f"{abs(x_hat - true_root):.6f}" if true_root is not None else "N/A"]
            for i, x_hat in enumerate(aitken_fix)
        ]
        print(tabulate(
            table_data_fix_aitken,
            headers=["Iteration", "x_hat", "f(x_hat)", "Error"],
            tablefmt="grid",
            floatfmt=".6f",
            stralign="center",
            numalign="center"
        ))
        save_table("fixed_point_aitken", table_data_fix_aitken, ["Iteration", "x_hat", "f(x_hat)", "Error"])
    else:
        print("Not enough iterations for Aitken acceleration.")

    # False Position
    print("\nFalse Position Method Iteration Table:")
    start_time = time.time()
    root_fal, iter_fal, c_fal, intervals_a_fal, intervals_b_fal, func_c_values_fal, errors_fal = false_position(a, b, e, max_iterations)
    time_fal = time.time() - start_time
    table_data_fal = [
        [it, f"{a_val:.6f}", f"{b_val:.6f}", f"{c_val:.6f}", f"{fc_val:.6f}", f"{err:.6f}"]
        for it, a_val, b_val, c_val, fc_val, err in zip(iter_fal, intervals_a_fal, intervals_b_fal, c_fal, func_c_values_fal, errors_fal)
    ]
    print(tabulate(
        table_data_fal,
        headers=["Iteration", "a", "b", "c", "f(c)", "Error |x_{n+1} - x_n|"],
        tablefmt="grid",
        floatfmt=".6f",
        stralign="center",
        numalign="center"
    ))
    save_table("false_position", table_data_fal, ["Iteration", "a", "b", "c", "f(c)", "Error |x_{n+1} - x_n|"])
    if len(iter_fal) > 0:
        if len(errors_fal) > 0 and errors_fal[-1] < e:
            print(f"Stopped at iteration {iter_fal[-1]} because |x_{{n+1}} - x_n| = {errors_fal[-1]:.6e} < ε = {e:.6e}")
        elif len(func_c_values_fal) > 0 and func_c_values_fal[-1] == 0.0:
            print(f"Stopped at iteration {iter_fal[-1]} because f(c) = 0.0")
        else:
            print(f"Stopped at iteration {iter_fal[-1]} after reaching max iterations ({max_iterations})")
    else:
        if root_fal is not None:
            print(f"Stopped at iteration 1 because endpoint root found at x = {root_fal:.6f}")
        else:
            print("Failed to iterate: No root in interval or invalid interval")
    error_fal = abs(root_fal - true_root) if root_fal is not None and true_root is not None else np.nan
    stop_error_fal = errors_fal[-1] if errors_fal else np.nan
    results.append(['False Position', len(iter_fal), f"{root_fal:.19f}" if root_fal else 'N/A', f"{error_fal:.19f}" if not np.isnan(error_fal) else 'N/A', f"{stop_error_fal:.19f}" if not np.isnan(stop_error_fal) else 'N/A', time_fal])

    # False Position with Aitken
    print("\nFalse Position with Aitken Acceleration:")
    aitken_fal = aitken_acceleration(c_fal)
    if aitken_fal:
        table_data_fal_aitken = [
            [i + 1, f"{x_hat:.6f}", f"{func(x_hat):.19f}", f"{abs(x_hat - true_root):.6f}" if true_root is not None else "N/A"]
            for i, x_hat in enumerate(aitken_fal)
        ]
        print(tabulate(
            table_data_fal_aitken,
            headers=["Iteration", "x_hat", "f(x_hat)", "Error"],
            tablefmt="grid",
            floatfmt=".6f",
            stralign="center",
            numalign="center"
        ))
        save_table("false_position_aitken", table_data_fal_aitken, ["Iteration", "x_hat", "f(x_hat)", "Error"])
    else:
        print("Not enough iterations for Aitken acceleration.")

    # Secant
    print("\nSecant Method Iteration Table:")
    start_time = time.time()
    root_sec, iter_sec, x_sec, func_values_sec, errors_sec = secant(x0, b, e, max_iterations)
    time_sec = time.time() - start_time
    table_data_sec = [
        [it, f"{x_val:.6f}", f"{f_val:.6f}", f"{err:.6f}"]
        for it, x_val, f_val, err in zip(iter_sec, x_sec, func_values_sec, errors_sec)
    ]
    print(tabulate(
        table_data_sec,
        headers=["Iteration", "x_n", "f(x_n)", "Error |x_{n+1} - x_n|"],
        tablefmt="grid",
        floatfmt=".6f",
        stralign="center",
        numalign="center"
    ))
    save_table("secant", table_data_sec, ["Iteration", "x_n", "f(x_n)", "Error |x_{n+1} - x_n|"])
    if len(errors_sec) > 0 and errors_sec[-1] < e:
        print(f"Stopped at iteration {iter_sec[-1]} because |x_{{n+1}} - x_n| = {errors_sec[-1]:.6e} < ε = {e:.6e}")
    elif len(func_values_sec) > 0 and abs(func_values_sec[-1]) < 1e-10:
        print(f"Stopped at iteration {iter_sec[-1]} because |f(x_n)| = {abs(func_values_sec[-1]):.6e} < 1e-10")
    else:
        print(f"Stopped at iteration {iter_sec[-1]} after reaching max iterations ({max_iterations})")
    error_sec = abs(root_sec - true_root) if root_sec is not None and true_root is not None else np.nan
    stop_error_sec = errors_sec[-1] if errors_sec else np.nan
    results.append(['Secant', len(iter_sec), f"{root_sec:.19f}" if root_sec else 'N/A', f"{error_sec:.19f}" if not np.isnan(error_sec) else 'N/A', f"{stop_error_sec:.19f}" if not np.isnan(stop_error_sec) else 'N/A', time_sec])

    # Secant with Aitken
    print("\nSecant Method with Aitken Acceleration:")
    aitken_sec = aitken_acceleration(x_sec)
    if aitken_sec:
        table_data_sec_aitken = [
            [i + 1, f"{x_hat:.6f}", f"{func(x_hat):.19f}", f"{abs(x_hat - true_root):.6f}" if true_root is not None else "N/A"]
            for i, x_hat in enumerate(aitken_sec)
        ]
        print(tabulate(
            table_data_sec_aitken,
            headers=["Iteration", "x_hat", "f(x_hat)", "Error"],
            tablefmt="grid",
            floatfmt=".6f",
            stralign="center",
            numalign="center"
        ))
        save_table("secant_aitken", table_data_sec_aitken, ["Iteration", "x_hat", "f(x_hat)", "Error"])
    else:
        print("Not enough iterations for Aitken acceleration.")

    # Comparison Table
    print("\nComparison of Root-Finding Methods:")
    print(tabulate(
        results,
        headers=["Method", "Iterations", "Root", "Abs Error |x - x_true|", "Stopping Error", "Time (s)"],
        tablefmt="grid",
        floatfmt=(".0f", ".0f", ".19f", ".19f", ".19f", ".6f"),
        stralign="center",
        numalign="center"
    ))
    save_table("comparison", results, ["Method", "Iterations", "Root", "Abs Error |x - x_true|", "Stopping Error", "Time (s)"])

    # Convergence Plot
    plt.figure(figsize=(10, 6))
    try:
        plt.style.use('seaborn-v0_8')
    except OSError:
        print("Warning: 'seaborn-v0_8' style not found. Falling back to 'ggplot'.")
        plt.style.use('ggplot')
        plt.rcParams.update({
            'axes.facecolor': '#f5f5f5',
            'axes.edgecolor': 'black',
            'grid.color': 'gray',
            'grid.linestyle': '--',
            'grid.alpha': 0.7,
            'axes.linewidth': 1
        })
    if root_bis is not None and true_root is not None:
        errors_bis_plot = [abs(x - true_root) for x in c_bis]
        plt.plot(range(1, len(errors_bis_plot) + 1), errors_bis_plot, color='#1f77b4', linestyle='-', marker='o', linewidth=2, markersize=8, label='Bisection')
    if true_root is not None:
        errors_new_plot = [abs(x - true_root) for x in x_new]
        plt.plot(range(1, len(errors_new_plot) + 1), errors_new_plot, color='#ff7f0e', linestyle='-', marker='s', linewidth=2, markersize=8, label='Newton-Raphson')
        errors_fix_plot = [abs(x - true_root) for x in x_fix]
        plt.plot(range(1, len(errors_fix_plot) + 1), errors_fix_plot, color='#2ca02c', linestyle='-', marker='^', linewidth=2, markersize=8, label='Fixed Point')
        errors_sec_plot = [abs(x - true_root) for x in x_sec]
        plt.plot(range(1, len(errors_sec_plot) + 1), errors_sec_plot, color='#9467bd', linestyle='-', marker='v', linewidth=2, markersize=8, label='Secant')
    if root_fal is not None and true_root is not None:
        errors_fal_plot = [abs(x - true_root) for x in c_fal]
        plt.plot(range(1, len(errors_fal_plot) + 1), errors_fal_plot, color='#d62728', linestyle='-', marker='D', linewidth=2, markersize=8, label='False Position')
    plt.axhline(y=e, color='gray', linestyle='--', linewidth=1, alpha=0.5, label=f'Tolerance ε = {e}')
    plt.yscale('log')
    plt.title('Convergence of Root-Finding Methods for $f(x)$', fontsize=16, pad=15)
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Error $|x_n - \\text{true}_x|$', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, loc='upper right', frameon=True, edgecolor='black')
    plt.tight_layout()
    plt.savefig('graphs/convergence_comparison.png', dpi=300, format='png', bbox_inches='tight')
    plt.close()

    # Trajectory Plot
    plt.figure(figsize=(10, 6))
    try:
        plt.style.use('seaborn-v0_8')
    except OSError:
        plt.style.use('ggplot')
        plt.rcParams.update({
            'axes.facecolor': '#f5f5f5',
            'axes.edgecolor': 'black',
            'grid.color': 'gray',
            'grid.linestyle': '--',
            'grid.alpha': 0.7,
            'axes.linewidth': 1
        })
    x = np.linspace(a - 0.5, b + 0.5, 200)
    y = func(x)
    plt.plot(x, y, color='#333333', linestyle='-', linewidth=2.5, label='$f(x)$')
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    if true_root is not None:
        plt.axvline(x=true_root, color='black', linestyle=':', linewidth=1.5, alpha=0.7, label=f'Root ≈ {true_root:.6f}')
    if root_bis is not None:
        bis_x = c_bis
        bis_y = [func(x_n) for x_n in bis_x]
        plt.plot(bis_x, bis_y, color='#1f77b4', linestyle='--', linewidth=1, alpha=0.7)
        for i, (x_n, y_n) in enumerate(zip(bis_x, bis_y)):
            plt.scatter([x_n], [y_n], color='#1f77b4', marker='o', s=100, alpha=0.8, label='Bisection' if i == 0 else '')
    new_x = x_new
    new_y = [func(x_n) for x_n in x_new]
    plt.plot(new_x, new_y, color='#ff7f0e', linestyle='--', linewidth=1, alpha=0.7)
    for i, (x_n, y_n) in enumerate(zip(new_x, new_y)):
        plt.scatter([x_n], [y_n], color='#ff7f0e', marker='s', s=100, alpha=0.8, label='Newton-Raphson' if i == 0 else '')
    fix_x = x_fix
    fix_y = [func(x_n) for x_n in x_fix]
    plt.plot(fix_x, fix_y, color='#2ca02c', linestyle='--', linewidth=1, alpha=0.7)
    for i, (x_n, y_n) in enumerate(zip(fix_x, fix_y)):
        plt.scatter([x_n], [y_n], color='#2ca02c', marker='^', s=100, alpha=0.8, label='Fixed Point' if i == 0 else '')
    if root_fal is not None:
        fal_x = c_fal
        fal_y = [func(x_n) for x_n in c_fal]
        plt.plot(fal_x, fal_y, color='#d62728', linestyle='--', linewidth=1, alpha=0.7)
        for i, (x_n, y_n) in enumerate(zip(fal_x, fal_y)):
            plt.scatter([x_n], [y_n], color='#d62728', marker='D', s=100, alpha=0.8, label='False Position' if i == 0 else '')
    sec_x = x_sec
    sec_y = [func(x_n) for x_n in x_sec]
    plt.plot(sec_x, sec_y, color='#9467bd', linestyle='--', linewidth=1, alpha=0.7)
    for i, (x_n, y_n) in enumerate(zip(sec_x, sec_y)):
        plt.scatter([x_n], [y_n], color='#9467bd', marker='v', s=100, alpha=0.8, label='Secant' if i == 0 else '')
    plt.title('Trajectory of Root-Finding Methods for $f(x)$', fontsize=16, pad=15)
    plt.xlabel('$x$', fontsize=14)
    plt.ylabel('$f(x)$', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, loc='upper left', bbox_to_anchor=(1, 1), frameon=True, edgecolor='black')
    plt.tight_layout()
    plt.savefig('graphs/trajectory_comparison.png', dpi=300, format='png', bbox_inches='tight')
    plt.close()

    # Number of Iterations Comparison Plot
    plt.figure(figsize=(10, 6))
    try:
        plt.style.use('seaborn-v0_8')
    except OSError:
        plt.style.use('ggplot')
        plt.rcParams.update({
            'axes.facecolor': '#f5f5f5',
            'axes.edgecolor': 'black',
            'grid.color': 'gray',
            'grid.linestyle': '--',
            'grid.alpha': 0.7,
            'axes.linewidth': 1
        })
    methods = ['Bisection', 'Newton-Raphson', 'Fixed Point', 'False Position', 'Secant']
    iterations = [len(iter_bis), len(iter_new), len(iter_fix), len(iter_fal), len(iter_sec)]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    bars = plt.bar(methods, iterations, color=colors)
    plt.title('Number of Iterations for Root-Finding Methods', fontsize=16, pad=15)
    plt.xlabel('Method', fontsize=14)
    plt.ylabel('Iterations', fontsize=14)
    plt.xticks(fontsize=12, rotation=15)
    plt.yticks(range(0, max(iterations) + 2, 1), fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{int(height)}', ha='center', va='bottom', fontsize=12)
    plt.tight_layout()
    plt.savefig('graphs/iterations_comparison.png', dpi=300, format='png', bbox_inches='tight')
    plt.close()

    # Error Ratio Plot
    plt.figure(figsize=(10, 6))
    try:
        plt.style.use('seaborn-v0_8')
    except OSError:
        plt.style.use('ggplot')
        plt.rcParams.update({
            'axes.facecolor': '#f5f5f5',
            'axes.edgecolor': 'black',
            'grid.color': 'gray',
            'grid.linestyle': '--',
            'grid.alpha': 0.7,
            'axes.linewidth': 1
        })
    if true_root is not None:
        p_values = {
            'Bisection': 1,
            'Newton-Raphson': 2,
            'Fixed Point': 1,
            'False Position': 1,
            'Secant': (1 + np.sqrt(5)) / 2
        }
        if root_bis is not None and len(c_bis) > 1:
            errors_bis = [abs(x - true_root) for x in c_bis]
            ratios_bis = [errors_bis[i+1] / (errors_bis[i]**p_values['Bisection'] + 1e-10) for i in range(len(errors_bis)-1)]
            plt.plot(range(2, len(ratios_bis)+2), ratios_bis, color='#1f77b4', linestyle='-', marker='o', linewidth=2, markersize=6, label=f'Bisection (p={p_values["Bisection"]})')
        if len(x_new) > 1:
            errors_new = [abs(x - true_root) for x in x_new]
            ratios_new = [errors_new[i+1] / (errors_new[i]**p_values['Newton-Raphson'] + 1e-10) for i in range(len(errors_new)-1)]
            plt.plot(range(2, len(ratios_new)+2), ratios_new, color='#ff7f0e', linestyle='-', marker='s', linewidth=2, markersize=6, label=f'Newton-Raphson (p={p_values["Newton-Raphson"]})')
        if len(x_fix) > 1:
            errors_fix = [abs(x - true_root) for x in x_fix]
            ratios_fix = [errors_fix[i+1] / (errors_fix[i]**p_values['Fixed Point'] + 1e-10) for i in range(len(errors_fix)-1)]
            plt.plot(range(2, len(ratios_fix)+2), ratios_fix, color='#2ca02c', linestyle='-', marker='^', linewidth=2, markersize=6, label=f'Fixed Point (p={p_values["Fixed Point"]})')
        if root_fal is not None and len(c_fal) > 1:
            errors_fal = [abs(x - true_root) for x in c_fal]
            ratios_fal = [errors_fal[i+1] / (errors_fal[i]**p_values['False Position'] + 1e-10) for i in range(len(errors_fal)-1)]
            plt.plot(range(2, len(ratios_fal)+2), ratios_fal, color='#d62728', linestyle='-', marker='D', linewidth=2, markersize=6, label=f'False Position (p={p_values["False Position"]})')
        if len(x_sec) > 1:
            errors_sec = [abs(x - true_root) for x in x_sec]
            ratios_sec = [errors_sec[i+1] / (errors_sec[i]**p_values['Secant'] + 1e-10) for i in range(len(errors_sec)-1)]
            plt.plot(range(2, len(ratios_sec)+2), ratios_sec, color='#9467bd', linestyle='-', marker='v', linewidth=2, markersize=6, label=f'Secant (p={p_values["Secant"]:.3f})')
        plt.yscale('log')
        plt.title('Error Ratio $|e_{n+1}| / |e_n|^p$', fontsize=14, pad=15)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('$|e_{n+1}| / |e_n|^p$', fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10, loc='upper right')
        plt.tight_layout()
        plt.savefig('graphs/error_ratio_plot.png', dpi=300, format='png', bbox_inches='tight')
        plt.close()

    # Aitken’s Acceleration Improvement Plot
    plt.figure(figsize=(10, 6))
    try:
        plt.style.use('seaborn-v0_8')
    except OSError:
        plt.style.use('ggplot')
        plt.rcParams.update({
            'axes.facecolor': '#f5f5f5',
            'axes.edgecolor': 'black',
            'grid.color': 'gray',
            'grid.linestyle': '--',
            'grid.alpha': 0.7,
            'axes.linewidth': 1
        })
    if true_root is not None:
        if root_bis is not None and len(aitken_bis) > 0:
            errors_bis = [abs(x - true_root) for x in c_bis]
            errors_bis_aitken = [abs(x_hat - true_root) for x_hat in aitken_bis]
            plt.plot(range(1, len(errors_bis) + 1), errors_bis, color='#1f77b4', linestyle='-', marker='o', linewidth=2, markersize=8, label='Bisection')
            plt.plot(range(1, len(errors_bis_aitken) + 1), errors_bis_aitken, color='#1f77b4', linestyle='--', marker='s', linewidth=2, markersize=8, label='Bisection (Aitken)')
        if len(aitken_new) > 0:
            errors_new = [abs(x - true_root) for x in x_new]
            errors_new_aitken = [abs(x_hat - true_root) for x_hat in aitken_new]
            plt.plot(range(1, len(errors_new) + 1), errors_new, color='#ff7f0e', linestyle='-', marker='s', linewidth=2, markersize=8, label='Newton-Raphson')
            plt.plot(range(1, len(errors_new_aitken) + 1), errors_new_aitken, color='#ff7f0e', linestyle='--', marker='^', linewidth=1, markersize=8, label='Newton-Raphson (Aitken)')
        if len(aitken_fix) > 0:
            errors_fix = [abs(x - true_root) for x in x_fix]
            errors_fix_aitken = [abs(x_hat - true_root) for x_hat in aitken_fix]
            plt.plot(range(1, len(errors_fix) + 1), errors_fix, color='#2ca02c', linestyle='-', marker='^', linewidth=2, markersize=8, label='Fixed Point')
            plt.plot(range(1, len(errors_fix_aitken) + 1), errors_fix_aitken, color='#2ca02c', linestyle='--', marker='s', linewidth=1, markersize=8, label='Fixed Point (Aitken)')
        if root_fal is not None and len(aitken_fal) > 0:
            errors_fal = [abs(x - true_root) for x in c_fal]
            errors_fal_aitken = [abs(x_hat - true_root) for x_hat in aitken_fal]
            plt.plot(range(1, len(errors_fal) + 1), errors_fal, color='#d62728', linestyle='-', marker='D', linewidth=2, markersize=8, label='False Position')
            plt.plot(range(1, len(errors_fal_aitken) + 1), errors_fal_aitken, color='#d62728', linestyle='--', marker='s', linewidth=1, markersize=8, label='False Position (Aitken)')
        if len(aitken_sec) > 0:
            errors_sec = [abs(x - true_root) for x in x_sec]
            errors_sec_aitken = [abs(x_hat - true_root) for x_hat in aitken_sec]
            plt.plot(range(1, len(errors_sec) + 1), errors_sec, color='#9467bd', linestyle='-', marker='v', linewidth=2, markersize=8, label='Secant')
            plt.plot(range(1, len(errors_sec_aitken) + 1), errors_sec_aitken, color='#9467bd', linestyle='--', marker='s', linewidth=1, markersize=8, label='Secant (Aitken)')
        plt.axhline(y=e, color='gray', linestyle='--', linewidth=1, alpha=0.5, label=f'Tolerance ε = {e}')
        plt.yscale('log')
        plt.title('Aitken’s Acceleration Error Reduction for $f(x)$', fontsize=16, pad=15)
        plt.xlabel('Iteration', fontsize=14)
        plt.ylabel('Error $|x - \\text{true}_x|$', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12, loc='upper right', frameon=True, edgecolor='black')
        plt.tight_layout()
        plt.savefig('graphs/aitken_improvement.png', dpi=300, format='png', bbox_inches='tight')
        plt.close()

    return x_new, c_bis

def main():
    parser = argparse.ArgumentParser(description="Root-Finding Numerical Methods")
    parser.add_argument('--function', type=str, default='x3_minus_cosx',
                        choices=list(FUNCTIONS.keys()),
                        help='Function to solve (default: x3_minus_cosx)')
    parser.add_argument('--x0', type=float, default=None, help='Initial guess (overrides default)')
    parser.add_argument('--a', type=float, default=None, help='Interval start (overrides default)')
    parser.add_argument('--b', type=float, default=None, help='Interval end (overrides default)')
    parser.add_argument('--e', type=float, default=1e-6, help='Tolerance (default: 1e-6)')
    parser.add_argument('--max_iterations', type=int, default=10, help='Max iterations (default: 10)')
    args = parser.parse_args()

    params, description = set_function(args.function)
    print(f"Selected function: {description}")
    x0 = args.x0 if args.x0 is not None else params['x0']
    a = args.a if args.a is not None else params['a']
    b = args.b if args.b is not None else params['b']
    e = args.e
    max_iterations = args.max_iterations

    x_new, c_bis = compare_methods(x0, a, b, e, max_iterations)
    secant_from_last_two(x_new, c_bis, e, max_iterations)

if __name__ == "__main__":
    import os
    os.makedirs("tables", exist_ok=True)
    os.makedirs("graphs", exist_ok=True)
    main()