import sympy as sym
import control as ctrl
import numpy as np
import matplotlib.pyplot as plt

m, ell, x3, x4, M, g, F, m = sym.symbols('m, ell, x3, x4, M, g, F, m')

# φ(F, x3, x4)
phi = 4*m*ell*x4**2*sym.sin(x3) + 4*F - 3*m*g*sym.sin(x3)*sym.cos(x3)
phi /= 4*(M+m) - 3*m*sym.cos(x3)**2

dphi_x3 = phi.diff(x3)
dphi_x4 = phi.diff(x4)
dphi_F = phi.diff(F)

# Equilibrium point
Feq = 0
x3eq = 0
x4eq = 0

dphi_F_eq = dphi_F.subs([(F, Feq), (x3, x3eq), (x4, x4eq)])
dphi_x3_eq = dphi_x3.subs([(F, Feq), (x3, x3eq), (x4, x4eq)])
dphi_x4_eq = dphi_x4.subs([(F, Feq), (x3, x3eq), (x4, x4eq)])

a = dphi_F_eq
b = -dphi_x3_eq
c = 3/(ell*(4*M + m))
d = 3*(M+m)*g/(ell*(4*M + m))

# GIVEN VALUES!
M_value = 0.3
m_value = 0.1
g_value = 9.81
ell_value = 0.35

def evaluate_at_given_parameters(z):
    """
    This function blah blah
    :param z:
    :return:
    """
    return float(z.subs([(M, M_value), (m, m_value), (ell, ell_value), (g, g_value)]))

a_value = evaluate_at_given_parameters(a)
b_value = evaluate_at_given_parameters(b)
c_value = evaluate_at_given_parameters(c)
d_value = evaluate_at_given_parameters(d)

# -----------------------------------
# Control library of Python
# -----------------------------------
transfer_function_F2x3 = ctrl.TransferFunction([-c_value], [1, 0, -d_value])

t_step, x3_step = ctrl.step_response(transfer_function_F2x3)

n_points = 500
t_final = 2
t_span = np.linspace(0, t_final, n_points)# array of time instants
input_signal = np.sin(5 * np.sqrt(t_span **2 * 100))# input signal
tf = ctrl.TransferFunction(1, [0.5, 0.1, 1])# transfer function
t_out, y_out, x_out = ctrl.forced_response(tf, t_span, input_signal)
plt.plot(t_out, y_out)
plt.show()
plt.xlabel("Time/seconds")
plt.ylabel("Angle from Vertical/degrees")