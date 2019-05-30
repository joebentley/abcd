import sympy
from transfer_func import PolyTransferFunc

s, gamma = sympy.symbols('s gamma')

tf_1dof = (s - sympy.I * gamma) / (s + sympy.I * gamma)
tf_2dof = (s**2 / gamma + s - sympy.I * gamma) / (-s**2 / gamma + s + sympy.I * gamma)


def test_expr_gives_expected_result():
    poly_fraction = PolyTransferFunc.from_transfer_function(tf_1dof, s)
    assert poly_fraction.expr == tf_1dof


def test_poly_fraction_from_transfer_func_gives_expected_result():
    poly_fraction = PolyTransferFunc.from_transfer_function(tf_1dof, s)
    assert poly_fraction.numer.equals(s - sympy.I * gamma)
    assert poly_fraction.denom.equals(s + sympy.I * gamma)
    poly_fraction = PolyTransferFunc.from_transfer_function(tf_2dof, s)
    assert poly_fraction.numer.equals(s**2 / gamma + s - sympy.I * gamma)


def test_getting_coefficients_gives_expected_result():
    poly_fraction = PolyTransferFunc.from_transfer_function(tf_2dof, s)
    assert poly_fraction.numer_coeffs == [1 / gamma, 1, -sympy.I * gamma]
    assert poly_fraction.denom_coeffs == [-1 / gamma, 1, sympy.I * gamma]


def test_controllable_canonical_form_gives_expected_state_space():
    poly_tf = PolyTransferFunc.from_transfer_function(tf_2dof, s)
    state_space = poly_tf.as_ccf_state_space

    assert state_space.num_inputs == state_space.num_outputs == 1
    assert state_space.num_dof == 2

    a_mat = sympy.Matrix([[0, 1], [sympy.I * gamma**2, gamma]])
    b_mat = sympy.Matrix([[0], [1]])
    c_mat = sympy.Matrix([[0, -2*gamma]])
    d_mat = sympy.Matrix([[-1]])

    assert state_space.a == a_mat
    assert state_space.b == b_mat
    assert state_space.c == c_mat
    assert state_space.d == d_mat

    assert state_space.siso_transfer_function.equals(tf_2dof)


def test_ccf_for_the_coupled_cavity_transfer_function(run_slow):
    """ This is to test using the work being done locally in coupled-cavity.nb """

    s = sympy.symbols("s")
    omega_s, Omega, tau_1, gamma_1, gamma_1_prime =\
        sympy.symbols("omega_s Omega tau_1 gamma_1 gamma_1'", real=True, positive=True)
    I = sympy.I

    prefactor = I * tau_1 * (gamma_1 - gamma_1_prime)

    tf = (-prefactor * Omega**2 + 2 * gamma_1 * Omega + prefactor * omega_s**2) /\
         (+prefactor * Omega**2 + 2 * gamma_1 * Omega - prefactor * omega_s**2)

    tf_normalized_freq = tf.subs(Omega, -I*omega_s*s)  # do we need to do this?

    prefactor = 2 * gamma_1 / (tau_1 * (gamma_1 - gamma_1_prime) * omega_s)

    # testing my re-arranging skills
    assert sympy.simplify(tf_normalized_freq - ((s**2 - prefactor * s + 1)/(-s**2 - prefactor*s - 1))) == 0

    poly_tf = PolyTransferFunc.from_transfer_function(tf_normalized_freq, s)

    ss = poly_tf.as_ccf_state_space

    assert ss.a == sympy.Matrix([[0, 1], [-1, -prefactor]])
    assert ss.b == sympy.Matrix([[0], [1]])
    assert ss.c == sympy.Matrix([[0, 2 * prefactor]])
    assert ss.d == sympy.Matrix([[-1]])

    # the simplifications make these tasks pretty slow
    if run_slow:
        assert sympy.simplify(ss.siso_transfer_function.subs(s, I * Omega / omega_s) - tf) == 0
        assert sympy.simplify(ss.siso_transfer_function - tf_normalized_freq) == 0
        assert PolyTransferFunc.from_transfer_function(ss.siso_transfer_function, s) == poly_tf

