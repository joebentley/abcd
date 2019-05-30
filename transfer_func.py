import sympy
import system


class PolyTransferFunc:
    """ Represents a transfer function as a fraction of two polynomials. """

    def __init__(self, numer_poly: sympy.Poly, denom_poly: sympy.Poly):
        """ Construct a new transfer function from the given polynomials. """
        if not (isinstance(numer_poly, sympy.Poly) and isinstance(denom_poly, sympy.Poly)):
            raise TypeError("arguments must be sympy Poly objects")

        self.numer = numer_poly
        self.denom = denom_poly

    @classmethod
    def from_transfer_function(cls, expr: sympy.Expr, s: sympy.Symbol):
        """
        Construct a transfer function from the given expression with s being a normalized frequency variable.
        :param expr: the expr to be turned into a transfer function. Should be the ratio of two polynomials.
        :param s: the normalized frequency variable used in expr
        :return:
        """

        if not isinstance(expr, sympy.Expr):
            raise TypeError("expr must be a sympy expression")
        if not isinstance(s, sympy.Symbol):
            raise TypeError("s must be the sympy Symbol used as the complex frequency")

        numer, denom = sympy.fraction(expr)
        numer_poly, denom_poly = sympy.poly(numer, gens=s), sympy.poly(denom, gens=s)
        return cls(numer_poly, denom_poly)

    @property
    def numer_coeffs(self):
        """ Return the coefficients of all the terms on the numerator, highest order first. """
        return self.numer.all_coeffs()

    @property
    def denom_coeffs(self):
        """ Return the coefficients of all the terms on the denominator, highest order first. """
        return self.denom.all_coeffs()

    @property
    def as_ccf_state_space(self) -> system.System:
        """
        Convert to a SISO state-space representation in Controllable Canonical Form
        TODO: support multiple transfer functions, or transfer functions between two ladder operators
        :return: a time-domain System which represents the transfer function
        """
        n_dof = max(self.numer.degree(), self.denom.degree())

        # assume single-input single-output (SISO)

        b = self.numer_coeffs  # b[i] are the coefficients of the terms on the numerator
        a = self.denom_coeffs  # a[i] are the coefficients of the terms on the denominator

        # pad from left with zeroes
        a = [0] * (n_dof - len(a)) + a
        b = [0] * (n_dof - len(b)) + b

        # normalize all elements w.r.t a[0] if it is non-zero
        if a[0] != 0:
            b = list(map(lambda bn: bn / a[0], b))
            a = list(map(lambda an: an / a[0], a))
            assert a[0] == 1

        # reverse so that first element is the lowest degree coefficient
        a = list(reversed(a))
        b = list(reversed(b))

        # construct dynamics matrix
        a_mat = sympy.zeros(n_dof)
        for i in range(n_dof):
            if i > 0:
                a_mat[i - 1, i] = 1

            a_mat[n_dof - 1, i] = -a[i]

        # construct input coupling matrix (always the same)
        b_mat = sympy.zeros(n_dof, 1)
        b_mat[n_dof - 1, 0] = 1

        # construct output coupling matrix
        b0 = b[len(b) - 1]
        c_mat = sympy.zeros(1, n_dof)
        for i in range(n_dof):
            c_mat[0, i] = a[i] - b[i] * b0

        # construct direct-feed "matrix"
        d_mat = b0

        # simplify all the results (TODO: this is obviously slow)
        a_mat = sympy.simplify(a_mat)
        b_mat = sympy.simplify(b_mat)
        c_mat = sympy.simplify(c_mat)
        d_mat = sympy.simplify(d_mat)

        return system.System(a_mat, b_mat, c_mat, d_mat)

    @property
    def expr(self) -> sympy.Expr:
        """ Divide and return the numerator and denominator as a single expression """
        return self.numer / self.denom

    def __str__(self):
        return str(self.expr)

    def equals(self, other):
        """ Apply sympy.simplify to work out whether the two expressions are equal """
        return sympy.simplify(self.expr - other.expr) == 0

    def __eq__(self, other):
        return self.equals(other)
