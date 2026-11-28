"""Skip CVXPY's attribute check when a value is assigned to an attribute-free leaf.

Every MPC step assigns ~100 parameter values (``Parameter.value = ...``) and CVXPY writes back every variable value
after a solve; each assignment runs ``Leaf._validate_value``, which projects the value onto the leaf's attribute set
(nonneg, bounds, PSD, ...) and compares with ``np.allclose`` -- about a tenth of an episode's wall time. For a leaf with
no attributes, no bounds, no sparsity and a real (non-complex) value the projection is the value itself, so the check
can only pass and the stored value is ``intf.convert(val)`` either way. This fast path stores exactly that value after
the same shape check; every other case goes through CVXPY's own validation. The stored values, hence every solve, are
bitwise unchanged (``tests/test_ecc.py``).
"""
from __future__ import annotations

import numpy as np
from cvxpy import interface as intf
from cvxpy.expressions.leaf import Leaf

_ORIGINAL = Leaf._validate_value


def _validate_value(self, val, sparse_path=False):
    if val is not None and not sparse_path and self.num_attributes == 0 and self.bounds is None \
            and self.sparse_idx is None:
        converted = intf.convert(val)
        if type(converted) is np.ndarray and converted.dtype.kind in 'fiub':
            if intf.shape(converted) != self.shape:
                raise ValueError("Invalid dimensions %s for %s value." % (intf.shape(converted), self.__class__.__name__))
            return converted
    return _ORIGINAL(self, val, sparse_path)


Leaf._validate_value = _validate_value
