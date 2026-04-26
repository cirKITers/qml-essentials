import os
from dataclasses import dataclass
from typing import Optional, List, Union, Dict, Callable, Tuple
import csv
import jax.numpy as jnp
import jax

from qml_essentials import operations as op
from qml_essentials import yaqsi as ys
from qml_essentials.utils import safe_random_split
from qml_essentials.tape import active_pulse_tape
from qml_essentials.unitary import UnitaryGates
import logging

log = logging.getLogger(__name__)


@dataclass
class DecompositionStep:
    """One step in a composite pulse gate decomposition.

    Attributes:
        gate: Child PulseParams object for this step.
        wire_fn: Wire selection - ``"all"``, ``"target"``, or ``"control"``.
        angle_fn: Maps parent angle(s) ``w`` to child angle.
            ``None`` means pass ``w`` through unchanged.
    """

    gate: "PulseParams"
    wire_fn: str = "all"
    angle_fn: Optional[Callable] = None


class PulseParams:
    """Container for hierarchical pulse parameters.

    Leaf nodes hold direct parameters; composite nodes hold a list of
    :class:`DecompositionStep` objects that describe how the gate is
    built from simpler gates.

    Attributes:
        name: Gate identifier (e.g. ``"RX"``, ``"H"``).
        decomposition: List of :class:`DecompositionStep` (composite only).
    """

    def __init__(
        self,
        name: str = "",
        params: Optional[jnp.ndarray] = None,
        decomposition: Optional[List[DecompositionStep]] = None,
    ) -> None:
        """
        Args:
            name: Gate name.
            params: Direct pulse parameters (leaf gates).
                Mutually exclusive with *decomposition*.
            decomposition: List of :class:`DecompositionStep` (composite gates).
                Mutually exclusive with *params*.
        """
        assert (params is None) != (
            decomposition is None
        ), "Exactly one of `params` or `decomposition` must be provided."

        self.decomposition = decomposition
        # Derive _pulse_obj for backward compat with childs/leafs/split_params
        self._pulse_obj = (
            [step.gate for step in decomposition] if decomposition else None
        )

        if params is not None:
            self._params = params

        self.name = name

    def __len__(self) -> int:
        """
        Get the total number of pulse parameters.

        For composite gates, returns the accumulated count from all children.

        Returns:
            int: Total number of pulse parameters.
        """
        return len(self.params)

    def __getitem__(self, idx: int) -> Union[float, jnp.ndarray]:
        """
        Access pulse parameter(s) by index.

        For leaf gates, returns the parameter at the given index.
        For composite gates, returns parameters of the child at the given index.

        Args:
            idx (int): Index to access.

        Returns:
            Union[float, jnp.ndarray]: Parameter value or child parameters.
        """
        if self.is_leaf:
            return self.params[idx]
        else:
            return self.childs[idx].params

    def __str__(self) -> str:
        """Return string representation (gate name)."""
        return self.name

    def __repr__(self) -> str:
        """Return repr string (gate name)."""
        return self.name

    @property
    def is_leaf(self) -> bool:
        """Check if this is a leaf node (direct parameters, no children)."""
        return self._pulse_obj is None

    @property
    def size(self) -> int:
        """Get the total parameter count (alias for __len__)."""
        return len(self)

    @property
    def leafs(self) -> List["PulseParams"]:
        """
        Get all leaf nodes in the hierarchy.

        Recursively collects all leaf PulseParams objects in the tree.

        Returns:
            List[PulseParams]: List of unique leaf nodes.
        """
        if self.is_leaf:
            return [self]

        leafs = []
        for obj in self._pulse_obj:
            leafs.extend(obj.leafs)

        return list(set(leafs))

    @property
    def childs(self) -> List["PulseParams"]:
        """
        Get direct children of this node.

        Returns:
            List[PulseParams]: List of child PulseParams objects, or empty list
                if this is a leaf node.
        """
        if self.is_leaf:
            return []

        return self._pulse_obj

    @property
    def shape(self) -> List[int]:
        """
        Get the shape of pulse parameters.

        For leaf nodes, returns list with parameter count.
        For composite nodes, returns nested list of child shapes.

        Returns:
            List[int]: Parameter shape specification.
        """
        if self.is_leaf:
            return [len(self.params)]

        shape = []
        for obj in self.childs:
            shape.append(*obj.shape())

        return shape

    @property
    def params(self) -> jnp.ndarray:
        """
        Get or compute pulse parameters.

        For leaf nodes, returns internal pulse parameters.
        For composite nodes, returns concatenated parameters from all children.

        Returns:
            jnp.ndarray: Pulse parameters array.
        """
        if self.is_leaf:
            return self._params

        params = self.split_params(params=None, leafs=False)

        return jnp.concatenate(params)

    @params.setter
    def params(self, value: jnp.ndarray) -> None:
        """
        Set pulse parameters.

        For leaf nodes, sets internal parameters directly.
        For composite nodes, distributes values across children.

        Args:
            value (jnp.ndarray): Pulse parameters to set.

        Raises:
            AssertionError: If value is not jnp.ndarray for leaf nodes.
        """
        if self.is_leaf:
            assert isinstance(value, jnp.ndarray), "params must be a jnp.ndarray"
            self._params = value
            return

        idx = 0
        for obj in self.childs:
            nidx = idx + obj.size
            obj.params = value[idx:nidx]
            idx = nidx

    @property
    def leaf_params(self) -> jnp.ndarray:
        """
        Get parameters from all leaf nodes.

        Returns:
            jnp.ndarray: Concatenated parameters from all leaf nodes.
        """
        if self.is_leaf:
            return self._params

        params = self.split_params(None, leafs=True)

        return jnp.concatenate(params)

    @leaf_params.setter
    def leaf_params(self, value: jnp.ndarray) -> None:
        """
        Set parameters for all leaf nodes.

        Args:
            value (jnp.ndarray): Parameters to distribute across leaf nodes.
        """
        if self.is_leaf:
            self._params = value
            return

        idx = 0
        for obj in self.leafs:
            nidx = idx + obj.size
            obj.params = value[idx:nidx]
            idx = nidx

    def split_params(
        self,
        params: Optional[jnp.ndarray] = None,
        leafs: bool = False,
    ) -> List[jnp.ndarray]:
        """
        Split parameters into sub-arrays for children or leaves.

        Args:
            params (Optional[jnp.ndarray]): Parameters to split. If None,
                uses internal parameters.
            leafs (bool): If True, splits across leaf nodes; if False,
                splits across direct children. Defaults to False.

        Returns:
            List[jnp.ndarray]: List of parameter arrays for children or leaves.
        """
        if params is None:
            if self.is_leaf:
                return self._params

            objs = self.leafs if leafs else self.childs
            s_params = []
            for obj in objs:
                s_params.append(obj.params)

            return s_params
        else:
            if self.is_leaf:
                return params

            objs = self.leafs if leafs else self.childs
            s_params = []
            idx = 0
            for obj in objs:
                nidx = idx + obj.size
                s_params.append(params[idx:nidx])
                idx = nidx

            return s_params


class PulseEnvelope:
    """Registry of pulse envelope shapes.

    Each envelope is a pure function ``(p, t, t_c) -> amplitude`` that
    computes the pulse envelope *without* carrier modulation.  The carrier
    ``cos(omega_c * t + phi_c)`` is applied separately in the coefficient
    functions built by :meth:`build_coeff_fns`.

    Attributes:
        REGISTRY: Mapping from envelope name to metadata dict containing
            ``fn`` (callable), ``n_envelope_params`` (int), and per-gate
            default parameter arrays.
    """

    @staticmethod
    def gaussian(p, t, t_c):
        """Gaussian envelope. ``p = [A, sigma]``."""
        A, sigma = p[0], p[1]
        return A * jnp.exp(-0.5 * ((t - t_c) / sigma) ** 2)

    @staticmethod
    def square(p, t, t_c):
        """Rectangular envelope. ``p = [A, width]``."""
        A, width = p[0], p[1]
        return A * (jnp.abs(t - t_c) <= width / 2)

    @staticmethod
    def cosine(p, t, t_c):
        """Raised cosine envelope. ``p = [A, width]``."""
        A, width = p[0], p[1]
        x = jnp.clip((t - t_c) / width, -0.5, 0.5)
        return A * jnp.cos(jnp.pi * x)

    @staticmethod
    def drag(p, t, t_c):
        """DRAG (Derivative Removal by Adiabatic Gate). ``p = [A, beta, sigma]``."""
        A, beta, sigma = p[0], p[1], p[2]
        g = A * jnp.exp(-0.5 * ((t - t_c) / sigma) ** 2)
        dg = g * (-(t - t_c) / sigma**2)
        return g + beta * dg

    @staticmethod
    def sech(p, t, t_c):
        """Hyperbolic secant envelope. ``p = [A, sigma]``."""
        A, sigma = p[0], p[1]
        return A / jnp.cosh((t - t_c) / sigma)

    # ``n_envelope_params`` counts only the envelope parameters (excluding
    # the evolution time ``t`` which is always the last element of the full
    # pulse parameter vector).
    REGISTRY = {
        "gaussian": {
            "fn": gaussian.__func__,
            "n_envelope_params": 2,
            "defaults": {
                "RX": jnp.array(
                    [0.3792368267874316,1.6156375956310989,3.014586015465094]
                ),
                "RY": jnp.array(
                    [0.3844185780214828,1.5961986560887964,2.9696202536794125]
                ),
            },
        },
        "square": {
            "fn": square.__func__,
            "n_envelope_params": 2,
            "defaults": {
                "RX": jnp.array([1.0306297147472894,0.992023715364146,0.9842684532062465]),
                "RY": jnp.array([1.016019943399879,0.9920256225480624,0.9694968585764263]),
            },
        },
        "cosine": {
            "fn": cosine.__func__,
            "n_envelope_params": 2,
            "defaults": {
                "RX": jnp.array([1.0, 1.0, 1.0]),
                "RY": jnp.array([1.0, 1.0, 1.0]),
            },
        },
        "drag": {
            "fn": drag.__func__,
            "n_envelope_params": 3,
            "defaults": {
                "RX": jnp.array([0.2783615044304307,0.49528556398943036,4.656436407163601,3.7843936100321023]),
                "RY": jnp.array([0.32202151885106306,0.4071648932963889,7.032768067573978,3.150322229118806]),
            },
        },
        "sech": {
            "fn": sech.__func__,
            "n_envelope_params": 2,
            "defaults": {
                "RX": jnp.array([1.0, 1.0, 1.0]),
                "RY": jnp.array([1.0, 1.0, 1.0]),
            },
        },
        "general": {
            "fn": None,
            "n_envelope_params": 0,
            "defaults": {
                "RZ": jnp.array([0.5]),
                "CZ": jnp.array([0.3182410939687649]),
            },
        },
    }

    @staticmethod
    def available() -> List[str]:
        """Return list of registered envelope names."""
        return list(PulseEnvelope.REGISTRY.keys())

    @staticmethod
    def get(name: str) -> dict:
        """Look up envelope metadata by name.

        Raises:
            ValueError: If *name* is not registered.
        """
        if name not in PulseEnvelope.REGISTRY:
            raise ValueError(
                f"Unknown pulse envelope '{name}'. "
                f"Available: {PulseEnvelope.available()}"
            )
        return PulseEnvelope.REGISTRY[name]

    @staticmethod
    def build_coeff_fns(
        envelope_fn: Callable,
        omega_c: float,
        omega_q: float,
        rwa: bool = False,
        frame: str = "lab",
    ) -> Tuple[Callable, Callable, Callable, Callable]:
        """Build the four interaction-picture coefficient functions.

        The lab-frame Hamiltonian is

            H(t,Π) = H_static + Σ_j S_j(t;Π) H_j ,
            S_j(t;Π) = E_j(t;Π) · cos(ω_c·t + φ_c) ,

        and the interaction-picture transform with respect to
        ``H_static = (ω_q/2)·Z`` produces

            H̃_j(t) = exp(+i H_static t) H_j exp(-i H_static t) ,
            H_I(t) = Σ_j S_j(t) H̃_j(t) .

        For a single qubit driven on X, ``H̃_X(t) = cos(ω_q·t) X
        − sin(ω_q·t) Y``, so

            H_I(t) = Ω(t) · cos(ω_c·t + φ) ·
                     [ cos(ω_q·t) · X  −  sin(ω_q·t) · Y ] .

        ``rwa=False`` (default) keeps **both** the slow and the fast
        counter-rotating components.

        ``rwa=True`` drops the fast (~2·ω_q on resonance) terms and
        keeps only the slow envelope, yielding the analytical RWA

            H_I^RWA(t) = (Ω(t)/2) · [ cos(φ) X + sin(φ) Y ] .

        For RX (``φ = 0``) this reduces to ``(Ω/2)·X``; for RY
        (``φ = +π/2``) to ``(Ω/2)·Y``.  This is dramatically cheaper to
        integrate (no fast oscillations → adaptive ODE solver takes
        large steps).  

        Each returned function has a unique ``__code__`` object so the
        yaqsi solver cache assigns separate compiled XLA programs per
        envelope shape and per (gate, component) pair.

        The rotation angle ``w`` is expected as the **last** element of
        the parameter array ``p`` (i.e. ``p[-1]``).  Envelope parameters
        occupy ``p[:-1]``.

        Args:
            envelope_fn: Pure envelope function ``(p, t, t_c) -> scalar``.
            omega_c: Carrier frequency.
            omega_q: Qubit frequency (interaction-picture rotation rate).
            rwa: When ``True``, return the RWA-truncated coefficients
                (no fast counter-rotating terms). Default ``False``
            frame: Algebraic representation of the exact (non-RWA)
                coefficients.  Mathematically equivalent options:

                * ``"lab"`` (default): the literal form
                  ``Ω(t) cos(ω_c t + φ) cos(ω_q t)`` (and the analogous
                  ``-sin`` term).  Two trig multiplications per call;
                  contains all four product frequencies implicitly.
                * ``"drive"``: applies the product-to-sum identity to
                  expose the slow ``(ω_c-ω_q)`` and fast ``(ω_c+ω_q)``
                  modes explicitly,
                  ``cos(ω_c t)cos(ω_q t) =
                  ½[cos((ω_c-ω_q)t) + cos((ω_c+ω_q)t)]``.  Algebraically
                  identical to ``"lab"`` (no RWA, no information lost).
                  Primary use: combined with the ``magnus2``/``magnus4``
                  yaqsi solvers, the explicit slow/fast decomposition
                  is sometimes numerically better-conditioned and lets
                  the user pick a fixed grid based on the slow
                  frequency alone (``Δ = |ω_c-ω_q|``) when the fast
                  ``(ω_c+ω_q)`` mode is well-resolved by the chosen
                  step.

                Ignored when ``rwa=True``.

        Returns:
            Tuple ``(coeff_RX_X, coeff_RX_Y, coeff_RY_X, coeff_RY_Y)``
            of coefficient functions for the X- and Y-components of the
            RX and RY interaction-picture Hamiltonians.
        """
        if frame not in ("lab", "drive"):
            raise ValueError(
                f"Unknown frame {frame!r}; expected 'lab' or 'drive'."
            )
        if rwa:
            # RWA-truncated coefficients (no carrier, no fast factors).
            # H_I^RWA = (Ω(t)/2) [cos(φ) X + sin(φ) Y]; we keep the
            # ``p[-1]`` rotation-angle scaling so the calling
            # ParametrizedHamiltonian shape is unchanged.
            half = jnp.asarray(0.5)

            def _coeff_RX_X(p, t):
                t_c = t / 2
                env = envelope_fn(p, t, t_c)
                return half * env * p[-1]

            def _coeff_RX_Y(p, t):  # Y component vanishes for RX (φ=0)
                t_c = t / 2
                env = envelope_fn(p, t, t_c)
                return jnp.zeros_like(half * env * p[-1])

            def _coeff_RY_X(p, t):  # X component vanishes for RY (φ=π/2)
                t_c = t / 2
                env = envelope_fn(p, t, t_c)
                return jnp.zeros_like(half * env * p[-1])

            def _coeff_RY_Y(p, t):
                t_c = t / 2
                env = envelope_fn(p, t, t_c)
                return half * env * p[-1]

            return _coeff_RX_X, _coeff_RX_Y, _coeff_RY_X, _coeff_RY_Y

        if frame == "drive":
            # Drive-frame: same exact dynamics, expressed via the
            # product-to-sum identities so the slow ``Δ = ω_c - ω_q``
            # and fast ``Σ = ω_c + ω_q`` modes appear explicitly.
            # Mathematically identical to the ``lab`` branch below.
            #
            # Identities used:
            #   cos(ω_c t) cos(ω_q t) = ½[cos(Δ t) + cos(Σ t)]
            #   cos(ω_c t) sin(ω_q t) = ½[sin(Σ t) − sin(Δ t)]
            #   −sin(ω_c t) cos(ω_q t) = −½[sin(Σ t) + sin(Δ t)]
            #   −sin(ω_c t) sin(ω_q t) = ½[cos(Σ t) − cos(Δ t)]
            # (RY uses cos(ω_c t + π/2) = −sin(ω_c t).)
            omega_d = omega_c - omega_q
            omega_s = omega_c + omega_q
            half = jnp.asarray(0.5)

            def _coeff_RX_X(p, t):
                t_c = t / 2
                env = envelope_fn(p, t, t_c)
                mod = half * (jnp.cos(omega_d * t) + jnp.cos(omega_s * t))
                return env * mod * p[-1]

            def _coeff_RX_Y(p, t):
                t_c = t / 2
                env = envelope_fn(p, t, t_c)
                mod = -half * (jnp.sin(omega_s * t) - jnp.sin(omega_d * t))
                return env * mod * p[-1]

            def _coeff_RY_X(p, t):
                t_c = t / 2
                env = envelope_fn(p, t, t_c)
                mod = -half * (jnp.sin(omega_s * t) + jnp.sin(omega_d * t))
                return env * mod * p[-1]

            def _coeff_RY_Y(p, t):
                t_c = t / 2
                env = envelope_fn(p, t, t_c)
                mod = -half * (jnp.cos(omega_s * t) - jnp.cos(omega_d * t))
                return env * mod * p[-1]

            return _coeff_RX_X, _coeff_RX_Y, _coeff_RY_X, _coeff_RY_Y


        # RX uses carrier phase phi = 0 so that after RWA
        #   cos(ω_q τ)·cos(ω_q τ)  averages to +1/2  → drives +X
        #   -cos(ω_q τ)·sin(ω_q τ) averages to  0    → Y cancels
        # giving H_I^RWA ≈ (Ω/2)·X → U ≈ exp(-iθ/2 X), matching op.RX.
        # The exact form below KEEPS the fast 2·ω_q components.
        def _coeff_RX_X(p, t):
            t_c = t / 2
            env = envelope_fn(p, t, t_c)
            carrier = jnp.cos(omega_c * t)
            return env * carrier * jnp.cos(omega_q * t) * p[-1]

        def _coeff_RX_Y(p, t):
            t_c = t / 2
            env = envelope_fn(p, t, t_c)
            carrier = jnp.cos(omega_c * t)
            return -env * carrier * jnp.sin(omega_q * t) * p[-1]

        # RY uses carrier phase phi = +pi/2 so the RWA component drives +Y.
        def _coeff_RY_X(p, t):
            t_c = t / 2
            env = envelope_fn(p, t, t_c)
            carrier = jnp.cos(omega_c * t + jnp.pi / 2)
            return env * carrier * jnp.cos(omega_q * t) * p[-1]

        def _coeff_RY_Y(p, t):
            t_c = t / 2
            env = envelope_fn(p, t, t_c)
            carrier = jnp.cos(omega_c * t + jnp.pi / 2)
            return -env * carrier * jnp.sin(omega_q * t) * p[-1]

        return _coeff_RX_X, _coeff_RX_Y, _coeff_RY_X, _coeff_RY_Y


class PulseInformation:
    """Stores pulse parameter counts and optimized pulse parameters.

    Call :meth:`set_envelope` to switch the active pulse shape.  This
    rebuilds all :class:`PulseParams` trees so that parameter counts
    and defaults match the selected envelope.
    """

    _envelope: str = "drag" #"gaussian"
    # Whether to apply the rotating-wave approximation when building the
    # interaction-picture coefficient functions.  
    # Default ``False`` (exact dynamics, no RWA).  
    # Setting to ``True`` drops the fast counter-rotating terms — 
    # much faster to integrate
    # See :meth:`PulseEnvelope.build_coeff_fns`.
    _rwa: bool = False
    # Algebraic representation of the (non-RWA) coefficients.  Either
    # ``"lab"`` or ``"drive"`` (product-to-sum decomposition).  
    # Mathematically equivalent — see :meth:`PulseEnvelope.build_coeff_fns`
    # when ``"drive"`` is numerically advantageous (mainly with the Magnus solvers).
    _frame: str = "lab"

    @classmethod
    def _build_leaf_gates(cls):
        """(Re-)create leaf PulseParams from the active envelope defaults."""
        defaults = PulseEnvelope.get(cls._envelope)["defaults"]
        general = PulseEnvelope.get("general")["defaults"]

        cls.RX = PulseParams(name="RX", params=defaults["RX"])
        cls.RY = PulseParams(name="RY", params=defaults["RY"])

        cls.RZ = PulseParams(name="RZ", params=general["RZ"])
        cls.CZ = PulseParams(name="CZ", params=general["CZ"])

    @classmethod
    def _build_composite_gates(cls):
        """(Re-)create composite PulseParams trees from current leaves."""
        cls.H = PulseParams(
            name="H",
            decomposition=[
                DecompositionStep(cls.RZ, "all", lambda w: jnp.pi),
                DecompositionStep(cls.RY, "all", lambda w: jnp.pi / 2),
            ],
        )
        cls.CX = PulseParams(
            name="CX",
            decomposition=[
                DecompositionStep(cls.H, "target", lambda w: 0.0),
                DecompositionStep(cls.CZ, "all", lambda w: 0.0),
                DecompositionStep(cls.H, "target", lambda w: 0.0),
            ],
        )
        cls.CY = PulseParams(
            name="CY",
            decomposition=[
                DecompositionStep(cls.RZ, "target", lambda w: -jnp.pi / 2),
                DecompositionStep(cls.CX, "all"),
                DecompositionStep(cls.RZ, "target", lambda w: jnp.pi / 2),
            ],
        )
        cls.CRX = PulseParams(
            name="CRX",
            decomposition=[
                DecompositionStep(cls.RZ, "target", lambda w: jnp.pi / 2),
                DecompositionStep(cls.RY, "target", lambda w: w / 2),
                DecompositionStep(cls.CX, "all", lambda w: 0.0),
                DecompositionStep(cls.RY, "target", lambda w: -w / 2),
                DecompositionStep(cls.CX, "all", lambda w: 0.0),
                DecompositionStep(cls.RZ, "target", lambda w: -jnp.pi / 2),
            ],
        )
        cls.CRY = PulseParams(
            name="CRY",
            decomposition=[
                DecompositionStep(cls.RY, "target", lambda w: w / 2),
                DecompositionStep(cls.CX, "all", lambda w: 0.0),
                DecompositionStep(cls.RY, "target", lambda w: -w / 2),
                DecompositionStep(cls.CX, "all", lambda w: 0.0),
            ],
        )
        cls.CRZ = PulseParams(
            name="CRZ",
            decomposition=[
                DecompositionStep(cls.RZ, "target", lambda w: w / 2),
                DecompositionStep(cls.CX, "all", lambda w: 0.0),
                DecompositionStep(cls.RZ, "target", lambda w: -w / 2),
                DecompositionStep(cls.CX, "all", lambda w: 0.0),
            ],
        )
        cls.Rot = PulseParams(
            name="Rot",
            decomposition=[
                DecompositionStep(cls.RZ, "all", lambda w: w[0]),
                DecompositionStep(cls.RY, "all", lambda w: w[1]),
                DecompositionStep(cls.RZ, "all", lambda w: w[2]),
            ],
        )
        cls.unique_gate_set = [cls.RX, cls.RY, cls.RZ, cls.CZ]

    @classmethod
    def set_envelope(
        cls,
        name: str,
        rwa: Optional[bool] = None,
        frame: Optional[str] = None,
    ) -> None:
        """Switch pulse envelope and rebuild all PulseParams trees.

        Also updates the coefficient functions used by :class:`PulseGates`.

        Args:
            name: One of :meth:`PulseEnvelope.available`.
            rwa: If given, also update the RWA flag.  If ``None`` (the
                default), the current value of ``cls._rwa`` is kept.
                See :meth:`PulseEnvelope.build_coeff_fns` for the
                physical meaning of the flag.
            frame: If given, also update the coefficient frame
                (``"lab"`` or ``"drive"``).  ``None`` keeps the current
                value of ``cls._frame``.  Ignored when ``rwa=True`` or
                when the existing RWA flag is on.
        """
        info = PulseEnvelope.get(name)  # validates name
        cls._envelope = name
        if rwa is not None:
            cls._rwa = bool(rwa)
        if frame is not None:
            if frame not in ("lab", "drive"):
                raise ValueError(
                    f"Unknown frame {frame!r}; expected 'lab' or 'drive'."
                )
            cls._frame = frame
        cls._build_leaf_gates()
        cls._build_composite_gates()

        # Rebuild interaction-picture coefficient functions on PulseGates.
        # Four functions: (RX_X, RX_Y, RY_X, RY_Y) — one per (gate, Pauli)
        # component of the proper interaction-picture drive Hamiltonian.
        rx_x, rx_y, ry_x, ry_y = PulseEnvelope.build_coeff_fns(
            info["fn"],
            PulseGates.omega_c,
            PulseGates.omega_q,
            rwa=cls._rwa,
            frame=cls._frame,
        )
        PulseGates._coeff_RX_X = staticmethod(rx_x)
        PulseGates._coeff_RX_Y = staticmethod(rx_y)
        PulseGates._coeff_RY_X = staticmethod(ry_x)
        PulseGates._coeff_RY_Y = staticmethod(ry_y)
        # Backward-compat aliases for older introspection (point at the
        # X-component which dominates RX, Y-component which dominates RY).
        PulseGates._coeff_Sx = staticmethod(rx_x)
        PulseGates._coeff_Sy = staticmethod(ry_y)
        PulseGates._active_envelope = name
        PulseGates._active_rwa = cls._rwa
        PulseGates._active_frame = cls._frame

        log.info(
            f"Pulse envelope set to '{name}' "
            f"(RWA {'on' if cls._rwa else 'off'}, frame={cls._frame})"
        )

    @classmethod
    def set_rwa(cls, rwa: bool) -> None:
        """Toggle the rotating-wave approximation for pulse coefficients.

        Rebuilds the coefficient functions for the currently active
        envelope so the change takes effect immediately.  Default is
        ``False`` (exact interaction picture).
        See :meth:`PulseEnvelope.build_coeff_fns` for details
        """
        cls.set_envelope(cls._envelope, rwa=bool(rwa))

    @classmethod
    def get_envelope(cls) -> str:
        """Return the name of the active pulse envelope."""
        return cls._envelope

    @classmethod
    def get_rwa(cls) -> bool:
        """Return whether the RWA flag is currently active."""
        return cls._rwa

    @classmethod
    def set_frame(cls, frame: str) -> None:
        """Switch the algebraic representation of the (non-RWA) coefficients.

        ``"lab"`` (default) and ``"drive"`` are mathematically
        identical (no information lost, no RWA applied) — see
        :meth:`PulseEnvelope.build_coeff_fns` for when ``"drive"`` is
        useful.  Rebuilds the coefficient functions for the currently
        active envelope so the change takes effect immediately.
        """
        cls.set_envelope(cls._envelope, frame=str(frame))

    @classmethod
    def get_frame(cls) -> str:
        """Return the active coefficient frame (``"lab"`` or ``"drive"``)."""
        return cls._frame

    @staticmethod
    def gate_by_name(gate):
        if isinstance(gate, str):
            return getattr(PulseInformation, gate, None)
        else:
            return getattr(PulseInformation, gate.__name__, None)

    @staticmethod
    def num_params(gate):
        return len(PulseInformation.gate_by_name(gate))

    @staticmethod
    def update_params(path=f"{os.getcwd()}/qml_essentials/qoc_results.csv"):
        if os.path.isfile(path):
            log.info(f"Loading optimized pulses from {path}")
            with open(path, "r") as f:
                reader = csv.reader(f)

                for row in reader:
                    log.debug(
                        f"Loading optimized pulses for {row[0]}\
                            (Fidelity: {float(row[1]):.5f}): {row[2:]}"
                    )
                    PulseInformation.OPTIMIZED_PULSES[row[0]] = jnp.array(
                        [float(x) for x in row[2:]]
                    )
        else:
            log.error(f"No optimized pulses found at {path}")

    @staticmethod
    def shuffle_params(random_key):
        log.info(
            f"Shuffling optimized pulses with random key {random_key}\
              of gates {PulseInformation.unique_gate_set}"
        )
        for gate in PulseInformation.unique_gate_set:
            random_key, sub_key = safe_random_split(random_key)
            gate.params = jax.random.uniform(sub_key, (len(gate),))


# Initialise PulseInformation with default (gaussian) envelope
PulseInformation._build_leaf_gates()
PulseInformation._build_composite_gates()


class PulseGates:
    """Pulse-level implementations of quantum gates.

    Implements quantum gates using time-dependent Hamiltonians and pulse
    sequences, following the approach from https://doi.org/10.5445/IR/1000184129.
    The active pulse envelope is selected via
    :meth:`PulseInformation.set_envelope`.

    Attributes:
        omega_q: Qubit frequency (10π).
        omega_c: Carrier frequency (10π).
        _active_envelope: Name of the currently active envelope shape.
    """

    # NOTE: Implementation of S, RX, RY, RZ, CZ, CNOT/CX and H pulse level
    #   gates closely follow https://doi.org/10.5445/IR/1000184129
    omega_q = 10 * jnp.pi
    omega_c = 10 * jnp.pi

    X = jnp.array([[0, 1], [1, 0]])
    Y = jnp.array([[0, -1j], [1j, 0]])
    Z = jnp.array([[1, 0], [0, -1]])

    Id = jnp.eye(2, dtype=jnp.complex64)

    _H_CZ = (jnp.pi / 4) * (
        jnp.kron(Id, Id) - jnp.kron(Z, Id) - jnp.kron(Id, Z) + jnp.kron(Z, Z)
    )

    _H_corr = jnp.pi / 2 * jnp.eye(2, dtype=jnp.complex64)

    _active_envelope: str = "gaussian"
    # Mirrors :attr:`PulseInformation._rwa`; kept here for introspection
    # of which coefficient regime the active ``_coeff_*`` functions
    # implement.  Updated by :meth:`PulseInformation.set_envelope` /
    # :meth:`PulseInformation.set_rwa`.
    _active_rwa: bool = False
    _active_frame: str = "lab"

    # Default coefficient functions for the gaussian envelope; the active
    # envelope's `set_envelope` will overwrite these.  Each gate uses two
    # coefficients (X- and Y-component of the proper interaction-picture
    # drive Hamiltonian).

    @staticmethod
    def _coeff_RX_X(p, t):
        """RX coefficient for the X term (gaussian default)."""
        t_c = t / 2
        env = PulseEnvelope.gaussian(p, t, t_c)
        carrier = jnp.cos(PulseGates.omega_c * t)
        return env * carrier * jnp.cos(PulseGates.omega_q * t) * p[-1]

    @staticmethod
    def _coeff_RX_Y(p, t):
        """RX coefficient for the Y term (gaussian default)."""
        t_c = t / 2
        env = PulseEnvelope.gaussian(p, t, t_c)
        carrier = jnp.cos(PulseGates.omega_c * t)
        return -env * carrier * jnp.sin(PulseGates.omega_q * t) * p[-1]

    @staticmethod
    def _coeff_RY_X(p, t):
        """RY coefficient for the X term (gaussian default)."""
        t_c = t / 2
        env = PulseEnvelope.gaussian(p, t, t_c)
        carrier = jnp.cos(PulseGates.omega_c * t + jnp.pi / 2)
        return env * carrier * jnp.cos(PulseGates.omega_q * t) * p[-1]

    @staticmethod
    def _coeff_RY_Y(p, t):
        """RY coefficient for the Y term (gaussian default)."""
        t_c = t / 2
        env = PulseEnvelope.gaussian(p, t, t_c)
        carrier = jnp.cos(PulseGates.omega_c * t + jnp.pi / 2)
        return -env * carrier * jnp.sin(PulseGates.omega_q * t) * p[-1]

    # Backward-compat aliases (resolve to the dominant component of each gate).
    _coeff_Sx = _coeff_RX_X
    _coeff_Sy = _coeff_RY_Y

    @staticmethod
    def _coeff_Sz(p, t):
        """Coefficient function for RZ pulse: p * w."""
        return p[0] * p[1]

    @staticmethod
    def _coeff_Sc(p, t):
        """Constant coefficient for H correction phase."""
        return -1.0

    @staticmethod
    def _coeff_Scz(p, t):
        """Coefficient function for CZ pulse."""
        return p * jnp.pi

    @staticmethod
    def _record_pulse_event(gate_name, w, wires, pulse_params, parent=None):
        """Append a PulseEvent to the active pulse tape if recording.

        This is called from leaf gate methods (RX, RY, RZ, CZ) so that
        :func:`~qml_essentials.tape.pulse_recording` can collect events
        without the caller needing to know about the tape.
        """
        ptape = active_pulse_tape()
        if ptape is None:
            return

        from qml_essentials.drawing import PulseEvent, LEAF_META

        meta = LEAF_META.get(gate_name, {})
        wires_list = [wires] if isinstance(wires, int) else list(wires)

        if meta.get("physical", False):
            info = PulseEnvelope.get(PulseInformation.get_envelope())
            pp = PulseInformation.gate_by_name(gate_name).split_params(pulse_params)
            env_p = pp[:-1]
            dur = float(pp[-1])
            ptape.append(
                PulseEvent(
                    gate=gate_name,
                    wires=wires_list,
                    envelope_fn=info["fn"],
                    envelope_params=jnp.array(env_p),
                    w=float(w),
                    duration=dur,
                    carrier_phase=meta["carrier_phase"],
                    parent=parent,
                )
            )
        else:
            pp = PulseInformation.gate_by_name(gate_name).split_params(pulse_params)
            ptape.append(
                PulseEvent(
                    gate=gate_name,
                    wires=wires_list,
                    envelope_fn=None,
                    envelope_params=jnp.ravel(jnp.asarray(pp)),
                    w=float(w) if not isinstance(w, list) else 0.0,
                    duration=1.0,
                    carrier_phase=0.0,
                    parent=parent,
                )
            )

    @staticmethod
    def Rot(
        phi: float,
        theta: float,
        omega: float,
        wires: Union[int, List[int]],
        pulse_params: Optional[jnp.ndarray] = None,
        noise_params: Optional[Dict[str, float]] = None,
        random_key: Optional[jax.random.PRNGKey] = None,
    ) -> None:
        """
        Apply general rotation via decomposition: RZ(phi) · RY(theta) · RZ(omega).

        Args:
            phi (float): First rotation angle.
            theta (float): Second rotation angle.
            omega (float): Third rotation angle.
            wires (Union[int, List[int]]): Qubit index or indices to apply rotation to.
            pulse_params (Optional[jnp.ndarray]): Pulse parameters for the
                composing gates. If None, uses optimized parameters.
            noise_params (Optional[Dict[str, float]]): Noise parameters dictionary.
            random_key (Optional[jax.random.PRNGKey]): JAX random key for compatibility

        Returns:
            None: Gates are applied in-place to the circuit.
        """
        if noise_params is not None and "GateError" in noise_params:
            phi, random_key = UnitaryGates.GateError(phi, noise_params, random_key)
            theta, random_key = UnitaryGates.GateError(theta, noise_params, random_key)
            omega, random_key = UnitaryGates.GateError(omega, noise_params, random_key)
        PulseGates._execute_composite("Rot", [phi, theta, omega], wires, pulse_params)
        UnitaryGates.Noise(wires, noise_params)

    @staticmethod
    def PauliRot(
        pauli: str,
        theta: float,
        wires: Union[int, List[int]],
        pulse_params: Optional[jnp.ndarray] = None,
        noise_params: Optional[Dict[str, float]] = None,
        random_key: Optional[jax.random.PRNGKey] = None,
    ) -> None:
        """Not implemented as a PulseGate."""
        raise NotImplementedError("PauliRot gate is not implemented as PulseGate")

    @staticmethod
    def RX(
        w: float,
        wires: Union[int, List[int]],
        pulse_params: Optional[jnp.ndarray] = None,
        noise_params: Optional[Dict[str, float]] = None,
        random_key: Optional[jax.random.PRNGKey] = None,
    ) -> None:
        """Apply X-axis rotation using the active pulse envelope.

        Args:
            w: Rotation angle in radians.
            wires: Qubit index or indices.
            pulse_params: Envelope parameters ``[env_0, ..., env_n, t]``.
                If ``None``, uses optimized defaults.
            noise_params (Optional[Dict[str, float]]): Noise parameters dictionary.
            random_key (Optional[jax.random.PRNGKey]): JAX random key for compatibility
        """
        pulse_params = PulseInformation.RX.split_params(pulse_params)

        PulseGates._record_pulse_event("RX", w, wires, pulse_params)
        t = pulse_params[-1]

        # Proper interaction-picture drive Hamiltonian for RX:
        #   H_I(τ) = Ω(τ)·cos(ω_c·τ) · [ cos(ω_q·τ)·X − sin(ω_q·τ)·Y ]
        # which on resonance averages (RWA) to +(Ω/2)·X while the
        # 2·ω_q counter-rotating part oscillates and cancels.
        H_X = op.Hermitian(PulseGates.X, wires=wires, record=False)
        H_Y = op.Hermitian(PulseGates.Y, wires=wires, record=False)
        H_eff = (
            PulseGates._coeff_RX_X * H_X
            + PulseGates._coeff_RX_Y * H_Y
        )

        # Pack: [envelope_params..., w] - evolution time is the last element
        # of pulse_params (pulse_params[-1]).
        w, random_key = UnitaryGates.GateError(w, noise_params, random_key)
        env_params = jnp.array([*pulse_params[:-1], w])
        # Both terms share the same parameter array.
        ys.evolve(H_eff, name="RX")([env_params, env_params], t)
        UnitaryGates.Noise(wires, noise_params)

    @staticmethod
    def RY(
        w: float,
        wires: Union[int, List[int]],
        pulse_params: Optional[jnp.ndarray] = None,
        noise_params: Optional[Dict[str, float]] = None,
        random_key: Optional[jax.random.PRNGKey] = None,
    ) -> None:
        """Apply Y-axis rotation using the active pulse envelope.

        Args:
            w: Rotation angle in radians.
            wires: Qubit index or indices.
            pulse_params: Envelope parameters ``[env_0, ..., env_n, t]``.
                If ``None``, uses optimized defaults.
            noise_params (Optional[Dict[str, float]]): Noise parameters dictionary.
            random_key (Optional[jax.random.PRNGKey]): JAX random key for compatibility
        """
        pulse_params = PulseInformation.RY.split_params(pulse_params)

        PulseGates._record_pulse_event("RY", w, wires, pulse_params)
        t = pulse_params[-1]

        # See NOTE in RX: same proper interaction-picture form, with
        # carrier phase ϕ = +π/2 so the slow RWA component drives +Y.
        H_X = op.Hermitian(PulseGates.X, wires=wires, record=False)
        H_Y = op.Hermitian(PulseGates.Y, wires=wires, record=False)
        H_eff = (
            PulseGates._coeff_RY_X * H_X
            + PulseGates._coeff_RY_Y * H_Y
        )

        # Pack w into the params so the coefficient function doesn't need
        # a closure - this enables JIT solver cache sharing across all RY calls.
        w, random_key = UnitaryGates.GateError(w, noise_params, random_key)
        env_params = jnp.array([*pulse_params[:-1], w])
        ys.evolve(H_eff, name="RY")([env_params, env_params], t)
        UnitaryGates.Noise(wires, noise_params)

    @staticmethod
    def RZ(
        w: float,
        wires: Union[int, List[int]],
        pulse_params: Optional[float] = None,
        noise_params: Optional[Dict[str, float]] = None,
        random_key: Optional[jax.random.PRNGKey] = None,
    ) -> None:
        """
        Apply Z-axis rotation using pulse-level implementation.

        Implements RZ rotation using virtual Z rotations (phase tracking)
        without physical pulse application.

        Args:
            w (float): Rotation angle in radians.
            wires (Union[int, List[int]]): Qubit index or indices to apply rotation to.
            pulse_params (Optional[float]): Duration parameter for the pulse.
                Rotation angle = w * 2 * pulse_params. Defaults to 0.5 if None.
            noise_params (Optional[Dict[str, float]]): Noise parameters dictionary.
            random_key (Optional[jax.random.PRNGKey]): JAX random key for compatibility

        Returns:
            None: Gate is applied in-place to the circuit.
        """
        pulse_params = PulseInformation.RZ.split_params(pulse_params)

        PulseGates._record_pulse_event("RZ", w, wires, pulse_params)

        _H = op.Hermitian(PulseGates.Z, wires=wires, record=False)
        H_eff = PulseGates._coeff_Sz * _H

        # Pack w into the params so the coefficient function doesn't need
        # a closure - [pulse_param_scalar, w] enables JIT solver cache sharing.
        # pulse_params may be a 1-element array or scalar; ravel + index to
        # ensure a scalar for concatenation.
        w, random_key = UnitaryGates.GateError(w, noise_params, random_key)
        pp_scalar = jnp.ravel(jnp.asarray(pulse_params))[0]
        ys.evolve(H_eff, name="RZ")([jnp.array([pp_scalar, w])], 1)

        UnitaryGates.Noise(wires, noise_params)

    @staticmethod
    def _resolve_wires(wire_fn, wires):
        """Resolve a wire selector string to actual wire(s).

        Args:
            wire_fn: ``"all"``, ``"target"``, or ``"control"``.
            wires: Parent gate's wire(s) (int or list).

        Returns:
            Wire(s) for the child gate.
        """
        wires_list = [wires] if isinstance(wires, int) else list(wires)
        if wire_fn == "all":
            return wires if len(wires_list) > 1 else wires_list[0]
        if wire_fn == "target":
            return wires_list[-1] if len(wires_list) > 1 else wires_list[0]
        if wire_fn == "control":
            return wires_list[0]
        raise ValueError(f"Unknown wire_fn: {wire_fn!r}")

    @staticmethod
    def _execute_composite(gate_name, w, wires, pulse_params=None):
        """Execute a composite gate by walking its decomposition.

        Reads the :class:`DecompositionStep` list from
        :class:`PulseInformation` and dispatches each step to the
        appropriate ``PulseGates`` method.

        Args:
            gate_name: Gate name (e.g. ``"H"``, ``"CX"``).
            w: Rotation angle(s) passed to the parent gate.
            wires: Wire(s) of the parent gate.
            pulse_params: Optional pulse parameters (split across children).
        """
        pp_obj = PulseInformation.gate_by_name(gate_name)
        parts = pp_obj.split_params(pulse_params)

        for step, child_params in zip(pp_obj.decomposition, parts):
            child_wires = PulseGates._resolve_wires(step.wire_fn, wires)
            child_w = step.angle_fn(w) if step.angle_fn is not None else w
            child_gate = getattr(PulseGates, step.gate.name)

            # Leaf gates that take a rotation angle
            if step.gate.name in ("RX", "RY", "RZ"):
                child_gate(child_w, wires=child_wires, pulse_params=child_params)
            # Leaf gates without a rotation angle
            elif step.gate.name in ("CZ",):
                child_gate(wires=child_wires, pulse_params=child_params)
            # Composite gates with a rotation angle (CRX, CRY, CRZ, Rot, ...)
            elif step.gate.name in ("Rot",):
                # Rot expects (phi, theta, omega, wires, ...)
                child_gate(*child_w, wires=child_wires, pulse_params=child_params)
            elif step.gate.decomposition is not None and step.gate.name in (
                "CRX",
                "CRY",
                "CRZ",
            ):
                child_gate(child_w, wires=child_wires, pulse_params=child_params)
            # Other composite gates (H, CX, CY, ...)
            else:
                child_gate(wires=child_wires, pulse_params=child_params)

    @staticmethod
    def H(
        wires: Union[int, List[int]],
        pulse_params: Optional[jnp.ndarray] = None,
        noise_params: Optional[Dict[str, float]] = None,
        random_key: Optional[jax.random.PRNGKey] = None,
    ) -> None:
        """Apply Hadamard gate using pulse decomposition.

        Decomposes as RZ(π) · RY(π/2) followed by a correction phase.

        Args:
            wires (Union[int, List[int]]): Qubit index or indices.
            noise_params (Optional[Dict[str, float]]): Noise parameters dictionary.
            random_key (Optional[jax.random.PRNGKey]): JAX random key for compatibility
                (not used in this gate).
        """
        PulseGates._execute_composite("H", 0.0, wires, pulse_params)

        # Correction phase unique to the H gate
        _H = op.Hermitian(PulseGates._H_corr, wires=wires, record=False)
        H_corr = PulseGates._coeff_Sc * _H
        ys.evolve(H_corr, name="H")([0], 1)
        UnitaryGates.Noise(wires, noise_params)

    @staticmethod
    def CX(
        wires: List[int],
        pulse_params: Optional[jnp.ndarray] = None,
        noise_params: Optional[Dict[str, float]] = None,
        random_key: Optional[jax.random.PRNGKey] = None,
    ) -> None:
        """Apply CNOT gate via decomposition: H(target) · CZ · H(target).

        Args:
            wires (List[int]): Control and target qubit indices [control, target].
            pulse_params (Optional[jnp.ndarray]): Pulse parameters for the
                composing gates. If None, uses optimized parameters.
            noise_params (Optional[Dict[str, float]]): Noise parameters dictionary.
            random_key (Optional[jax.random.PRNGKey]): JAX random key for compatibility
                (not used in this gate).

        Returns:
            None: Gate is applied in-place to the circuit.
        """
        PulseGates._execute_composite("CX", 0.0, wires, pulse_params)
        UnitaryGates.Noise(wires, noise_params)

    @staticmethod
    def CY(
        wires: List[int],
        pulse_params: Optional[jnp.ndarray] = None,
        noise_params: Optional[Dict[str, float]] = None,
        random_key: Optional[jax.random.PRNGKey] = None,
    ) -> None:
        """Apply controlled-Y via decomposition.

        Args:
            wires (List[int]): Control and target qubit indices [control, target].
            pulse_params (Optional[jnp.ndarray]): Pulse parameters for the
                composing gates. If None, uses optimized parameters.
            noise_params (Optional[Dict[str, float]]): Noise parameters dictionary.
            random_key (Optional[jax.random.PRNGKey]): JAX random key for compatibility
                (not used in this gate).

        """
        PulseGates._execute_composite("CY", 0.0, wires, pulse_params)
        UnitaryGates.Noise(wires, noise_params)

    @staticmethod
    def CZ(
        wires: List[int],
        pulse_params: Optional[float] = None,
        noise_params: Optional[Dict[str, float]] = None,
        random_key: Optional[jax.random.PRNGKey] = None,
    ) -> None:
        """Apply controlled-Z using ZZ coupling Hamiltonian.

        Args:
            wires (List[int]): Control and target qubit indices.
            pulse_params (Optional[float]): Time or duration parameter for
                the pulse evolution. If None, uses optimized value.
            noise_params (Optional[Dict[str, float]]): Noise parameters dictionary.
            random_key (Optional[jax.random.PRNGKey]): JAX random key for compatibility
                (not used in this gate).

        """
        if pulse_params is None:
            pulse_params = PulseInformation.CZ.params

        PulseGates._record_pulse_event("CZ", 0.0, wires, pulse_params)

        _H = op.Hermitian(PulseGates._H_CZ, wires=wires, record=False)
        H_eff = PulseGates._coeff_Scz * _H
        ys.evolve(H_eff, name="CZ")([pulse_params], 1)
        UnitaryGates.Noise(wires, noise_params)

    @staticmethod
    def CRX(
        w: float,
        wires: List[int],
        pulse_params: Optional[jnp.ndarray] = None,
        noise_params: Optional[Dict[str, float]] = None,
        random_key: Optional[jax.random.PRNGKey] = None,
    ) -> None:
        """Apply controlled-RX via decomposition.

        Args:
            w (float): Rotation angle in radians.
            wires (List[int]): Control and target qubit indices [control, target].
            pulse_params (Optional[jnp.ndarray]): Pulse parameters for the
                composing gates. If None, uses optimized parameters.
            noise_params (Optional[Dict[str, float]]): Noise parameters dictionary.
            random_key (Optional[jax.random.PRNGKey]): JAX random key for compatibility
                (not used in this gate).
        """
        PulseGates._execute_composite("CRX", w, wires, pulse_params)
        UnitaryGates.Noise(wires, noise_params)

    @staticmethod
    def CRY(
        w: float,
        wires: List[int],
        pulse_params: Optional[jnp.ndarray] = None,
        noise_params: Optional[Dict[str, float]] = None,
        random_key: Optional[jax.random.PRNGKey] = None,
    ) -> None:
        """Apply controlled-RY via decomposition.

        Args:
            w (float): Rotation angle in radians.
            wires (List[int]): Control and target qubit indices [control, target].
            pulse_params (Optional[jnp.ndarray]): Pulse parameters for the
                composing gates. If None, uses optimized parameters.
            noise_params (Optional[Dict[str, float]]): Noise parameters dictionary.
            random_key (Optional[jax.random.PRNGKey]): JAX random key for compatibility
        """
        w, random_key = UnitaryGates.GateError(w, noise_params, random_key)
        PulseGates._execute_composite("CRY", w, wires, pulse_params)
        UnitaryGates.Noise(wires, noise_params)

    @staticmethod
    def CRZ(
        w: float,
        wires: List[int],
        pulse_params: Optional[jnp.ndarray] = None,
        noise_params: Optional[Dict[str, float]] = None,
        random_key: Optional[jax.random.PRNGKey] = None,
    ) -> None:
        """Apply controlled-RZ via decomposition.

        Args:
            w (float): Rotation angle in radians.
            wires (List[int]): Control and target qubit indices [control, target].
            pulse_params (Optional[jnp.ndarray]): Pulse parameters for the
                composing gates. If None, uses optimized parameters.
            noise_params (Optional[Dict[str, float]]): Noise parameters dictionary.
            random_key (Optional[jax.random.PRNGKey]): JAX random key for compatibility
        """
        w, random_key = UnitaryGates.GateError(w, noise_params, random_key)
        PulseGates._execute_composite("CRZ", w, wires, pulse_params)
        UnitaryGates.Noise(wires, noise_params)


class PulseParamManager:
    def __init__(self, pulse_params: jnp.ndarray):
        self.pulse_params = pulse_params
        self.idx = 0

    def get(self, n: int):
        """Return the next n parameters and advance the cursor."""
        if self.idx + n > len(self.pulse_params):
            raise ValueError("Not enough pulse parameters left for this gate")
        # TODO: we squeeze here to get rid of any extra hidden dimension
        params = self.pulse_params[self.idx : self.idx + n].squeeze()
        self.idx += n
        return params
