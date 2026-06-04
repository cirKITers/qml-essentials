from __future__ import annotations
import sys
import math
import warnings
import itertools
from collections import defaultdict
import jax.numpy as jnp
from jax import random
import numpy as np
from scipy.stats import rankdata
from functools import reduce, lru_cache
from typing import List, Tuple, Optional, Any, Dict, Union

from qml_essentials.model import Model
from qml_essentials.pauli import PauliCircuit
from qml_essentials.operations import PauliWord

import logging

log = logging.getLogger(__name__)


class Coefficients:
    @classmethod
    def get_spectrum(
        cls,
        model: Model,
        mfs: int = 1,
        mts: int = 1,
        shift=False,
        trim=False,
        numerical_cap: Optional[float] = -1,
        **kwargs,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Extracts the coefficients of a given model using a FFT (jnp-fft).

        Note that the coefficients are complex numbers, but the imaginary part
        of the coefficients should be very close to zero, since the expectation
        values of the Pauli operators are real numbers.

        It can perform oversampling in both the frequency and time domain
        using the `mfs` and `mts` arguments.

        Args:
            model (Model): The model to sample.
            mfs (int): Multiplicator for the highest frequency. Default is 2.
            mts (int): Multiplicator for the number of time samples. Default is 1.
            shift (bool): Whether to apply jnp-fftshift. Default is False.
            trim (bool): Whether to remove the Nyquist frequency if spectrum is even.
                Default is False.
            numerical_cap (Optional[float]): Numerical cap for the coefficients.
            kwargs (Any): Additional keyword arguments for the model function.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: Tuple containing the coefficients
            and frequencies.
        """
        kwargs.setdefault("force_mean", True)
        kwargs.setdefault("execution_type", "expval")

        coeffs, freqs = cls._fourier_transform(model, mfs=mfs, mts=mts, **kwargs)

        if not jnp.isclose(jnp.sum(coeffs).imag, 0.0, rtol=1.0e-5):
            raise ValueError(
                f"Spectrum is not real. Imaginary part of coefficients is:\
                {jnp.sum(coeffs).imag}"
            )

        if trim:
            for ax in range(model.n_input_feat):
                if coeffs.shape[ax] % 2 == 0:
                    coeffs = np.delete(coeffs, len(coeffs) // 2, axis=ax)
                    freqs = [np.delete(freq, len(freq) // 2, axis=ax) for freq in freqs]

        if shift:
            coeffs = jnp.fft.fftshift(coeffs, axes=list(range(model.n_input_feat)))
            freqs = np.fft.fftshift(freqs)

        if numerical_cap > 0:
            # set coeffs below threshold to zero
            coeffs = jnp.where(
                jnp.abs(coeffs) < numerical_cap,
                jnp.zeros_like(coeffs),
                coeffs,
            )

        if len(freqs) == 1:
            freqs = freqs[0]

        return coeffs, freqs

    @classmethod
    def _fourier_transform(
        cls, model: Model, mfs: int, mts: int, **kwargs: Any
    ) -> jnp.ndarray:
        # Create a frequency vector with as many frequencies as model degrees,
        # oversampled by mfs
        n_freqs: jnp.ndarray = jnp.array(
            [mfs * model.degree[i] for i in range(model.n_input_feat)]
        )

        start, stop, step = 0, 2 * mts * jnp.pi, 2 * jnp.pi / n_freqs
        # Stretch according to the number of frequencies
        inputs: List = [
            jnp.arange(start, stop, step[i]) for i in range(model.n_input_feat)
        ]

        # permute with input dimensionality
        nd_inputs = jnp.array(
            jnp.meshgrid(*[inputs[i] for i in range(model.n_input_feat)])
        ).T.reshape(-1, model.n_input_feat)

        # Output vector is not necessarily the same length as input
        outputs = model(inputs=nd_inputs, **kwargs)
        outputs = outputs.reshape(
            *[inputs[i].shape[0] for i in range(model.n_input_feat)], -1
        ).squeeze()

        coeffs = jnp.fft.fftn(outputs, axes=list(range(model.n_input_feat)))

        freqs = [
            jnp.fft.fftfreq(int(mts * n_freqs[i]), 1 / n_freqs[i])
            for i in range(model.n_input_feat)
        ]
        # freqs = jnp.fft.fftfreq(mts * n_freqs, 1 / n_freqs)

        # TODO: this could cause issues with multidim input
        # FIXME: account for different frequencies in multidim input scenarios
        # Run the fft and rearrange +
        # normalize the output (using product if multidim)
        return (
            coeffs / math.prod(outputs.shape[0 : model.n_input_feat]),
            freqs,
        )

    @classmethod
    def get_psd(cls, coeffs: jnp.ndarray) -> jnp.ndarray:
        """
        Calculates the power spectral density (PSD) from given Fourier coefficients.

        Args:
            coeffs (jnp.ndarray): The Fourier coefficients.

        Returns:
            jnp.ndarray: The power spectral density.
        """
        # TODO: if we apply trim=True in advance, this will be slightly wrong..

        def abs2(x):
            return x.real**2 + x.imag**2

        scale = 2.0 / (len(coeffs) ** 2)
        return scale * abs2(coeffs)

    @classmethod
    def evaluate_Fourier_series(
        cls,
        coefficients: jnp.ndarray,
        frequencies: jnp.ndarray,
        inputs: Union[jnp.ndarray, list, float],
    ) -> float:
        """
        Evaluate the function value of a Fourier series at one point.

        Args:
            coefficients (jnp.ndarray): Coefficients of the Fourier series.
            frequencies (jnp.ndarray): Corresponding frequencies.
            inputs (jnp.ndarray): Point at which to evaluate the function.
        Returns:
            float: The function value at the input point.
        """
        if isinstance(frequencies, list):
            if len(coefficients.shape) <= len(frequencies):
                coefficients = coefficients[..., jnp.newaxis]
        else:
            if len(coefficients.shape) == 1:
                coefficients = coefficients[..., jnp.newaxis]

        if isinstance(inputs, list):
            inputs = jnp.array(inputs)
        if len(inputs.shape) < 1:
            inputs = inputs[jnp.newaxis, ...]

        if isinstance(frequencies, list):
            input_dim = len(frequencies)
            frequencies = jnp.stack(jnp.meshgrid(*frequencies))
            if input_dim != len(inputs):
                frequencies = jnp.repeat(
                    frequencies[jnp.newaxis, ...], inputs.shape[0], axis=0
                )
                freq_inputs = jnp.einsum("bi...,b->b...", frequencies, inputs)
                exponents = jnp.exp(1j * freq_inputs).T
                exp = jnp.einsum("jl...k,jl...b->b...k", coefficients, exponents)
            else:
                freq_inputs = jnp.einsum("i...,i->...", frequencies, inputs)
                exponents = jnp.exp(1j * freq_inputs).T
                exp = jnp.einsum("jl...k,jl...->k...", coefficients, exponents)
        else:
            frequencies = jnp.repeat(
                frequencies[jnp.newaxis, ...], inputs.shape[0], axis=0
            )
            freq_inputs = jnp.einsum("i...,i->i...", frequencies, inputs)
            exponents = jnp.exp(1j * freq_inputs)
            exp = jnp.einsum("j...k,ij...->ik...", coefficients, exponents)

        return jnp.squeeze(jnp.real(exp))


class FourierTree:
    """
    Sine-cosine tree representation for the algorithm by Nemkov et al.

    Computes the analytical Fourier coefficients/frequencies of a Pauli-Clifford
    circuit.  The symbolic structure of the tree (which Pauli rotations
    contribute sine/cosine factors to which leaf, and the leaf observables) is
    built once in NumPy; the parameter-dependent coefficients are then obtained
    with a small number of vectorised JAX operations, so the result remains
    jittable / differentiable with respect to the model parameters.

    Multiple input features are supported: the resulting spectrum is the
    d-dimensional set of frequency vectors.

    **Usage**:
    ```
    model = Model(...)
    tree = FourierTree(model)
    exp = tree()                          # expectation value
    coeff_list, freq_list = tree.get_spectrum()
    ```
    """

    def __init__(self, model: Model):
        """
        Tree initialisation, based on the Pauli-Clifford representation of a
        model.

        Args:
            model (Model): The Model, for which to build the tree.
        """
        self.model = model
        self.n_qubits = model.n_qubits

        quantum_tape = self._build_canonical_tape(model.params, None)

        self.parameters = [jnp.squeeze(p) for p in quantum_tape.get_parameters()]
        self.n_params = len(self.parameters)
        self.input_indices, self.all_input_indices = quantum_tape.get_input_indices()

        # Pauli generators of the (canonical) rotations, as symbolic words.
        self.pauli_words: List[PauliWord] = [
            PauliWord.from_operation(op, self.n_qubits)
            for op in quantum_tape.operations
        ]

        # Cumulative X/Y support of the rotations[0..k] (for light-cone early
        # stopping).  cumulative_xy[k] is True on every qubit touched by an X/Y
        # generator in any rotation up to index k.
        self.cumulative_xy: List[np.ndarray] = []
        running = np.zeros(self.n_qubits, dtype=bool)
        for pw in self.pauli_words:
            running = np.logical_or(running, pw.xy_mask)
            self.cumulative_xy.append(running.copy())

        # Observable Pauli words (one tree root each).
        self.observable_words: List[PauliWord] = [
            PauliWord.from_operation(obs, self.n_qubits)
            for obs in quantum_tape.observables
        ]

        # Variational parameter columns (everything that is not an input).
        input_set = set(self.all_input_indices)
        self.var_positions = np.array(
            [i for i in range(self.n_params) if i not in input_set], dtype=np.int64
        )
        # Ordered list of input feature keys (d-dimensional spectrum).
        self.features = sorted(self.input_indices.keys())

        # The explicit leaf structure is built lazily: for deep circuits the
        # number of tree paths explodes combinatorially, while the canonical
        # form above (and the merged-state support DP) remain cheap.
        self._structure_built = False

    def _ensure_structure(self) -> None:
        """Build the explicit leaf/spectrum structure on first use."""
        if not self._structure_built:
            # Symbolic structure: per root (S, C, terms) leaf arrays ...
            self._build_leaf_arrays()
            # ... and the parameter-independent frequency/weight structure.
            self._build_spectrum_structure()
            self._structure_built = True

    def _build_canonical_tape(self, params, inputs):
        """Record the circuit and transform it to Pauli-Clifford normal form."""
        # The tree describes a single parameter set.  Models can carry batched
        # parameters (e.g. after FCC sampling via ``initialize_params(repeat)``);
        # fall back to the first set in that case.
        params = jnp.asarray(params)
        if params.ndim > 2 and params.shape[0] > 1:
            warnings.warn(
                f"FourierTree supports a single parameter set; using the first "
                f"of {params.shape[0]} batched parameter sets.",
                UserWarning,
            )
            params = params[0]

        inputs = self.model._inputs_validation(
            list(range(1, self.model.n_input_feat + 1)) if inputs is None else inputs
        )
        raw_tape = self.model.script._record(params=params, inputs=inputs)
        _, obs_list = self.model._build_obs()
        return PauliCircuit.from_parameterised_circuit(
            raw_tape, observables=obs_list, n_qubits=self.n_qubits
        )

    # ------------------------------------------------------------------
    # Symbolic tree construction (NumPy)
    # ------------------------------------------------------------------
    def _build_leaf_arrays(self) -> None:
        """Collect the tree leaves for every root into integer count matrices.

        For each root (observable) this produces:
            - ``S``: (n_leaves, n_params) sine-factor counts per parameter,
            - ``C``: (n_leaves, n_params) cosine-factor counts per parameter,
            - ``terms``: (n_leaves,) complex leaf constants ``<0|O_leaf|0>``.
        """
        self.leaf_arrays: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for obs_word in self.observable_words:
            leaves: List[Tuple[np.ndarray, np.ndarray, complex]] = []
            zeros = np.zeros(self.n_params, dtype=np.int64)
            self._collect_leaves(
                obs_word, self.n_params - 1, zeros.copy(), zeros.copy(), leaves
            )
            if leaves:
                S = np.stack([leaf[0] for leaf in leaves])
                C = np.stack([leaf[1] for leaf in leaves])
                terms = np.array([leaf[2] for leaf in leaves], dtype=np.complex128)
            else:
                S = np.zeros((0, self.n_params), dtype=np.int64)
                C = np.zeros((0, self.n_params), dtype=np.int64)
                terms = np.zeros(0, dtype=np.complex128)
            self.leaf_arrays.append((S, C, terms))

    def _collect_leaves(
        self,
        observable: PauliWord,
        pauli_idx: int,
        sin_counts: np.ndarray,
        cos_counts: np.ndarray,
        leaves: List[Tuple[np.ndarray, np.ndarray, complex]],
    ) -> None:
        """Recursively enumerate the leaves of the coefficient tree.

        The incoming sine/cosine factor (from the parent edge) is already
        accumulated into ``sin_counts``/``cos_counts``.  This fuses the tree
        construction and leaf traversal of the original implementation into a
        single NumPy pass (no per-node JAX scatter updates).
        """
        if self._early_stopping_possible(pauli_idx, observable):
            return

        # Skip trailing Pauli rotations that commute with the observable.
        while pauli_idx >= 0:
            last = self.pauli_words[pauli_idx]
            if not observable.commutes_with(last):
                break
            pauli_idx -= 1
        else:  # leaf reached
            term = observable.zero_expectation()
            if term != 0:
                leaves.append((sin_counts, cos_counts, term))
            return

        last = self.pauli_words[pauli_idx]

        # Left child: cosine factor for this parameter, same observable.
        cos_left = cos_counts.copy()
        cos_left[pauli_idx] += 1
        self._collect_leaves(
            observable, pauli_idx - 1, sin_counts.copy(), cos_left, leaves
        )

        # Right child: sine factor, observable becomes  P . O.
        sin_right = sin_counts.copy()
        sin_right[pauli_idx] += 1
        self._collect_leaves(
            last.compose(observable),
            pauli_idx - 1,
            sin_right,
            cos_counts.copy(),
            leaves,
        )

    def _early_stopping_possible(self, pauli_idx: int, observable: PauliWord) -> bool:
        """Whether a node can be discarded (all reachable expectations vanish).

        Mirrors the criterion of Nemkov et al. (light cone): a qubit on which
        the observable carries an X/Y must be covered by an X/Y generator of
        some remaining rotation (rotations[0..pauli_idx]); otherwise that X/Y can
        never be rotated into a diagonal term and the whole node contributes
        zero.  Equivalently, the node survives iff every qubit is either I/Z in
        the observable or covered by the cumulative rotation X/Y support.
        """
        obs_iz = np.logical_not(observable.xy_mask)
        combined = np.logical_or(obs_iz, self.cumulative_xy[pauli_idx]).all()
        return not bool(combined)

    # ------------------------------------------------------------------
    # Frequency / weight structure (NumPy, parameter independent)
    # ------------------------------------------------------------------
    def _build_spectrum_structure(self) -> None:
        """Build, per root, the frequency vectors and the (n_freq, n_leaves)
        weight matrix ``W`` such that ``coeffs = W @ (terms * variational)``.
        """
        self.freqs_per_root: List[np.ndarray] = []
        self.weights_per_root: List[np.ndarray] = []
        d = len(self.features)

        for S, C, _ in self.leaf_arrays:
            n_leaves = S.shape[0]
            freq_to_col: Dict[tuple, np.ndarray] = defaultdict(
                lambda: np.zeros(n_leaves, dtype=np.complex128)
            )
            for leaf in range(n_leaves):
                # Per-feature sine/cosine counts and their binomial expansion.
                per_feature_terms = []
                half_exp = 0
                for feat in self.features:
                    cols = self.input_indices[feat]
                    s = int(S[leaf, cols].sum())
                    c = int(C[leaf, cols].sum())
                    half_exp += s + c
                    per_feature_terms.append(self._binomial_terms(s, c))
                half = 0.5**half_exp

                if d == 0:
                    freq_to_col[(0,)][leaf] += half
                    continue

                for combo in itertools.product(*per_feature_terms):
                    omega = tuple(int(o) for o, _ in combo)
                    weight = half
                    for _, w in combo:
                        weight *= w
                    freq_to_col[omega][leaf] += weight

            if freq_to_col:
                omegas = sorted(freq_to_col.keys())
                W = np.stack([freq_to_col[o] for o in omegas])  # (n_freq, n_leaves)
                freqs = np.array(omegas, dtype=np.int64)  # (n_freq, d)
            else:
                freqs = np.zeros((1, max(d, 1)), dtype=np.int64)
                W = np.zeros((1, n_leaves), dtype=np.complex128)

            # Collapse to 1-D frequency array for the single-feature case.
            if freqs.shape[1] == 1:
                freqs = freqs[:, 0]
            self.freqs_per_root.append(freqs)
            # Keep W in NumPy complex128: its entries are dyadic rationals
            # (binomial weights x 0.5^k x i^m), which are exact in float64 --
            # this allows exact symbolic zero-tests in get_exact_support.
            self.weights_per_root.append(W)

    @staticmethod
    def _binomial_terms(s: int, c: int) -> List[Tuple[int, float]]:
        """Expand ``cos^c (i sin)^s`` in ``e^{i omega x}`` (without the 0.5 factor).

        Returns a list of ``(omega, weight)`` with
        ``omega = 2a + 2b - s - c`` and ``weight = C(s,a) C(c,b) (-1)^{s-a}``.
        """
        terms = []
        for a in range(s + 1):
            for b in range(c + 1):
                weight = math.comb(s, a) * math.comb(c, b) * (-1) ** (s - a)
                terms.append((2 * a + 2 * b - s - c, float(weight)))
        return terms

    # ------------------------------------------------------------------
    # Vectorised numeric evaluation (JAX)
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_pow(base: jnp.ndarray, exp: jnp.ndarray) -> jnp.ndarray:
        """Elementwise ``base ** exp`` for real base and non-negative integer
        exponents, correct for negative bases (avoids ``log`` of negatives).

        Args:
            base: real array of shape ``(n,)``.
            exp: integer array of shape ``(n_leaves, n)``.
        """
        mag = jnp.abs(base)[None, :] ** exp
        sign = jnp.where(exp % 2 == 0, 1.0, jnp.sign(base)[None, :])
        return sign * mag

    _I_POW = None  # set lazily to jnp.array([1, 1j, -1, -1j])

    def _leaf_factors(
        self, S: np.ndarray, C: np.ndarray, columns: np.ndarray
    ) -> jnp.ndarray:
        """Per-leaf product ``prod_i cos(theta_i)^{C} (i sin(theta_i))^{S}`` over
        the given parameter ``columns`` (vectorised over leaves).
        """
        if FourierTree._I_POW is None:
            FourierTree._I_POW = jnp.array([1, 1j, -1, -1j])

        if S.shape[0] == 0:
            return jnp.zeros(0, dtype=jnp.complex64)

        theta = jnp.stack([self.parameters[i] for i in columns])
        S_sub = jnp.asarray(S[:, columns])
        C_sub = jnp.asarray(C[:, columns])

        cos_part = self._safe_pow(jnp.cos(theta), C_sub)
        sin_mag = self._safe_pow(jnp.sin(theta), S_sub)
        i_part = FourierTree._I_POW[S_sub % 4]
        return jnp.prod(cos_part * sin_mag * i_part, axis=1)

    def __call__(
        self,
        params: Optional[jnp.ndarray] = None,
        inputs: Optional[jnp.ndarray] = None,
        **kwargs,
    ) -> jnp.ndarray:
        """
        Evaluate the expectation value(s) of the model's observables via the
        sine-cosine tree (equivalent to the circuit expectation).

        Args:
            params (Optional[jnp.ndarray]): Model parameters. Defaults to the
                model's parameters.
            inputs (Optional[jnp.ndarray]): Inputs to the circuit. Defaults to 1.

        Returns:
            jnp.ndarray: Expectation value per observable (or their mean if
                ``force_mean`` is set).

        Raises:
            NotImplementedError: For execution types other than "expval" or when
                noise is requested.
        """
        params = (
            self.model._params_validation(params)
            if params is not None
            else self.model.params
        )
        inputs = (
            self.model._inputs_validation(inputs)
            if inputs is not None
            else self.model._inputs_validation(1.0)
        )

        if kwargs.get("execution_type", "expval") != "expval":
            raise NotImplementedError(
                f'Currently, only "expval" execution type is supported when '
                f"building FourierTree. Got {kwargs.get('execution_type', 'expval')}."
            )
        if kwargs.get("noise_params", None) is not None:
            raise NotImplementedError(
                "Currently, noise is not supported when building FourierTree."
            )

        # Re-derive the (canonical) parameter values for the requested inputs;
        # the tree structure (leaf arrays) is unchanged.
        quantum_tape = self._build_canonical_tape(params, inputs)
        self.parameters = [jnp.squeeze(p) for p in quantum_tape.get_parameters()]

        self._ensure_structure()
        all_columns = np.arange(self.n_params, dtype=np.int64)
        results = []
        for S, C, terms in self.leaf_arrays:
            factors = self._leaf_factors(S, C, all_columns)
            results.append(jnp.real(jnp.sum(jnp.asarray(terms) * factors)))
        results = jnp.array(results)

        if kwargs.get("force_mean", False):
            return jnp.mean(results)
        return results

    def get_spectrum(
        self, force_mean: bool = False
    ) -> Tuple[List[jnp.ndarray], List[jnp.ndarray]]:
        """
        Compute the Fourier spectrum (coefficients and frequencies) of the tree.

        Args:
            force_mean (bool, optional): Average the coefficients over all
                observables (roots). Defaults to False.

        Returns:
            Tuple[List[jnp.ndarray], List[jnp.ndarray]]:
                - List of coefficients, one entry per observable (root).
                - List of corresponding frequencies, one entry per root.
                When ``force_mean`` is set, both lists have a single entry.
        """
        self._ensure_structure()
        per_root_coeffs: List[jnp.ndarray] = []
        for (S, C, terms), W in zip(self.leaf_arrays, self.weights_per_root):
            leaf_const = jnp.asarray(terms) * self._leaf_factors(
                S, C, self.var_positions
            )
            per_root_coeffs.append(jnp.asarray(W) @ leaf_const)

        return self._combine_roots(per_root_coeffs, self.freqs_per_root, force_mean)

    def get_exact_support(self, method: str = "tree") -> List[np.ndarray]:
        r"""Symbolically derive the exact frequency support (no sampling).

        A frequency :math:`\omega` belongs to the exact spectrum iff its
        coefficient :math:`c_\omega(\theta) = \sum_l W_{\omega l}\,
        \text{term}_l\, v_l(\theta)` is not identically zero in the
        variational parameters :math:`\theta`.

        Two methods are available:

        - ``"tree"`` (default, fully exact): enumerates the explicit tree
          leaves.  Because the branch index strictly decreases along every tree
          path, each parameter contributes **at most one** sine *or* cosine
          factor per leaf (:math:`S_{li}, C_{li} \in \{0, 1\}`).  Every
          variational leaf factor :math:`v_l` is therefore a *square-free*
          monomial over :math:`\{1, \cos\theta_i, i\sin\theta_i\}`, and
          monomials with distinct signatures are linearly independent functions
          (no :math:`\cos^2 + \sin^2` identities can arise without squares).
          Hence

          .. math::
              c_\omega \equiv 0 \iff \sum_{l \in g} W_{\omega l}\,\text{term}_l
              = 0 \quad \text{for every signature group } g.

          Since all involved quantities are dyadic rationals times
          :math:`\{\pm 1, \pm i\}`, the group sums are exact in float64 and the
          zero-test is exact.  The number of leaves can however grow
          exponentially with circuit depth.

        - ``"dp"`` (scalable): merges tree nodes with identical
          ``(rotation index, observable)`` — at most ``n_params * 4^n_qubits``
          states — and tracks the achievable input sine/cosine count pairs
          ``(s, c)`` per state.  The support is the union of the (exact)
          expansion supports of :math:`\cos^c x\, (i \sin x)^s` over all
          achievable pairs.  This is exact per tree path (including interior
          zero coefficients of the expansions), but unlike ``"tree"`` it cannot
          detect coefficients that cancel identically *across* paths with
          identical variational signatures (e.g. directly repeated encodings).
          It therefore yields a tight superset in such corner cases.
          Currently restricted to a single input feature.

        Args:
            method (str): ``"tree"`` (fully exact) or ``"dp"`` (scalable).

        Returns:
            List[np.ndarray]: For each observable (root), the frequency vectors
            with not-identically-zero coefficient — shape ``(n_freq,)`` for a
            single input feature, ``(n_freq, n_features)`` otherwise.
        """
        if method == "dp":
            return self._support_dp()
        if method != "tree":
            raise ValueError(f"Unknown method '{method}'. Use 'tree' or 'dp'.")

        self._ensure_structure()
        supports = []
        for (S, C, terms), W, freqs in zip(
            self.leaf_arrays, self.weights_per_root, self.freqs_per_root
        ):
            freqs = np.asarray(freqs)
            n_leaves = S.shape[0]
            if n_leaves == 0:
                supports.append(freqs[:0])
                continue

            # Group leaves by their variational sine/cosine signature.
            signature = np.hstack([S[:, self.var_positions], C[:, self.var_positions]])
            _, groups = np.unique(signature, axis=0, return_inverse=True)
            n_groups = int(groups.max()) + 1

            # Per-group sums of W[omega, l] * term_l, accumulated exactly.
            contrib = (W * terms[None, :]).T  # (n_leaves, n_freq)
            group_sums = np.zeros((n_groups, W.shape[0]), dtype=np.complex128)
            np.add.at(group_sums, groups, contrib)

            mask = (np.abs(group_sums) > 1e-12).any(axis=0)  # (n_freq,)
            supports.append(freqs[mask])
        return supports

    def _support_dp(self) -> List[np.ndarray]:
        """Merged-state dynamic program for the frequency support.

        Instead of enumerating all (worst-case exponentially many) tree paths,
        nodes are merged on ``(rotation index, bare observable)``.  Each state
        stores the set of achievable input ``(s, c)`` count pairs as a bitmask,
        so transitions are O(1) big-int operations.  See
        :meth:`get_exact_support` for semantics and limitations.
        """
        if len(self.features) != 1:
            raise NotImplementedError(
                "The 'dp' support method currently supports exactly one input "
                "feature; use method='tree' for multi-feature models."
            )

        n = self.n_qubits
        is_input = np.zeros(self.n_params, dtype=bool)
        is_input[self.all_input_indices] = True
        n_inp = int(is_input.sum())
        stride = n_inp + 1  # bit index for (s, c) is  s * stride + c

        def encode(word: PauliWord) -> Tuple[int, int]:
            x = z = 0
            for q in range(n):
                x |= int(word.x[q]) << q
                z |= int(word.z[q]) << q
            return x, z

        paulis = [encode(w) for w in self.pauli_words]
        cum_xy = []
        running = 0
        for xp, _ in paulis:
            running |= xp
            cum_xy.append(running)

        def parity(v: int) -> int:
            return bin(v).count("1") & 1

        def dp(idx: int, xo: int, zo: int, memo: dict) -> int:
            # Light-cone early stopping (cf. _early_stopping_possible).
            if idx >= 0 and (xo & ~cum_xy[idx]):
                return 0
            # Skip trailing rotations that commute with the observable.
            while idx >= 0:
                xp, zp = paulis[idx]
                if parity(xo & zp) ^ parity(zo & xp):
                    break
                idx -= 1
            else:  # leaf: counts (s=0, c=0) iff the observable is diagonal
                return 1 if xo == 0 else 0
            key = (idx, xo, zo)
            hit = memo.get(key)
            if hit is not None:
                return hit
            xp, zp = paulis[idx]
            cos_child = dp(idx - 1, xo, zo, memo)
            sin_child = dp(idx - 1, xo ^ xp, zo ^ zp, memo)
            if is_input[idx]:
                # Active input gate: cosine increments c, sine increments s.
                val = (cos_child << 1) | (sin_child << stride)
            else:
                val = cos_child | sin_child
            memo[key] = val
            return val

        # Recursion depth is bounded by the number of rotations.
        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(max(old_limit, self.n_params + 1000))
        try:
            supports = []
            for obs in self.observable_words:
                memo: dict = {}
                xo, zo = encode(obs)
                mask = dp(self.n_params - 1, xo, zo, memo)
                freqs: set = set()
                while mask:
                    bit = mask & -mask
                    i = bit.bit_length() - 1
                    freqs |= self._expansion_support(i // stride, i % stride)
                    mask ^= bit
                supports.append(np.array(sorted(freqs), dtype=np.int64))
        finally:
            sys.setrecursionlimit(old_limit)
        return supports

    @staticmethod
    @lru_cache(maxsize=None)
    def _expansion_support(s: int, c: int) -> frozenset:
        r"""Frequencies with non-zero coefficient in :math:`\cos^c x (i\sin x)^s`.

        Computed exactly with integer arithmetic via the polynomial
        :math:`(t - 1)^s (t + 1)^c` (with :math:`t = e^{2ix}` up to a shift);
        interior coefficients can vanish, e.g. :math:`\cos x \sin x` only
        contains :math:`\pm 2`.
        """
        coeffs = [1]
        for _ in range(s):  # multiply by (t - 1)
            new = [0] * (len(coeffs) + 1)
            for i, a in enumerate(coeffs):
                new[i + 1] += a
                new[i] -= a
            coeffs = new
        for _ in range(c):  # multiply by (t + 1)
            new = [0] * (len(coeffs) + 1)
            for i, a in enumerate(coeffs):
                new[i + 1] += a
                new[i] += a
            coeffs = new
        m = s + c
        return frozenset(2 * k - m for k, a in enumerate(coeffs) if a != 0)

    def _combine_roots(
        self,
        per_root_coeffs: List[jnp.ndarray],
        per_root_freqs: List[np.ndarray],
        force_mean: bool,
    ) -> Tuple[List[jnp.ndarray], List[jnp.ndarray]]:
        """Assemble the per-root spectra, optionally averaging over roots."""
        if not force_mean:
            coefficients = [jnp.asarray(c) for c in per_root_coeffs]
            frequencies = [jnp.asarray(f) for f in per_root_freqs]
            return coefficients, frequencies

        # Average over roots on the union of all frequency vectors.
        accum: Dict[tuple, complex] = defaultdict(complex)
        for coeffs, freqs in zip(per_root_coeffs, per_root_freqs):
            freqs_np = np.asarray(freqs)
            for k in range(freqs_np.shape[0]):
                key = (
                    (int(freqs_np[k]),)
                    if freqs_np.ndim == 1
                    else tuple(int(v) for v in freqs_np[k])
                )
                accum[key] += complex(coeffs[k])
        n_roots = max(len(per_root_coeffs), 1)
        keys = sorted(accum.keys())
        mean_coeffs = jnp.array([accum[k] / n_roots for k in keys])
        freq_arr = np.array(keys, dtype=np.int64)
        if freq_arr.shape[1] == 1:
            freq_arr = freq_arr[:, 0]
        return [mean_coeffs], [jnp.asarray(freq_arr)]


class FCC:
    @classmethod
    def get_fcc(
        cls,
        model: Model,
        n_samples: int,
        random_key: Optional[random.PRNGKey] = None,
        method: Optional[str] = "pearson",
        scale: Optional[bool] = False,
        weight: Optional[bool] = False,
        trim_redundant: Optional[bool] = True,
        **kwargs,
    ) -> float:
        """
        Shortcut method to get just the FCC.
        This includes
        1. What is done in `get_fourier_fingerprint`:
            1. Calculating the coefficients (using `n_samples`)
            2. Correlating the result from 1) using `method`
            3. Weighting the correlation matrix (if `weight` is True)
            4. Remove redundancies
        2. What is done in `calculate_fcc`:
            1. Absolute of the fingerprint
            2. Average

        Args:
            model (Model): The QFM model
            n_samples (int): Number of samples to calculate average of coefficients
            random_key (Optional[random.PRNGKey]): JAX random key for parameter
                initialization. If None, uses the model's internal random key.
            method (Optional[str], optional): Correlation method. Supported values are
                "pearson", "complex_pearson", and "spearman". Defaults to "pearson".
            scale (Optional[bool], optional): Whether to scale the number of samples.
                Defaults to False.
            weight (Optional[bool], optional): Whether to weight the correlation matrix.
                Defaults to False.
            trim_redundant (Optional[bool], optional): Whether to remove redundant
                correlations. Defaults to False.
            **kwargs (Any): Additional keyword arguments for the model function.

        Returns:
            float: The FCC
        """

        # Memory-efficient fast path
        if trim_redundant and not weight:
            _, coeffs, freqs = cls._calculate_coefficients(
                model, n_samples, random_key, scale, **kwargs
            )
            pos_idx = cls._calculate_mask(freqs)
            coeffs_flat = coeffs.reshape(-1, coeffs.shape[-1])
            coeffs_sub = coeffs_flat[pos_idx]

            fp = cls._correlate(coeffs_sub.transpose(), method=method)
            abs_fp = jnp.abs(fp)
            diag = jnp.abs(jnp.diagonal(fp))

            total_sum = jnp.nansum(abs_fp)
            total_count = jnp.sum(jnp.isfinite(abs_fp))
            diag_sum = jnp.nansum(diag)
            diag_count = jnp.sum(jnp.isfinite(diag))

            lower_sum = (total_sum - diag_sum) / 2.0
            lower_count = (total_count - diag_count) / 2.0
            return lower_sum / lower_count

        fourier_fingerprint, _ = cls.get_fourier_fingerprint(
            model,
            n_samples,
            random_key,
            method,
            scale,
            weight,
            trim_redundant=trim_redundant,
            **kwargs,
        )

        return cls.calculate_fcc(fourier_fingerprint)

    @classmethod
    def get_fourier_fingerprint(
        cls,
        model: Model,
        n_samples: int,
        random_key: Optional[random.PRNGKey] = None,
        method: Optional[str] = "pearson",
        scale: Optional[bool] = False,
        weight: Optional[bool] = False,
        trim_redundant: Optional[bool] = True,
        nan_to_one: Optional[bool] = False,
        **kwargs: Any,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Shortcut method to get just the fourier fingerprint.
        This includes
        1. Calculating the coefficients (using `n_samples`)
        2. Correlating the result from 1) using `method`
        3. Weighting the correlation matrix (if `weight` is True)
        4. Remove redundancies (if `trim_redundant` is True)

        Args:
            model (Model): The QFM model
            n_samples (int): Number of samples to calculate average of coefficients
            random_key (Optional[random.PRNGKey]): JAX random key for parameter
                initialization. If None, uses the model's internal random key.
            method (Optional[str], optional): Correlation method. Supported values are
                "pearson", "complex_pearson", and "spearman". Defaults to "pearson".
            scale (Optional[bool], optional): Whether to scale the number of samples.
                Defaults to False.
            weight (Optional[bool], optional): Whether to weight the correlation matrix.
                Defaults to False.
            trim_redundant (Optional[bool], optional): Whether to remove redundant
                correlations. Defaults to True.
            nan_to_one (Optional[bool], optional): Whether to set nan to 1.
                Defaults to False.
            **kwargs: Additional keyword arguments for the model function.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: The fourier fingerprint
            and the frequency indices
        """
        _, coeffs, freqs = cls._calculate_coefficients(
            model, n_samples, random_key, scale, **kwargs
        )

        # Memory-efficient fast path
        if trim_redundant and not weight:
            pos_idx = cls._calculate_mask(freqs)

            # Flatten all frequency axes; the last axis is the sample
            # axis. `_calculate_mask` returns flat indices in C order,
            # matching this reshape.
            coeffs_flat = coeffs.reshape(-1, coeffs.shape[-1])
            coeffs_sub = coeffs_flat[pos_idx]

            fourier_fingerprint = cls._correlate(coeffs_sub.transpose(), method=method)

            if nan_to_one:
                fourier_fingerprint = jnp.where(
                    jnp.isnan(fourier_fingerprint), 1.0, fourier_fingerprint
                )

            M = fourier_fingerprint.shape[0]
            lower_tri_mask = jnp.tri(M, k=-1, dtype=bool)
            fourier_fingerprint = jnp.where(
                lower_tri_mask, fourier_fingerprint, jnp.nan
            )

            row_mask = jnp.any(jnp.isfinite(fourier_fingerprint), axis=1)
            col_mask = jnp.any(jnp.isfinite(fourier_fingerprint), axis=0)
            fourier_fingerprint = fourier_fingerprint[row_mask][:, col_mask]

            return fourier_fingerprint, freqs

        fourier_fingerprint = cls._correlate(coeffs.transpose(), method=method)

        if nan_to_one:
            # set nan to 1
            fourier_fingerprint[jnp.isnan(fourier_fingerprint)] = 1.0

        # perform weighting if requested
        fourier_fingerprint = (
            cls._weighting_mean(fourier_fingerprint, coeffs)
            if weight
            else fourier_fingerprint
        )

        if trim_redundant:
            pos_idx = cls._calculate_mask(freqs)

            # restrict to the positive-frequency sub-block (M x M with
            # M = number of non-negative flat-frequencies) instead of
            # building a full N x N mask. This avoids the O(N^2) float
            fourier_fingerprint = fourier_fingerprint[pos_idx][:, pos_idx]

            # keep only the strict lower triangle; the rest -> nan
            M = fourier_fingerprint.shape[0]
            lower_tri_mask = jnp.tri(M, k=-1, dtype=bool)
            fourier_fingerprint = jnp.where(
                lower_tri_mask, fourier_fingerprint, jnp.nan
            )

            row_mask = jnp.any(jnp.isfinite(fourier_fingerprint), axis=1)
            col_mask = jnp.any(jnp.isfinite(fourier_fingerprint), axis=0)

            fourier_fingerprint = fourier_fingerprint[row_mask][:, col_mask]

        return fourier_fingerprint, freqs

    @classmethod
    def calculate_fcc(
        cls,
        fourier_fingerprint: jnp.ndarray,
    ) -> float:
        """
        Method to calculate the FCC based on an existing correlation matrix.
        Calculate absolute and then the average over this matrix.
        The Fingerprint can be obtained via `get_fourier_fingerprint`

        Args:
            fourier_fingerprint (jnp.ndarray): Correlation matrix of coefficients
        Returns:
            float: The FCC
        """
        # apply the mask on the fingerprint
        return jnp.nanmean(jnp.abs(fourier_fingerprint))

    @classmethod
    def _calculate_mask(cls, freqs: jnp.ndarray) -> jnp.ndarray:
        """
        Determine the flat indices of the Fourier correlation matrix
        that lie on a non-negative-frequency row/column. Together with
        the strict-lower-triangle condition (handled by the caller),
        these indices select the entries of the correlation matrix
        that survive the redundancy filter applied in
        `get_fourier_fingerprint`:

        - rows/columns whose flat frequency component is negative are
          discarded (they are the complex-conjugate redundancies of
          their positive counterparts);
        - of the remaining positive-frequency sub-block, only the
          strict lower triangle is kept (the upper triangle, including
          the diagonal, contains either duplicates from symmetry or
          self-correlations).

        Args:
            freqs (jnp.ndarray): Array of frequencies. Either a 1-D
                vector (single input feature) or a 2-D array of shape
                ``(n_input_feat, K)`` whose rows are the per-axis
                frequency vectors.

        Returns:
            jnp.ndarray: 1-D int array of flat indices selecting the
                non-negative-frequency rows/cols of the fingerprint.
        """
        freqs_arr = jnp.asarray(freqs)

        if freqs_arr.ndim == 1:
            pos_flat = freqs_arr >= 0
        else:
            # N-D case: build the per-axis non-negativity masks and
            # combine them via broadcasting (no float `jnp.outer`!),
            # then flatten to match the row-major flattening used by
            # the upstream coefficient/correlation pipeline.
            axes_pos = [freqs_arr[i] >= 0 for i in range(freqs_arr.shape[0])]
            expanded = []
            n_axes = len(axes_pos)
            for i, p in enumerate(axes_pos):
                shape = [1] * n_axes
                shape[i] = p.shape[0]
                expanded.append(p.reshape(shape))
            nd_pos = reduce(jnp.logical_and, expanded)
            pos_flat = nd_pos.flatten()

        return jnp.where(pos_flat)[0]

    @classmethod
    def _calculate_coefficients(
        cls,
        model: Model,
        n_samples: int,
        random_key: Optional[random.PRNGKey] = None,
        scale: bool = False,
        **kwargs: Any,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Calculates the Fourier coefficients of a given model
        using `n_samples`.
        Optionally, `noise_params` can be passed to perform noisy simulation.

        Args:
            model (Model): The QFM model
            n_samples (int): Number of samples to calculate average of coefficients
            random_key (Optional[random.PRNGKey]): JAX random key for parameter
                initialization. If None, uses the model's internal random key.
            scale (bool, optional): Whether to scale the number of samples.
                Defaults to False.
            **kwargs: Additional keyword arguments for the model function.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: Parameters and Coefficients of size NxK
        """
        if n_samples > 0:
            if scale:
                total_samples = int(
                    jnp.power(2, model.n_qubits) * n_samples * model.n_input_feat
                )
                log.info(f"Using {total_samples} samples.")
            else:
                total_samples = n_samples
            model.initialize_params(random_key, repeat=total_samples)
        else:
            total_samples = 1

        coeffs, freqs = Coefficients.get_spectrum(
            model, shift=True, trim=True, **kwargs
        )

        return model.params, coeffs, freqs

    @classmethod
    def _correlate(cls, mat: jnp.ndarray, method: str = "pearson") -> jnp.ndarray:
        """
        Correlates two arrays using `method`.
        Currently, `pearson`, `complex_pearson`, and `spearman` are supported.

        Args:
            mat (jnp.ndarray): Array of shape (N, K)
            method (str, optional): Correlation method. Defaults to "pearson".

        Raises:
            ValueError: If the method is not supported.

        Returns:
            jnp.ndarray: Correlation matrix of `a` and `b`.
        """
        assert len(mat.shape) >= 2, "Input matrix must have at least 2 dimensions"

        # Note that for the general n-D case, we have to flatten along
        # the first axis (last one is batch).
        # Note that the order here is important so we can easily filter out
        # negative coefficients later.
        # Consider the following example: [[1,2,3],[4,5,6],[7,8,9]]
        # we want to get [1, 4, 7, 2, 5, 8, 3, 6, 9]
        # such that after correlation, all positive indexed coefficients
        # will be in the bottom right quadrant
        if method == "pearson":
            result = cls._pearson(mat.reshape(mat.shape[0], -1))
            # result = cls._pearson(mat.reshape(mat.shape[-1], -1, order="F"))
        elif method == "complex_pearson":
            result = cls._complex_pearson(mat.reshape(mat.shape[0], -1))
        elif method == "spearman":
            result = cls._spearman(mat.reshape(mat.shape[0], -1))
            # result = cls._spearman(mat.reshape(mat.shape[-1], -1, order="F"))
        else:
            raise ValueError(
                f"Unknown correlation method: {method}. \
                             Must be 'pearson', 'complex_pearson' or 'spearman'."
            )

        return result

    @classmethod
    def _complex_pearson(
        cls, mat: jnp.ndarray, cov: Optional[bool] = False, minp: Optional[int] = 1
    ) -> jnp.ndarray:
        """
        Compute the complex Pearson correlation between columns of `mat`,
        permitting missing values (NaN or ±Inf).

        This uses the Hermitian normalized covariance
        sum(conj(x_i - mean_i) * (x_j - mean_j)) /
        sqrt(sum(abs(x_i - mean_i)**2) * sum(abs(x_j - mean_j)**2)).
        Consequently, if column j is exp(1j * phi) times column i, then
        abs(corr[i, j]) is 1 and angle(corr[i, j]) is phi.

        Args:
            mat : array_like, shape (N, K)
                Input data.
            cov : bool, optional
                If True, return the sample covariance matrix instead of
                correlation. Defaults to False.
            minp : int, optional
                Minimum number of paired observations required to form a correlation.
                If the number of valid pairs for (i, j) is < minp, the result is NaN.

        Returns:
            corr : ndarray, shape (K, K)
                Complex Pearson correlation matrix.
        """
        mat = jnp.asarray(mat)
        real_dtype = jnp.asarray(mat.real).dtype

        mask = jnp.isfinite(mat)
        fmask = mask.astype(real_dtype)
        safe = jnp.where(mask, mat, 0.0)

        nobs = fmask.T @ fmask
        nobs_safe = jnp.where(nobs > 0, nobs, 1.0)

        sum_x = safe.T @ fmask
        sum_y = fmask.T @ safe

        masked = safe * fmask
        sum_conj_xy = jnp.conj(masked).T @ masked

        safe_abs_sq = jnp.abs(safe) ** 2
        sum_abs_x2 = safe_abs_sq.T @ fmask
        sum_abs_y2 = fmask.T @ safe_abs_sq

        ssx = sum_abs_x2 - jnp.abs(sum_x) ** 2 / nobs_safe
        ssy = sum_abs_y2 - jnp.abs(sum_y) ** 2 / nobs_safe
        sxy = sum_conj_xy - (jnp.conj(sum_x) * sum_y) / nobs_safe

        if cov:
            denom = jnp.where(nobs > 1, nobs - 1, jnp.nan)
            result = sxy / denom
        else:
            denom = jnp.sqrt(ssx * ssy)
            result = jnp.where(denom > 0, sxy / denom, jnp.nan)
            magnitude = jnp.abs(result)
            result = jnp.where(magnitude > 1.0, result / magnitude, result)

        result = jnp.where(nobs < minp, jnp.nan, result)

        return result

    @classmethod
    def _pearson(
        cls, mat: jnp.ndarray, cov: Optional[bool] = False, minp: Optional[int] = 1
    ) -> jnp.ndarray:
        """
        Based on Pandas correlation method as implemented here:
        https://github.com/pandas-dev/pandas/blob/main/pandas/_libs/algos.pyx

        Compute Pearson correlation between columns of `mat`,
        permitting missing values (NaN or ±Inf).

        If the input is complex, real and imaginary parts are stacked along
        the sample axis so that both components contribute to the correlation
        without discarding information.

        Args:
            mat : array_like, shape (N, K)
                Input data.
            cov : bool, optional
                If True, return the sample covariance matrix instead of
                correlation. Defaults to False.
            minp : int, optional
                Minimum number of paired observations required to form a correlation.
                If the number of valid pairs for (i, j) is < minp, the result is NaN.

        Returns:
            corr : ndarray, shape (K, K)
                Pearson correlation matrix.
        """
        # Preserve complex information by splitting into real / imag samples
        if jnp.iscomplexobj(mat):
            mat = jnp.concatenate([mat.real, mat.imag], axis=0)

        mat = jnp.asarray(mat)

        # pre-compute finite mask  (N, K)
        mask = jnp.isfinite(mat)
        fmask = mask.astype(mat.dtype)

        # Replace non-finite entries with 0 so arithmetic is safe;
        # the mask keeps track of validity.
        safe = jnp.where(mask, mat, 0.0)

        # Pairwise valid-observation counts  (K, K)
        nobs = fmask.T @ fmask

        # Pairwise sums (only over mutually valid rows)
        # For columns i, j the "valid" rows are mask[:,i] & mask[:,j].
        # sum_x[i,j] = sum of mat[:,i] where both i and j are valid.
        # Using: safe[:,i] * mask[:,j] zeroes out rows invalid for j.
        # Then summing over N gives sum_x[i,j].
        # safe.T @ fmask  gives (K, K) where entry (i,j) = sum of safe[:,i]*mask[:,j]
        sum_x = safe.T @ fmask  # (K, K) – row-var sums
        sum_y = fmask.T @ safe  # (K, K) – col-var sums

        # Note: explicit means (sum_x/nobs, sum_y/nobs) are not needed as
        # separate variables — the computational formula used below
        # (e.g. ssx = Σx² − (Σx)²/n) implicitly handles mean-centering.

        # Cross products, sum-of-squares via computational formula:
        #   ssx = Σx² − (Σx)²/n,  ssy = Σy² − (Σy)²/n,
        #   sxy = Σxy − (Σx)(Σy)/n
        # All sums are taken over the pairwise-valid subset for each (i,j).
        masked = safe * fmask  # same as safe but explicit
        sum_xy = masked.T @ masked  # (K, K)

        # ssx[i,j] = sum_xx_ij - nobs * mean_x^2   (but sum_xx_ij uses pair mask)
        # We need sum of x^2 over the *pair* mask, not just column mask.
        # sum_x2[i,j] = sum_n  safe[n,i]^2 * mask[n,i] * mask[n,j]
        safe_sq = safe**2
        sum_x2 = safe_sq.T @ fmask  # (K, K)
        sum_y2 = fmask.T @ safe_sq  # (K, K)

        ssx = sum_x2 - sum_x**2 / jnp.where(nobs > 0, nobs, 1.0)
        ssy = sum_y2 - sum_y**2 / jnp.where(nobs > 0, nobs, 1.0)
        sxy = sum_xy - (sum_x * sum_y) / jnp.where(nobs > 0, nobs, 1.0)

        if cov:
            denom = jnp.where(nobs > 1, nobs - 1, jnp.nan)
            result = sxy / denom
        else:
            denom = jnp.sqrt(ssx * ssy)
            result = jnp.where(denom > 0, sxy / denom, jnp.nan)
            # clip numerical drift to [-1, 1]
            result = jnp.clip(result, -1.0, 1.0)

        # Enforce minp: set entries with too few observations to NaN
        result = jnp.where(nobs < minp, jnp.nan, result)

        return result

    @classmethod
    def _spearman(cls, mat: jnp.ndarray, minp: Optional[int] = 1) -> jnp.ndarray:
        """
        Based on Pandas correlation method as implemented here:
        https://github.com/pandas-dev/pandas/blob/main/pandas/_libs/algos.pyx

        Compute Spearman correlation between columns of `mat`,
        permitting missing values (NaN or ±Inf).

        If the input is complex, real and imaginary parts are stacked along
        the sample axis so that both components contribute to the correlation
        without discarding information.

        Args:
            mat : array_like, shape (N, K)
                Input data.
            minp : int, optional
                Minimum number of paired observations required to form a correlation.
                If the number of valid pairs for (i, j) is < minp, the result is NaN.

        Returns:
            corr : ndarray, shape (K, K)
                Spearman correlation matrix.
        """
        # Preserve complex information by splitting into real / imag samples
        if jnp.iscomplexobj(mat):
            mat = jnp.concatenate([mat.real, mat.imag], axis=0)

        mat = jnp.asarray(mat)
        N, K = mat.shape

        # trivial all-NaN answer if too few rows
        if N < minp:
            return jnp.full((K, K), jnp.nan)

        # mask of finite entries
        mask = jnp.isfinite(mat)  # shape (N, K), dtype=bool

        # precompute ranks column-wise ignoring NaNs
        ranks = np.full((N, K), np.nan)
        for j in range(K):
            valid = mask[:, j]
            if valid.any():
                ranks[valid, j] = rankdata(mat[valid, j], method="average")

        ranks = jnp.asarray(ranks)

        # Vectorised Pearson on the ranks
        # Replace NaN ranks with 0; use mask to track validity.
        rank_mask = jnp.isfinite(ranks)
        safe_ranks = jnp.where(rank_mask, ranks, 0.0)

        # Pairwise valid-observation counts  (K, K)
        fmask = rank_mask.astype(ranks.dtype)
        nobs = fmask.T @ fmask

        # Pairwise sums over mutually-valid rows
        sum_x = safe_ranks.T @ fmask  # (K, K)
        sum_y = fmask.T @ safe_ranks  # (K, K)

        # Pairwise products
        masked_ranks = safe_ranks * fmask  # same as safe_ranks
        sum_xy = masked_ranks.T @ masked_ranks  # (K, K)

        safe_sq = safe_ranks**2
        sum_x2 = safe_sq.T @ fmask  # (K, K)
        sum_y2 = fmask.T @ safe_sq  # (K, K)

        nobs_safe = jnp.where(nobs > 0, nobs, 1.0)
        ssx = sum_x2 - sum_x**2 / nobs_safe
        ssy = sum_y2 - sum_y**2 / nobs_safe
        sxy = sum_xy - (sum_x * sum_y) / nobs_safe

        denom = jnp.sqrt(ssx * ssy)
        result = jnp.where(denom > 0, sxy / denom, jnp.nan)
        result = jnp.clip(result, -1.0, 1.0)

        # Enforce minp
        result = jnp.where(nobs < minp, jnp.nan, result)

        return result

    @classmethod
    def _weighting_linear(cls, fourier_fingerprint: jnp.ndarray) -> jnp.ndarray:
        """
        Performs weighting on the given correlation matrix.
        Here, low-frequent coefficients are weighted more heavily.

        Args:
            fourier_fingerprint (jnp.ndarray): Correlation matrix
        """
        assert (
            fourier_fingerprint.shape[0] % 2 != 0
            and fourier_fingerprint.shape[1] % 2 != 0
        ), (
            "Correlation matrix must have odd dimensions. \
            Hint: use `trim` argument when calling `get_spectrum`."
        )
        assert fourier_fingerprint.shape[0] == fourier_fingerprint.shape[1], (
            "Correlation matrix must be square."
        )

        # The weight matrix produced by the previous quadrant-mirror
        # construction has a closed form: it is a "tent" sum along the
        # two axes. Concretely, with N = fourier_fingerprint.shape[0]
        # (odd) and center = N // 2,
        #     W[i, j] = u[i] + u[j]
        # where u[k] = (center - |k - center|) / (2 * center)
        # is a triangular weighting peaking at the centre (the zero
        # frequency) and decaying linearly to 0 at the spectrum edges.
        N = fourier_fingerprint.shape[0]
        center = N // 2
        k = jnp.arange(N)
        u = (center - jnp.abs(k - center)) / (2 * center)

        return fourier_fingerprint * (u[:, None] + u[None, :])

    @classmethod
    def _weighting_mean(
        cls, fourier_fingerprint: jnp.ndarray, coeffs: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Performs weighting on the given correlation matrix.
        Here, we use the product of the mean of the coefficients as weights.
        This suppresses correlations where the mean of the coefficients is near zero.

        Args:
            fourier_fingerprint (jnp.ndarray): Correlation matrix
            coeffs (jnp.ndarray): Fourier coefficients
        """
        assert fourier_fingerprint.shape[0] == fourier_fingerprint.shape[1], (
            "Correlation matrix must be square."
        )
        assert len(coeffs.shape) >= 2, (
            "Coefficient matrix must contain coefficient axes and a sample axis."
        )

        coefficient_means = jnp.abs(jnp.mean(coeffs, axis=-1))
        coefficient_means = coefficient_means.T.reshape(-1)

        assert fourier_fingerprint.shape[0] == coefficient_means.shape[0], (
            "Correlation matrix size must match the number of Fourier coefficients."
        )

        # Apply the rank-1 weight w[i] * w[j] via broadcasting instead
        # of materialising an explicit `jnp.outer` N x N intermediate.
        return (
            fourier_fingerprint
            * coefficient_means[:, None]
            * coefficient_means[None, :]
        )


class Datasets:
    @classmethod
    def generate_fourier_series(
        cls,
        random_key: random.PRNGKey,
        model: Model,
        coefficients_min: float = 0.0,
        coefficients_max: float = 1.0,
        zero_centered: bool = False,
    ) -> jnp.ndarray:
        """
        Generates the Fourier series representation of a function.
        It uses the `model.frequencies` property to retrieve the frequency
        information. This ensures that the resulting Fourier series is
        compatible with the model.

        This function is capable of generating $D$-dimensional Fourier series
        (again defined by `model.n_input_feat`).
        The highest frequency $N$ is retrieved per dimension.

        Samples of the Fourier coefficients are drawn from a uniform circle.

        Args:
            random_key (random.PRNGKey): Random number key for JAX.
            model (Model): The quantum circuit model.
            coefficients_min (float, optional): Minimum value for the coefficients.
                Defaults to 0.0.
            coefficients_max (float, optional): Maximum value for the coefficients.
                Defaults to 1.0.
            zero_centered (bool, optional): Whether to zero-center the coefficients.
                Defaults to False.

        Returns:
            jnp.ndarray: Input domain samples with shape ((N,)*D, D)
            jnp.ndarray: Fourier series values with shape ((N,)*D)
            jnp.ndarray: Fourier coefficients with shape ((N,)*D)

        """
        # TODO: the following code can be considered to
        # capturing a truly random spectrum.
        # add some constraints on the spectrum, i.e. not fully

        # Note: one key observation for understanding the following code is,
        # that instead of wrapping your head around symmetries in multi-
        # dimensional coefficient matrices, one can simply look at the flattened
        # version of such a matrix and reshape later. It just works out.

        # going from [0, 2pi] with the resolution required for highest frequency
        # permute with input dimensionality to get an n-d grid of domain samples
        # the output shape comes from the fact that want to create a "coordinate system"
        domain_samples_per_input_dim = jnp.stack(
            jnp.meshgrid(
                *[jnp.arange(0, 2 * jnp.pi, 2 * jnp.pi / d) for d in model.degree]
            )
        ).T.reshape(-1, model.n_input_feat)

        # generate the frequency indices for each dimension.
        # this will have the same shape as the domain samples
        frequencies = jnp.stack(jnp.meshgrid(*model.frequencies)).T.reshape(
            -1, model.n_input_feat
        )

        # using the frequency information, sample coefficients for each dimension
        # shape: (input_dims, n_freqs_per_input_dim // 2 + 1)

        coefficients = cls.uniform_circle(
            random_key,
            low=coefficients_min,
            high=coefficients_max,
            size=math.prod(model.degree) // 2 + 1,
        )

        # zero center (first coeff = 0)
        # we can assume the first coeff is the offset, because we're dealing
        # with a non-symmetric spectrum here
        if zero_centered:
            coefficients = coefficients.at[0].set(0.0)
        else:
            coefficients = coefficients.at[0].set(coefficients[0].real)

        # ensure symmetry (here, non_negative_ is removed!),
        # giving us the full coefficients vector
        coefficients = jnp.concat(
            [
                jnp.flip(coefficients[..., 1:]).conjugate(),
                coefficients,
            ],
            axis=-1,
        )

        # Vectorized version of $f(x) = \sum_{n=0}^{N-1} c_n * e^{i * \omega_n * x}$
        # it takes into account the input dimension, i.e. the output is a matrix
        # normalization uses the n_freqs component of the coefficients
        values = jnp.real(
            (
                jnp.exp(1j * (domain_samples_per_input_dim @ frequencies.T))
                * coefficients
            ).sum(axis=1)
            / coefficients.size
        )

        # return all the information we have
        return [
            domain_samples_per_input_dim.reshape(*model.degree, -1),
            values.reshape(model.degree),
            coefficients.reshape(model.degree),
        ]

    @classmethod
    def uniform_circle(
        cls,
        random_key: random.PRNGKey,
        size: Union[jnp.ndarray, List, int],
        low=0.0,
        high=1.0,
    ):
        """
        Random number generator for complex numbers sampled inside the unit circle

        Args:
            random_key (random.PRNGKey): Random number key for JAX.
            size (Union[jnp.ndarray, int]): Number of samples. If a 2D array is passed,
                the first dimension will be the number of dimensions.
            low (float, optional): Minimum Radius. Defaults to 0.0.
            high (float, optional): Maximum Radius. Defaults to 1.0.

        Returns
            jnp.ndarray: Array of complex numbers with shape of `size`
        """

        if isinstance(size, int):
            size = jnp.array([size])

        random_key, random_key1 = random.split(random_key)
        return jnp.sqrt(
            random.uniform(random_key, size, minval=low, maxval=high)
        ) * jnp.exp(2j * jnp.pi * random.uniform(random_key1, size))
