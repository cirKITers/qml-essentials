from __future__ import annotations
import math
from collections import defaultdict
from dataclasses import dataclass
import pennylane as qml
import pennylane.numpy as np
from pennylane.operation import Operator
import pennylane.ops.op_math as qml_op
from scipy.stats import rankdata
from functools import reduce
from typing import List, Tuple, Optional, Any, Dict, Union

from qml_essentials.model import Model

import logging

log = logging.getLogger(__name__)


class Coefficients:
    @staticmethod
    def get_spectrum(
        model: Model,
        mfs: int = 1,
        mts: int = 1,
        shift=False,
        trim=False,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts the coefficients of a given model using a FFT (np-fft).

        Note that the coefficients are complex numbers, but the imaginary part
        of the coefficients should be very close to zero, since the expectation
        values of the Pauli operators are real numbers.

        It can perform oversampling in both the frequency and time domain
        using the `mfs` and `mts` arguments.

        Args:
            model (Model): The model to sample.
            mfs (int): Multiplicator for the highest frequency. Default is 2.
            mts (int): Multiplicator for the number of time samples. Default is 1.
            shift (bool): Whether to apply np-fftshift. Default is False.
            trim (bool): Whether to remove the Nyquist frequency if spectrum is even.
                Default is False.
            kwargs (Any): Additional keyword arguments for the model function.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing the coefficients and frequencies.
        """
        kwargs.setdefault("force_mean", True)
        kwargs.setdefault("execution_type", "expval")

        coeffs, freqs = Coefficients._fourier_transform(
            model, mfs=mfs, mts=mts, **kwargs
        )

        if not np.isclose(np.sum(coeffs).imag, 0.0, rtol=1.0e-5):
            raise ValueError(
                f"Spectrum is not real. Imaginary part of coefficients is:\
                {np.sum(coeffs).imag}"
            )

        if trim:
            for ax in range(model.n_input_feat):
                if coeffs.shape[ax] % 2 == 0:
                    coeffs = np.delete(coeffs, len(coeffs) // 2, axis=ax)
                    freqs = np.delete(freqs, len(freqs) // 2, axis=ax)

        if shift:
            coeffs = np.fft.fftshift(coeffs, axes=list(range(model.n_input_feat)))
            freqs = np.fft.fftshift(freqs)

        return coeffs, freqs

    @staticmethod
    def _fourier_transform(
        model: Model, mfs: int, mts: int, **kwargs: Any
    ) -> np.ndarray:
        # Create a frequency vector with as many frequencies as model degrees,
        # oversampled by nfs
        n_freqs: int = 2 * mfs * model.degree + 1

        start, stop, step = 0, 2 * mts * np.pi, 2 * np.pi / n_freqs
        # Stretch according to the number of frequencies
        inputs: np.ndarray = np.arange(start, stop, step)

        # permute with input dimensionality
        nd_inputs = np.array(np.meshgrid(*[inputs] * model.n_input_feat)).T.reshape(
            -1, model.n_input_feat
        )

        # Output vector is not necessarily the same length as input
        outputs = model(inputs=nd_inputs, **kwargs)
        outputs = outputs.reshape(*(inputs.shape * model.n_input_feat), -1).squeeze()

        coeffs = np.fft.fftn(outputs, axes=list(range(model.n_input_feat)))

        # TODO: in the future, this should take into account that there can be a
        # different number of frequencies per dimension
        freqs = [
            np.fft.fftfreq(mts * n_freqs, 1 / n_freqs)
            for _ in range(model.n_input_feat)
        ]
        # freqs = np.fft.fftfreq(mts * n_freqs, 1 / n_freqs)

        # TODO: this could cause issues with multidim input
        # FIXME: account for different frequencies in multidim input scenarios
        # Run the fft and rearrange +
        # normalize the output (using product if multidim)
        return (
            coeffs / np.prod(outputs.shape[0 : model.n_input_feat]),
            np.array(freqs).squeeze(),
        )

    @staticmethod
    def get_psd(coeffs: np.ndarray) -> np.ndarray:
        """
        Calculates the power spectral density (PSD) from given Fourier coefficients.

        Args:
            coeffs (np.ndarray): The Fourier coefficients.

        Returns:
            np.ndarray: The power spectral density.
        """
        # TODO: if we apply trim=True in advance, this will be slightly wrong..

        def abs2(x):
            return x.real**2 + x.imag**2

        scale = 2.0 / (len(coeffs) ** 2)
        return scale * abs2(coeffs)

    @staticmethod
    def evaluate_Fourier_series(
        coefficients: np.ndarray,
        frequencies: np.ndarray,
        inputs: Union[np.ndarray, list, float],
    ) -> float:
        """
        Evaluate the function value of a Fourier series at one point.

        Args:
            coefficients (np.ndarray): Coefficients of the Fourier series.
            frequencies (np.ndarray): Corresponding frequencies.
            inputs (np.ndarray): Point at which to evaluate the function.
        Returns:
            float: The function value at the input point.
        """
        dims = len(coefficients.shape)

        if not isinstance(inputs, (np.ndarray, list)):
            inputs = [inputs]

        frequencies = np.stack(np.meshgrid(*frequencies)).T.reshape(-1, dims)
        freq_inputs = np.einsum("...j,j->...", frequencies, inputs)
        coeffs = coefficients.flatten()
        freq_inputs = freq_inputs.flatten()

        exp = 0.0
        for omega_x, c in zip(freq_inputs, coeffs):
            exp += c * np.exp(1j * omega_x)

        return np.real_if_close(exp)


class FourierTree:
    """
    Sine-cosine tree representation for the algorithm by Nemkov et al.
    This tree can be used to obtain analytical Fourier coefficients for a given
    Pauli-Clifford circuit.
    """

    class CoefficientsTreeNode:
        """
        Representation of a node in the coefficients tree for the algorithm by
        Nemkov et al.
        """

        def __init__(
            self,
            parameter_idx: Optional[int],
            observable: FourierTree.PauliOperator,
            is_sine_factor: bool,
            is_cosine_factor: bool,
            left: Optional[FourierTree.CoefficientsTreeNode] = None,
            right: Optional[FourierTree.CoefficientsTreeNode] = None,
        ):
            """
            Coefficient tree node initialisation. Each node has information about
            its creation context and it's children, i.e.:

            Args:
                parameter_idx (Optional[int]): Index of the corresp. param. index i.
                observable (FourierTree.PauliOperator): The nodes observable to
                    obtain the expectation value that contributes to the constant
                    term.
                is_sine_factor (bool): If this node belongs to a sine coefficient.
                is_cosine_factor (bool): If this node belongs to a cosine coefficient.
                left (Optional[CoefficientsTreeNode]): left child (if any).
                right (Optional[CoefficientsTreeNode]): right child (if any).
            """
            self.parameter_idx = parameter_idx

            assert not (
                is_sine_factor and is_cosine_factor
            ), "Cannot be sine and cosine at the same time"
            self.is_sine_factor = is_sine_factor
            self.is_cosine_factor = is_cosine_factor

            # If the observable does not constist of only Z and I, the
            # expectation (and therefore the constant node term) is zero
            if np.logical_or(
                observable.list_repr == 0, observable.list_repr == 1
            ).any():
                self.term = 0.0
            else:
                self.term = observable.phase

            self.left = left
            self.right = right

        def evaluate(self, parameters: list[float]) -> float:
            """
            Recursive function to evaluate the expectation of the coefficient tree,
            starting from the current node.

            Args:
                parameters (list[float]): The parameters, by which the circuit (and
                    therefore the tree) is parametrised.

            Returns:
                float: The expectation for the current node and it's children.
            """
            factor = (
                parameters[self.parameter_idx]
                if self.parameter_idx is not None
                else 1.0
            )
            if self.is_sine_factor:
                factor = 1j * np.sin(factor)
            elif self.is_cosine_factor:
                factor = np.cos(factor)
            if not (self.left or self.right):  # leaf
                return factor * self.term

            sum_children = 0.0
            if self.left:
                left = self.left.evaluate(parameters)
                sum_children = sum_children + left
            if self.right:
                right = self.right.evaluate(parameters)
                sum_children = sum_children + right

            return factor * sum_children

        def get_leafs(
            self,
            sin_list: List[int],
            cos_list: List[int],
            existing_leafs: List[FourierTree.TreeLeaf] = [],
        ) -> List[FourierTree.TreeLeaf]:
            """
            Traverse the tree starting from the current node, to obtain the tree
            leafs only.
            The leafs correspond to the terms in the sine-cosine tree
            representation that eventually are used to obtain coefficients and
            frequencies.
            Sine and cosine lists are recursively passed to the children until a
            leaf is reached (top to bottom).
            Leafs are then passed bottom to top to the caller.

            Args:
                sin_list (List[int]): Current number of sine contributions for each
                    parameter. Has the same length as the parameters, as each
                    position corresponds to one parameter.
                cos_list (List[int]):  Current number of cosine contributions for
                    each parameter. Has the same length as the parameters, as each
                    position corresponds to one parameter.
                existing_leafs (List[TreeLeaf]): Current list of leaf nodes from
                    parents.

            Returns:
                List[TreeLeaf]: Updated list of leaf nodes.
            """

            if self.is_sine_factor:
                sin_list[self.parameter_idx] += 1
            if self.is_cosine_factor:
                cos_list[self.parameter_idx] += 1

            if not (self.left or self.right):  # leaf
                if self.term != 0.0:
                    return [FourierTree.TreeLeaf(sin_list, cos_list, self.term)]
                else:
                    return []

            if self.left:
                leafs_left = self.left.get_leafs(
                    sin_list.copy(), cos_list.copy(), existing_leafs.copy()
                )
            else:
                leafs_left = []

            if self.right:
                leafs_right = self.right.get_leafs(
                    sin_list.copy(), cos_list.copy(), existing_leafs.copy()
                )
            else:
                leafs_right = []

            existing_leafs.extend(leafs_left)
            existing_leafs.extend(leafs_right)
            return existing_leafs

    @dataclass
    class TreeLeaf:
        """
        Coefficient tree leafs according to the algorithm by Nemkov et al., which
        correspond to the terms in the sine-cosine tree representation that
        eventually are used to obtain coefficients and frequencies.

        Args:
            sin_indices (List[int]): Current number of sine contributions for each
                parameter. Has the same length as the parameters, as each
                position corresponds to one parameter.
            cos_list (List[int]):  Current number of cosine contributions for
                each parameter. Has the same length as the parameters, as each
                position corresponds to one parameter.
            term (np.complex): Constant factor of the leaf, depending on the
                expectation value of the observable, and a phase.
        """

        sin_indices: List[int]
        cos_indices: List[int]
        term: np.complex128

    class PauliOperator:
        """
        Utility class for storing Pauli Rotations, the corresponding indices
        in the XY-Space (whether there is a gate with X or Y generator at a
        certain qubit) and the phase.

        Args:
            pauli (Union[Operator, np.ndarray[int]]): Pauli Rotation Operation
                or list representation
            n_qubits (int): Number of qubits in the circuit
            prev_xy_indices (Optional[np.ndarray[bool]]): X/Y indices of the
                previous Pauli sequence. Defaults to None.
            is_observable (bool): If the operator is an observable. Defaults to
                False.
            is_init (bool): If this Pauli operator is initialised the first
                time. Defaults to True.
            phase (float): Phase of the operator. Defaults to 1.0
        """

        def __init__(
            self,
            pauli: Union[Operator, np.ndarray[int]],
            n_qubits: int,
            prev_xy_indices: Optional[np.ndarray[bool]] = None,
            is_observable: bool = False,
            is_init: bool = True,
            phase: float = 1.0,
        ):
            self.is_observable = is_observable
            self.phase = phase

            if is_init:
                if not is_observable:
                    pauli = pauli.generator()[0].base
                self.list_repr = self._create_list_representation(pauli, n_qubits)
            else:
                assert isinstance(pauli, np.ndarray)
                self.list_repr = pauli

            if prev_xy_indices is None:
                prev_xy_indices = np.zeros(n_qubits, dtype=bool)
            self.xy_indices = np.logical_or(
                prev_xy_indices,
                self._compute_xy_indices(self.list_repr, rev=is_observable),
            )

        @staticmethod
        def _compute_xy_indices(
            op: np.ndarray[int], rev: bool = False
        ) -> np.ndarray[bool]:
            """
            Computes the positions of X or Y gates to an one-hot encoded boolen
            array.

            Args:
                op (np.ndarray[int]): Pauli-Operation list representation.
                rev (bool): Whether to negate the array.

            Returns:
                np.ndarray[bool]: One hot encoded boolean array.
            """
            xy_indices = (op == 0) + (op == 1)
            if rev:
                xy_indices = ~xy_indices
            return xy_indices

        @staticmethod
        def _create_list_representation(op: Operator, n_qubits: int) -> np.ndarray[int]:
            """
            Create list representation of a Pennylane Operator.
            Generally, the list representation is a list of length n_qubits,
            where at each position a Pauli Operator is encoded as such:
                I: -1
                X: 0
                Y: 1
                Z: 2

            Args:
                op (Operator): Pennylane Operator
                n_qubits (int): number of qubits in the circuit

            Returns:
                np.ndarray[int]: List representation
            """
            pauli_repr = -np.ones(n_qubits, dtype=int)
            op = op.terms()[1][0] if isinstance(op, qml_op.Prod) else op
            op = op.base if isinstance(op, qml_op.SProd) else op

            if isinstance(op, qml.PauliX):
                pauli_repr[op.wires[0]] = 0
            elif isinstance(op, qml.PauliY):
                pauli_repr[op.wires[0]] = 1
            elif isinstance(op, qml.PauliZ):
                pauli_repr[op.wires[0]] = 2
            else:
                for o in op:
                    if isinstance(o, qml.PauliX):
                        pauli_repr[o.wires[0]] = 0
                    elif isinstance(o, qml.PauliY):
                        pauli_repr[o.wires[0]] = 1
                    elif isinstance(o, qml.PauliZ):
                        pauli_repr[o.wires[0]] = 2
            return pauli_repr

        def is_commuting(self, pauli: np.ndarray[int]) -> bool:
            """
            Computes if this Pauli commutes with another Pauli operator.
            This computation is based on the fact that The commutator is zero
            if and only if the number of anticommuting single-qubit Paulis is
            even.

            Args:
                pauli (np.ndarray[int]): List representation of another Pauli

            Returns:
                bool: If the current and other Pauli are commuting.
            """
            anticommutator = np.where(
                pauli < 0,
                False,
                np.where(
                    self.list_repr < 0,
                    False,
                    np.where(self.list_repr == pauli, False, True),
                ),
            )
            return not (np.sum(anticommutator) % 2)

        def tensor(self, pauli: np.ndarray[int]) -> FourierTree.PauliOperator:
            """
            Compute tensor product between the current Pauli and a given list
            representation of another Pauli operator.

            Args:
                pauli (np.ndarray[int]): List representation of Pauli

            Returns:
                FourierTree.PauliOperator: New Pauli operator object, which
                contains the tensor product
            """
            diff = (pauli - self.list_repr + 3) % 3
            phase = np.where(
                self.list_repr < 0,
                1.0,
                np.where(
                    pauli < 0,
                    1.0,
                    np.where(
                        diff == 2,
                        1.0j,
                        np.where(diff == 1, -1.0j, 1.0),
                    ),
                ),
            )

            obs = np.where(
                self.list_repr < 0,
                pauli,
                np.where(
                    pauli < 0,
                    self.list_repr,
                    np.where(
                        diff == 2,
                        (self.list_repr + 1) % 3,
                        np.where(diff == 1, (self.list_repr + 2) % 3, -1),
                    ),
                ),
            )
            phase = self.phase * np.prod(phase)
            return FourierTree.PauliOperator(
                obs, phase=phase, n_qubits=obs.size, is_init=False, is_observable=True
            )

    def __init__(self, model: Model, inputs=1.0):
        """
        Tree initialisation, based on the Pauli-Clifford representation of a model.
        Currently, only one input feature is supported.

        **Usage**:
        ```
        # initialise a model
        model = Model(...)

        # initialise and build FourierTree
        tree = FourierTree(model)

        # get expectaion value
        exp = tree()

        # Get spectrum (for each observable, we have one list element)
        coeff_list, freq_list = tree.spectrum()
        ```

        Args:
            model (Model): The Model, for which to build the tree
            inputs (bool, optional): Possible default inputs. Defaults to 1.0.
        """
        self.model = model
        self.tree_roots = None

        if not model.as_pauli_circuit:
            model.as_pauli_circuit = True

        inputs = (
            self.model._inputs_validation(inputs)
            if inputs is not None
            else self.model._inputs_validation(1.0)
        )
        inputs.requires_grad = False

        quantum_tape = qml.workflow.construct_tape(self.model.circuit)(
            params=model.params, inputs=inputs
        )
        self.parameters = [np.squeeze(p) for p in quantum_tape.get_parameters()]
        self.input_indices = [
            i for (i, p) in enumerate(self.parameters) if not p.requires_grad
        ]

        self.observables = self._encode_observables(quantum_tape.observables)

        pauli_rot = FourierTree.PauliOperator(
            quantum_tape.operations[0],
            self.model.n_qubits,
        )
        self.pauli_rotations = [pauli_rot]
        for op in quantum_tape.operations[1:]:
            pauli_rot = FourierTree.PauliOperator(
                op, self.model.n_qubits, pauli_rot.xy_indices
            )
            self.pauli_rotations.append(pauli_rot)

        self.tree_roots = self.build()
        self.leafs: List[List[FourierTree.TreeLeaf]] = self._get_tree_leafs()

    def __call__(
        self,
        params: Optional[np.ndarray] = None,
        inputs: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Evaluates the Fourier tree via sine-cosine terms sum. This is
        equivalent to computing the expectation value of the observables with
        respect to the corresponding circuit.

        Args:
            params (Optional[np.ndarray], optional): Parameters of the model.
                Defaults to None.
            inputs (Optional[np.ndarray], optional): Inputs to the circuit.
                Defaults to None.

        Returns:
            np.ndarray: Expectation value of the tree.

        Raises:
            NotImplementedError: When using other "execution_type" as expval.
            NotImplementedError: When using "noise_params"


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
        inputs.requires_grad = False

        if kwargs.get("execution_type", "expval") != "expval":
            raise NotImplementedError(
                f'Currently, only "expval" execution type is supported when '
                f"building FourierTree. Got {kwargs.get('execution_type', 'expval')}."
            )
        if kwargs.get("noise_params", None) is not None:
            raise NotImplementedError(
                "Currently, noise is not supported when building FourierTree."
            )

        quantum_tape = qml.workflow.construct_tape(self.model.circuit)(
            params=self.model.params, inputs=inputs
        )
        self.parameters = [np.squeeze(p) for p in quantum_tape.get_parameters()]
        self.input_indices = [
            i for (i, p) in enumerate(self.parameters) if not p.requires_grad
        ]

        results = np.zeros(len(self.tree_roots))
        for i, root in enumerate(self.tree_roots):
            results[i] = np.real_if_close(root.evaluate(self.parameters))

        if kwargs.get("force_mean", False):
            return np.mean(results)
        else:
            return results

    def build(self) -> List[CoefficientsTreeNode]:
        """
        Creates the coefficient tree, i.e. it creates and initialises the tree
        nodes.
        Leafs can be obtained separately in _get_tree_leafs, once the tree is
        set up.

        Returns:
            List[CoefficientsTreeNode]: The list of root nodes (one root for
                each observable).
        """
        tree_roots = []
        pauli_rotation_idx = len(self.pauli_rotations) - 1
        for obs in self.observables:
            root = self._create_tree_node(obs, pauli_rotation_idx)
            tree_roots.append(root)
        return tree_roots

    def _encode_observables(
        self, tape_obs: List[Operator]
    ) -> List[FourierTree.PauliOperator]:
        """
        Encodes Pennylane observables from tape as FourierTree.PauliOperator
        utility objects.

        Args:
            tape_obs (List[Operator]): Pennylane tape operations

        Returns:
            List[FourierTree.PauliOperator]: List of Pauli operators
        """
        observables = []
        for obs in tape_obs:
            observables.append(
                FourierTree.PauliOperator(obs, self.model.n_qubits, is_observable=True)
            )
        return observables

    def _get_tree_leafs(self) -> List[List[TreeLeaf]]:
        """
        Obtain all Leaf Nodes with its sine- and cosine terms.

        Returns:
            List[List[TreeLeaf]]: For each observable (root), the list of leaf
                nodes.
        """
        leafs = []
        for root in self.tree_roots:
            sin_list = np.zeros(len(self.parameters), dtype=np.int32)
            cos_list = np.zeros(len(self.parameters), dtype=np.int32)
            leafs.append(root.get_leafs(sin_list, cos_list, []))
        return leafs

    def get_spectrum(
        self, force_mean: bool = False
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Computes the Fourier spectrum for the tree, consisting of the
        frequencies and its corresponding coefficinets.
        If the frag force_mean was set in the constructor, the mean coefficient
        over all observables (roots) are computed.

        Args:
            force_mean (bool, optional): Whether to average over multiple
                observables. Defaults to False.

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray]]:
                - List of frequencies, one list for each observable (root).
                - List of corresponding coefficents, one list for each
                  observable (root).
        """
        parameter_indices = [
            i for i in range(len(self.parameters)) if i not in self.input_indices
        ]

        coeffs = []
        for leafs in self.leafs:
            freq_terms = defaultdict(np.complex128)
            for leaf in leafs:
                leaf_factor, s, c = self._compute_leaf_factors(leaf, parameter_indices)

                for a in range(s + 1):
                    for b in range(c + 1):
                        comb = math.comb(s, a) * math.comb(c, b) * (-1) ** (s - a)
                        freq_terms[2 * a + 2 * b - s - c] += comb * leaf_factor

            coeffs.append(freq_terms)

        frequencies, coefficients = self._freq_terms_to_coeffs(coeffs, force_mean)
        return coefficients, frequencies

    def _freq_terms_to_coeffs(
        self, coeffs: List[Dict[int, np.ndarray]], force_mean: bool
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Given a list of dictionaries of the form:
        [
            {
                freq_obs1_1: coeff1,
                freq_obs1_2: coeff2,
                ...
             },
            {
                freq_obs2_1: coeff3,
                freq_obs2_2: coeff4,
                ...
             }
            ...
        ],
        Compute two separate lists of frequencies and coefficients.
        such that:
        freqs: [
                [freq_obs1_1, freq_obs1_1, ...],
                [freq_obs2_1, freq_obs2_1, ...],
                ...
        ]
        coeffs: [
                [coeff1, coeff2, ...],
                [coeff3, coeff4, ...],
                ...
        ]

        If force_mean is set length of the resulting frequency and coefficent
        list is 1.

        Args:
            coeffs (List[Dict[int, np.ndarray]]): Frequency->Coefficients
                dictionary list, one dict for each observable (root).
            force_mean (bool): Whether to average coefficients over multiple
                observables.

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray]]:
                - List of frequencies, one list for each observable (root).
                - List of corresponding coefficents, one list for each
                  observable (root).
        """
        frequencies = []
        coefficients = []
        if force_mean:
            all_freqs = sorted(set([f for c in coeffs for f in c.keys()]))
            coefficients.append(
                np.array([np.mean([c.get(f, 0.0) for c in coeffs]) for f in all_freqs])
            )
            frequencies.append(np.array(all_freqs))
        else:
            for freq_terms in coeffs:
                freq_terms = dict(sorted(freq_terms.items()))
                frequencies.append(np.array(list(freq_terms.keys())))
                coefficients.append(np.array(list(freq_terms.values())))
        return frequencies, coefficients

    def _compute_leaf_factors(
        self, leaf: TreeLeaf, parameter_indices: List[int]
    ) -> Tuple[float, int, int]:
        """
        Computes the constant coefficient factor for each leaf.
        Additionally sine and cosine contributions of the input parameters for
        this leaf are returned, which are required to obtain the corresponding
        frequencies.

        Args:
            leaf (TreeLeaf): The leaf for which to compute the factor.
            parameter_indices (List[int]): Variational parameter indices.

        Returns:
            Tuple[float, int, int]:
                - float: the constant factor for the leaf
                - int: number of sine contributions of the input
                - int: number of cosine contributions of the input
        """
        leaf_factor = 1.0
        for i in parameter_indices:
            interm_factor = (
                np.cos(self.parameters[i]) ** leaf.cos_indices[i]
                * (1j * np.sin(self.parameters[i])) ** leaf.sin_indices[i]
            )
            leaf_factor = leaf_factor * interm_factor

        # Get number of sine and cosine factors to which the input contributes
        c = np.sum([leaf.cos_indices[k] for k in self.input_indices], dtype=np.int32)
        s = np.sum([leaf.sin_indices[k] for k in self.input_indices], dtype=np.int32)

        leaf_factor = leaf.term * leaf_factor * 0.5 ** (s + c)

        return leaf_factor, s, c

    def _early_stopping_possible(
        self, pauli_rotation_idx: int, observable: FourierTree.PauliOperator
    ):
        """
        Checks if a node for an observable can be discarded as all expecation
        values that can result through further branching are zero.
        The method is mentioned in the paper by Nemkov et al.: If the one-hot
        encoded indices for X/Y operations in the Pauli-rotation generators are
        a basis for that of the observable, the node must be processed further.
        If not, it can be discarded.

        Args:
            pauli_rotation_idx (int): Index of remaining Pauli rotation gates.
                Gates itself are attributes of the class.
            observable (FourierTree.PauliOperator): Current observable
        """
        xy_indices_obs = np.logical_or(
            observable.xy_indices, self.pauli_rotations[pauli_rotation_idx].xy_indices
        ).all()

        return not xy_indices_obs

    def _create_tree_node(
        self,
        observable: FourierTree.PauliOperator,
        pauli_rotation_idx: int,
        parameter_idx: Optional[int] = None,
        is_sine: bool = False,
        is_cosine: bool = False,
    ) -> Optional[CoefficientsTreeNode]:
        """
        Builds the Fourier-Tree according to the algorithm by Nemkov et al.

        Args:
            observable (FourierTree.PauliOperator): Current observable
            pauli_rotation_idx (int): Index of remaining Pauli rotation gates.
                Gates itself are attributes of the class.
            parameter_idx (Optional[int]): Index of the current parameter.
                Parameters itself are attributes of the class.
            is_sine (bool): If the current node is a sine (left) node.
            is_cosine (bool): If the current node is a cosine (right) node.

        Returns:
            Optional[CoefficientsTreeNode]: The resulting node. Children are set
            recursively. The top level receives the tree root.
        """
        if self._early_stopping_possible(pauli_rotation_idx, observable):
            return None

        # remove commuting paulis
        while pauli_rotation_idx >= 0:
            last_pauli = self.pauli_rotations[pauli_rotation_idx]
            if not observable.is_commuting(last_pauli.list_repr):
                break
            pauli_rotation_idx -= 1
        else:  # leaf
            return FourierTree.CoefficientsTreeNode(
                parameter_idx, observable, is_sine, is_cosine
            )

        last_pauli = self.pauli_rotations[pauli_rotation_idx]

        left = self._create_tree_node(
            observable,
            pauli_rotation_idx - 1,
            pauli_rotation_idx,
            is_cosine=True,
        )

        next_observable = self._create_new_observable(last_pauli.list_repr, observable)
        right = self._create_tree_node(
            next_observable,
            pauli_rotation_idx - 1,
            pauli_rotation_idx,
            is_sine=True,
        )

        return FourierTree.CoefficientsTreeNode(
            parameter_idx,
            observable,
            is_sine,
            is_cosine,
            left,
            right,
        )

    def _create_new_observable(
        self, pauli: np.ndarray[int], observable: FourierTree.PauliOperator
    ) -> FourierTree.PauliOperator:
        """
        Utility function to obtain the new observable for a tree node, if the
        last Pauli and the observable do not commute.

        Args:
            pauli (np.ndarray[int]): The int array representation of the last
                Pauli rotation in the operation sequence.
            observable (FourierTree.PauliOperator): The current observable.

        Returns:
            FourierTree.PauliOperator: The new observable.
        """
        observable = observable.tensor(pauli)
        return observable


class FCC:
    @staticmethod
    def get_fcc(
        model: Model,
        n_samples: int,
        seed: int,
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
            1. Calculating the coefficients (using `n_samples` and `seed`)
            2. Correlating the result from 1) using `method`
            3. Weighting the correlation matrix (if `weight` is True)
            4. Remove redundancies
        2. What is done in `calculate_fcc`:
            1. Absolute of the fingerprint
            2. Average

        Args:
            model (Model): The QFM model
            n_samples (int): Number of samples to calculate average of coefficients
            seed (int): Seed to initialize random parameters
            method (Optional[str], optional): Correlation method. Defaults to "pearson".
            scale (Optional[bool], optional): Whether to scale the number of samples.
                Defaults to False.
            weight (Optional[bool], optional): Whether to weight the correlation matrix.
                Defaults to False.
            trim_redundant (Optional[bool], optional): Whether to remove redundant
                correlations. Defaults to False.
            **kwargs: Additional keyword arguments for the model function.

        Returns:
            float: The FCC
        """
        fourier_fingerprint, freqs = FCC.get_fourier_fingerprint(
            model,
            n_samples,
            seed,
            method,
            scale,
            weight,
            trim_redundant=trim_redundant,
            **kwargs,
        )

        return FCC.calculate_fcc(fourier_fingerprint)

    def get_fourier_fingerprint(
        model: Model,
        n_samples: int,
        seed: int,
        method: Optional[str] = "pearson",
        scale: Optional[bool] = False,
        weight: Optional[bool] = False,
        trim_redundant: Optional[bool] = True,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Shortcut method to get just the fourier fingerprint.
        This includes
        1. Calculating the coefficients (using `n_samples` and `seed`)
        2. Correlating the result from 1) using `method`
        3. Weighting the correlation matrix (if `weight` is True)
        4. Remove redundancies (if `trim_redundant` is True)

        If trimming is enabled, `freqs` will be the trimmed down frequency vector.

        Args:
            model (Model): The QFM model
            n_samples (int): Number of samples to calculate average of coefficients
            seed (int): Seed to initialize random parameters
            method (Optional[str], optional): Correlation method. Defaults to "pearson".
            scale (Optional[bool], optional): Whether to scale the number of samples.
                Defaults to False.
            weight (Optional[bool], optional): Whether to weight the correlation matrix.
                Defaults to False.
            trim_redundant (Optional[bool], optional): Whether to remove redundant
                correlations. Defaults to True.
            **kwargs: Additional keyword arguments for the model function.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The fourier fingerprint and the frequency indices
        """
        _, coeffs, freqs = FCC._calculate_coefficients(
            model, n_samples, seed, scale, **kwargs
        )
        fourier_fingerprint = FCC._correlate(coeffs.transpose(), method=method)

        # perform weighting if requested
        fourier_fingerprint = (
            FCC._weighting(fourier_fingerprint) if weight else fourier_fingerprint
        )

        if trim_redundant:
            mask = FCC._calculate_mask(freqs)
            freqs = mask * freqs

            # apply the mask on the fingerprint
            fourier_fingerprint = mask * fourier_fingerprint

            row_mask = np.any(np.isfinite(fourier_fingerprint), axis=1)
            col_mask = np.any(np.isfinite(fourier_fingerprint), axis=0)

            fourier_fingerprint = fourier_fingerprint[row_mask][:, col_mask]
            freqs = freqs[row_mask][:, col_mask]

        return fourier_fingerprint, freqs

    @staticmethod
    def calculate_fcc(
        fourier_fingerprint: np.ndarray,
    ) -> float:
        """
        Method to calculate the FCC based on an existing correlation matrix.
        Calculate absolute and then the average over this matrix.
        The Fingerprint can be obtained via `get_fourier_fingerprint`

        Args:
            coeff_coeff_correlation (np.ndarray): Correlation matrix of coefficients
        Returns:
            float: The FCC
        """
        # apply the mask on the fingerprint
        return np.nanmean(np.abs(fourier_fingerprint))

    def _calculate_mask(freqs: np.ndarray) -> np.ndarray:
        """
        Method to calculate a mask filtering out redundant elements
        of the Fourier correlation matrix, based on the provided frequency vector.
        It does so by 'simulating' the operations that would be performed
        by `_correlate`.

        Args:
            freqs (np.ndarray): Array of frequencies

        Returns:
            np.ndarray: The mask
        """
        # TODO: this part can be heavily optimized, by e.g. using a "positive_only"
        # flag when calculating the coefficients. However this would change the numerical
        # values (while the order should be still the same).

        # disregard all the negativ frequencies
        freqs[freqs < 0] = np.nan
        # compute the outer product of the frequency vectors for arbitrary dimensions
        # or just use the existing frequency vector if it is 1D
        nd_freqs = (
            reduce(np.multiply, np.ix_(*freqs)) if len(freqs.shape) > 1 else freqs
        )
        # TODO: could prevent this if we're not using .squeeze()..

        # "simulate" what would happen on correlating the coefficients
        corr_freqs = np.outer(nd_freqs, nd_freqs)
        # mask all frequencies that are nan now (i.e. all correlations with a negative frequency component)
        corr_mask = np.where(np.isnan(corr_freqs), corr_freqs, 1)
        # from this, disregard all the other redundant correlations (i.e. c_0_1 = c_1_0)
        corr_mask[np.triu_indices(corr_mask.shape[0], 0)] = np.nan

        return corr_mask

    @staticmethod
    def _calculate_coefficients(
        model: Model,
        n_samples: int,
        seed: int,
        scale: bool = False,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates the Fourier coefficients of a given model using `n_samples` and `seed`.
        Optionally, `noise_params` can be passed to perform noisy simulation.

        Args:
            model (Model): The QFM model
            n_samples (int): Number of samples to calculate average of coefficients
            seed (int): Seed to initialize random parameters
            scale (bool, optional): Whether to scale the number of samples.
                Defaults to False.
            **kwargs: Additional keyword arguments for the model function.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Parameters and Coefficients of size NxK
        """
        if n_samples > 0:
            if scale:
                total_samples = int(
                    np.power(2, model.n_qubits) * n_samples * model.n_input_feat
                )
                log.info(f"Using {total_samples} samples.")
            else:
                total_samples = n_samples
            rng = np.random.default_rng(seed)
            model.initialize_params(rng=rng, repeat=total_samples)
        else:
            total_samples = 1

        coeffs, freqs = Coefficients.get_spectrum(
            model, shift=True, trim=True, **kwargs
        )

        return model.params, coeffs, freqs

    @staticmethod
    def _correlate(mat: np.ndarray, method: str = "pearson") -> np.ndarray:
        """
        Correlates two arrays using `method`.
        Currently, `pearson` and `spearman` are supported.

        Args:
            mat (np.ndarray): Array of shape (N, K)
            method (str, optional): Correlation method. Defaults to "pearson".

        Raises:
            ValueError: If the method is not supported.

        Returns:
            np.ndarray: Correlation matrix of `a` and `b`.
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
            result = FCC._pearson(mat.reshape(mat.shape[0], -1))
            # result = FCC._pearson(mat.reshape(mat.shape[-1], -1, order="F"))
        elif method == "spearman":
            result = FCC._spearman(mat.reshape(mat.shape[0], -1))
            # result = FCC._spearman(mat.reshape(mat.shape[-1], -1, order="F"))
        else:
            raise ValueError(
                f"Unknown correlation method: {method}. \
                             Must be 'pearson' or 'spearman'."
            )

        return result

    @staticmethod
    def _pearson(
        mat: np.ndarray, cov: Optional[bool] = False, minp: Optional[int] = 1
    ) -> np.ndarray:
        """
        Based on Pandas correlation method as implemented here:
        https://github.com/pandas-dev/pandas/blob/main/pandas/_libs/algos.pyx

        Compute Pearson correlation between columns of `mat`,
        permitting missing values (NaN or ±Inf).

        Args:
            mat : array_like, shape (N, K)
                Input data.
            minp : int, optional
                Minimum number of paired observations required to form a correlation.
                If the number of valid pairs for (i, j) is < minp, the result is NaN.

        Returns:
            corr : ndarray, shape (K, K)
                Pearson correlation matrix.
        """

        mat = np.asarray(mat, dtype=np.float64)
        N, K = mat.shape

        # pre‐compute finite‐mask
        mask = np.isfinite(mat)

        # output
        result = np.empty((K, K), dtype=np.float64)

        # TODO: optimize in future iterations
        # loop over column‐pairs
        for i in range(K):
            for j in range(i + 1):
                # find rows where both columns are finite
                m = mask[:, i] & mask[:, j]
                n = np.count_nonzero(m)
                if n < minp:
                    # too few pairs
                    value = np.nan
                else:
                    x = mat[m, i]
                    y = mat[m, j]

                    # compute means
                    mean_x = x.mean()
                    mean_y = y.mean()

                    # demeaned data
                    dx = x - mean_x
                    dy = y - mean_y

                    # sum of squares and cross‐prod
                    ssx = np.dot(dx, dx)
                    ssy = np.dot(dy, dy)
                    cxy = np.dot(dx, dy)

                    if cov:
                        # sample covariance (denominator n−1)
                        value = cxy / (n - 1) if n > 1 else np.nan
                    else:
                        # Pearson r = cov / (σx σy)
                        denom = np.sqrt(ssx * ssy)
                        if denom == 0.0:
                            value = np.nan
                        else:
                            value = cxy / denom
                            # clip numerical drift
                            if value > 1.0:
                                value = 1.0
                            elif value < -1.0:
                                value = -1.0

                result[i, j] = result[j, i] = value

        return result

    def _spearman(mat: np.ndarray, minp: Optional[int] = 1) -> np.ndarray:
        """
        Based on Pandas correlation method as implemented here:
        https://github.com/pandas-dev/pandas/blob/main/pandas/_libs/algos.pyx

        Compute Spearman correlation between columns of `mat`,
        permitting missing values (NaN or ±Inf).

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
        N, K = mat.shape
        # trivial all-NaN answer if too few rows
        if N < minp:
            return np.full((K, K), np.nan, dtype=float)

        # mask of finite entries
        mask = np.isfinite(mat)  # shape (N, K), dtype=bool

        # precompute ranks column-wise ignoring NaNs
        ranks = np.full((N, K), np.nan, dtype=float)
        for j in range(K):
            valid = mask[:, j]
            if valid.any():
                # rankdata by default gives average ranks for ties
                ranks[valid, j] = rankdata(mat[valid, j], method="average")

        # allocate result
        result = np.empty((K, K), dtype=float)

        # TODO: optimize in future iterations
        # loop lower triangle (including diagonal)
        for i in range(K):
            for j in range(i + 1):
                # find rows where both columns are finite
                valid = mask[:, i] & mask[:, j]
                nobs = valid.sum()

                if nobs < minp:
                    rho = np.nan
                else:
                    xi = ranks[valid, i]
                    yj = ranks[valid, j]
                    # subtract means
                    xi = xi - xi.mean()
                    yj = yj - yj.mean()
                    num = np.dot(xi, yj)
                    den = np.sqrt(np.dot(xi, xi) * np.dot(yj, yj))
                    rho = num / den if den > 0 else np.nan

                result[i, j] = rho
                result[j, i] = rho

        return result

    @staticmethod
    def _weighting(correlation: np.ndarray) -> np.ndarray:
        """
        Performs weighting on the given correlation matrix.
        Here, low-frequent coefficients are weighted more heavily.

        Args:
            correlation (np.ndarray): Correlation matrix
        """
        raise NotImplementedError("Weighting method is not implemented")
