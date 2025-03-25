from __future__ import annotations
import numpy as np
import math
from collections import defaultdict
from dataclasses import dataclass
import pennylane as qml
from pennylane.operation import Operator
import pennylane.ops.op_math as qml_op
from typing import List, Tuple, Optional, Any, Dict

from qml_essentials.model import Model


class Coefficients:

    @staticmethod
    def get_spectrum(
        model: Model,
        mfs: int = 1,
        mts: int = 1,
        shift=False,
        trim=False,
        **kwargs,
    ) -> np.ndarray:
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
            np.ndarray: The sampled Fourier coefficients.
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

        if trim and coeffs.size % 2 == 0:
            coeffs = np.delete(coeffs, len(coeffs) // 2)
            freqs = np.delete(freqs, len(freqs) // 2)

        if shift:
            return np.fft.fftshift(coeffs), np.fft.fftshift(freqs)
        else:
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
        inputs: np.ndarray = np.arange(start, stop, step) % (2 * np.pi)

        # permute with input dimensionality
        nd_inputs = np.array(
            np.meshgrid(*[inputs] * model.n_input_feat)
        ).T.reshape(-1, model.n_input_feat)

        # Output vector is not necessarily the same length as input
        outputs = model(inputs=nd_inputs, **kwargs).reshape(
            inputs.shape * model.n_input_feat
        )

        coeffs = np.fft.fftn(outputs)

        assert (
            mts * n_freqs,
        ) * model.n_input_feat == coeffs.shape, f"Expected shape\
            {(mts * n_freqs,) * model.n_input_feat} but got {coeffs.shape}"

        freqs = np.fft.fftfreq(mts * n_freqs, 1 / n_freqs)

        # Run the fft and rearrange + normalize the output
        return coeffs / outputs.size, freqs

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
        input: float,
        frequencies: Optional[np.ndarray] = None,
    ) -> float:
        """
        Evaluate the function value of a Fourier series at one point.

        Args:
            coefficients (np.ndarray): Coefficients of the Fourier series.
            input (float): Point at which to evaluate the function.
            frequencies (Optional[np.ndarray]): Corresponding frequencies in
                the form [-n_freq, ..., 0, ..., n_freq]. If None, the number of
                coefficients is to obtain sequential frequencies.

        Returns:
            float: The function value at the input point.
        """
        n_freq = len(coefficients) // 2
        if frequencies is None:
            frequencies = np.arange(-n_freq, n_freq + 1)

        exp = 0.0
        for omega, c in zip(frequencies, coefficients):
            exp += c * np.exp(1j * omega * input)

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
            observable: Operator,
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
                observable (Operator): The nodes observable to obtain the
                    expectation value that contributes to the constant term.
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

            if isinstance(observable, qml_op.SProd):
                term = observable.terms()[0][0]
                observable = observable.terms()[1][0]
            else:
                term = 1.0

            # If the observable does not constist of only Z and I, the
            # expectation (and therefore the constant node term) is zero
            if (
                isinstance(observable, qml_op.Prod)
                and any([isinstance(p, (qml.X, qml.Y)) for p in observable])
                or isinstance(observable, (qml.PauliX, qml.PauliY))
            ):
                self.term = 0.0
            else:
                self.term = term

            self.observable = observable

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

            Attributes:
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

        self.observables = quantum_tape.observables
        self.pauli_rotations = quantum_tape.operations

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
                f'building FourierTree. Got {kwargs.get("execution_type", "expval")}.'
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
        for obs in self.observables:
            pauli_rotation_indices = np.arange(
                len(self.pauli_rotations), dtype=np.int16
            )
            root = self._create_tree_node(obs, pauli_rotation_indices)
            tree_roots.append(root)
        return tree_roots

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
            force_mean (bool, optional): Whether to average coefficients over
                multiple observables. Defaults to False.

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

    def _create_tree_node(
        self,
        observable: Operator,
        pauli_rotation_indices: List[int],
        parameter_idx: Optional[int] = None,
        is_sine: bool = False,
        is_cosine: bool = False,
    ) -> CoefficientsTreeNode:
        """
        Builds the Fourier-Tree according to the algorithm by Nemkov et al.

        Args:
            observable (Operator): Current observable
            pauli_rotation_indices (List[int]): Indices of remaining Pauli
                rotation gates. Gates itself are attributes of the class.
            parameter_idx (Optional[int]): Index of the current parameter.
                Parameters itself are attributes of the class.
            is_sine (bool): If the current node is a sine (left) node.
            is_cosine (bool): If the current node is a cosine (right) node.

        Returns:
            CoefficientsTreeNode: The resulting node. Children are set
            recursively. The top level receives the tree root.
        """

        # remove commuting paulis
        idx = len(pauli_rotation_indices) - 1
        while idx >= 0:
            last_pauli = self.pauli_rotations[pauli_rotation_indices[idx]]
            if not qml.is_commuting(last_pauli.generator(), observable):
                break
            idx -= 1

        if idx < 0:  # leaf
            return FourierTree.CoefficientsTreeNode(
                parameter_idx, observable, is_sine, is_cosine
            )

        next_pauli_rotation_indices = pauli_rotation_indices[:idx]
        last_pauli_idx = pauli_rotation_indices[idx]
        last_pauli = self.pauli_rotations[last_pauli_idx]

        left = self._create_tree_node(
            observable,
            next_pauli_rotation_indices,
            last_pauli_idx,
            is_cosine=True,
        )

        next_observable = self._create_new_observable(
            last_pauli.generator(), observable
        )
        right = self._create_tree_node(
            next_observable,
            next_pauli_rotation_indices,
            last_pauli_idx,
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

    def _create_new_observable(self, pauli: Operator, observable: Operator) -> Operator:
        """
        Utility function to obtain the new observable for a tree node, if the
        last Pauli and the observable do not commute.

        Args:
            pauli (Operator): The generator of the last Pauli rotation in the
            operation sequence.
            observable (Operator): The current observable.

        Returns:
            Operator: The new observable.
        """

        pauli = pauli[0] / pauli.coeffs[0]  # ignore coefficients of generator
        obs = pauli @ observable
        obs = qml.simplify(obs)

        return obs
