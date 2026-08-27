import torch
from collections import OrderedDict
from torch import Tensor
from torch.utils.weak import WeakTensorKeyDictionary
from typing import Tuple, NamedTuple, Optional
from . import utils


class SpeciesEnergies(NamedTuple):
    species: Tensor
    energies: Tensor


class SpeciesCoordinates(NamedTuple):
    species: Tensor
    coordinates: Tensor


class ANIModel(torch.nn.ModuleDict):
    """ANI model that compute energies from species and AEVs.

    Different atom types might have different modules, when computing
    energies, for each atom, the module for its corresponding atom type will
    be applied to its AEV, after that, outputs of modules will be reduced along
    different atoms to obtain molecular energies.

    .. warning::

        The species must be indexed in 0, 1, 2, 3, ..., not the element
        index in periodic table. Check :class:`torchani.SpeciesConverter`
        if you want periodic table indexing.

    .. note:: The resulting energies are in Hartree.

    Arguments:
        modules (:class:`collections.abc.Sequence`): Modules for each atom
            types. Atom types are distinguished by their order in
            :attr:`modules`, which means, for example ``modules[i]`` must be
            the module for atom type ``i``. Different atom types can share a
            module by putting the same reference in :attr:`modules`.
    """

    @staticmethod
    def ensureOrderedDict(modules):
        if isinstance(modules, OrderedDict):
            return modules
        od = OrderedDict()
        for i, m in enumerate(modules):
            od[str(i)] = m
        return od

    def __init__(self, modules):
        super().__init__(self.ensureOrderedDict(modules))
        # Per-element gather indices depend only on the species tensor, not
        # on aev/coordinates. In MD (torchanipbe0.md) and geometry
        # optimization the same species tensor object is reused unchanged
        # across every step/iteration, but `(species == i).nonzero()` is a
        # synchronizing op (its output shape is data-dependent, so PyTorch
        # must block until the GPU finishes to size it) -- recomputing it
        # every call serializes the whole forward pass through the host on
        # every element of every ensemble member, which dominates wall time
        # for small/medium systems where the actual matmuls are cheap.
        # Cache the indices keyed by the species tensor's identity (not its
        # contents) so unchanged-species calls skip the sync entirely; the
        # weak key means the entry is dropped automatically once that
        # species tensor is garbage collected, so a distinct species tensor
        # can never collide with a stale cache entry.
        self._species_index_cache: "WeakTensorKeyDictionary" = WeakTensorKeyDictionary()

    def forward(self, species_aev: Tuple[Tensor, Tensor],  # type: ignore
                cell: Optional[Tensor] = None,
                pbc: Optional[Tensor] = None) -> SpeciesEnergies:
        species, aev = species_aev
        assert species.shape == aev.shape[:-1]

        atomic_energies = self._atomic_energies((species, aev))
        return SpeciesEnergies(species, torch.sum(atomic_energies, dim=1))

    @torch.jit.export
    def _atomic_energies(self, species_aev: Tuple[Tensor, Tensor]) -> Tensor:
        # Obtain the atomic energies associated with a given tensor of AEV's
        species, aev = species_aev
        # print('species', species)
        assert species.shape == aev.shape[:-1]
        species_ = species.flatten()
        aev = aev.flatten(0, 1)

        output = aev.new_zeros(species_.shape)

        scripting = torch.jit.is_scripting() or torch.compiler.is_compiling()
        indices = None if scripting else self._species_index_cache.get(species)
        if indices is None:
            indices = [(species_ == i).nonzero().flatten() for i in range(len(self))]
            if not scripting:
                self._species_index_cache[species] = indices

        for i, m in enumerate(self.values()):
            midx = indices[i]
            if midx.shape[0] > 0:
                input_ = aev.index_select(0, midx)
                output.index_copy_(0, midx, m(input_).flatten())
        output = output.view_as(species)
        return output


class Ensemble(torch.nn.ModuleList):
    """Compute the average output of an ensemble of modules."""

    def __init__(self, modules):
        super().__init__(modules)
        self.size = len(modules)

    def forward(self, species_input: Tuple[Tensor, Tensor],  # type: ignore
                cell: Optional[Tensor] = None,
                pbc: Optional[Tensor] = None) -> SpeciesEnergies:
        sum_ = 0
        for x in self:
            sum_ += x(species_input)[1]
        species, _ = species_input
        return SpeciesEnergies(species, sum_ / self.size)


class StackedElementMLP(torch.nn.Module):
    """A group of ``n_group`` structurally-identical per-element MLPs (same
    Linear/CELU layer shapes, different trained weights -- e.g. one element's
    network from each member of an :class:`Ensemble`, or several distinct
    per-element networks that happen to share an architecture) evaluated as
    one batched matmul per layer instead of ``n_group`` separate small
    ``nn.Sequential`` calls.

    All members receive the *same* input (they're alternative predictors for
    the same atoms, not a batch of different inputs), so this only pays for
    ``n_layers`` kernel launches total instead of ``n_group * n_layers``.
    Returns the unreduced ``(n_group, n_atoms)`` stack; the caller decides
    how to combine the group dimension (average, for an ensemble; or keep
    each index as a distinct output, for e.g. the m1/m2/m3/v dispersion
    coefficient networks).
    """

    def __init__(self, member_sequentials):
        super().__init__()
        assert len(member_sequentials) > 0
        ref = member_sequentials[0]
        for m in member_sequentials:
            assert len(m) == len(ref), \
                "StackedElementMLP requires identical architectures across the group"

        ops = []
        weights = []
        biases = []
        for layer_idx, layer in enumerate(ref):
            if isinstance(layer, torch.nn.Linear):
                for m in member_sequentials:
                    other = m[layer_idx]
                    assert isinstance(other, torch.nn.Linear) and \
                        other.weight.shape == layer.weight.shape, \
                        "StackedElementMLP requires identical architectures across the group"
                weights.append(torch.nn.Parameter(
                    torch.stack([m[layer_idx].weight for m in member_sequentials], dim=0)))
                biases.append(torch.nn.Parameter(
                    torch.stack([m[layer_idx].bias for m in member_sequentials], dim=0)))
                ops.append('linear')
            elif isinstance(layer, torch.nn.CELU):
                for m in member_sequentials:
                    other = m[layer_idx]
                    assert isinstance(other, torch.nn.CELU) and other.alpha == layer.alpha
                ops.append(('celu', layer.alpha))
            else:
                raise NotImplementedError(
                    f'StackedElementMLP only supports Linear/CELU layers, got {type(layer)}')

        self.ops = ops
        self.weights = torch.nn.ParameterList(weights)
        self.biases = torch.nn.ParameterList(biases)
        self.n_group = len(member_sequentials)

    def forward(self, x: Tensor) -> Tensor:
        # x: (n_atoms, in_features), shared across the group.
        h = x.unsqueeze(0)  # (1, n_atoms, in_features), broadcasts over the group dim
        wi = 0
        for op in self.ops:
            if op == 'linear':
                W = self.weights[wi]
                B = self.biases[wi]
                h = torch.matmul(h, W.transpose(1, 2)) + B.unsqueeze(1)
                wi += 1
            else:
                _, alpha = op
                h = torch.celu(h, alpha=alpha)
        return h.squeeze(-1)  # (n_group, n_atoms)


class BatchedElementGroup(torch.nn.Module):
    """Per-element dispatch (see :meth:`ANIModel._atomic_energies`) across a
    group of ``n_group`` structurally-identical per-element ensembles (see
    :class:`StackedElementMLP`). Uses the same species-identity index cache
    as :class:`ANIModel` -- species is constant for the whole lifetime of an
    MD run, so the ``nonzero()``-based per-element atom selection only needs
    to be computed once, not on every call.

    Returns the unreduced ``(n_group,) + species.shape`` tensor.
    """

    def __init__(self, per_element_members):
        # per_element_members: List[List[nn.Sequential]], outer index is the
        # element/species index, inner index is the group member for that
        # element (must be the same group, in the same order, for every
        # element).
        super().__init__()
        n_group = len(per_element_members[0])
        for members in per_element_members:
            assert len(members) == n_group, \
                "every element must have the same number of group members"
        self.n_group = n_group
        self.mlps = torch.nn.ModuleList(
            [StackedElementMLP(members) for members in per_element_members])
        self._species_index_cache: "WeakTensorKeyDictionary" = WeakTensorKeyDictionary()

    def forward(self, species: Tensor, aev: Tensor) -> Tensor:
        species_ = species.flatten()
        aev_flat = aev.flatten(0, 1)

        scripting = torch.jit.is_scripting() or torch.compiler.is_compiling()
        indices = None if scripting else self._species_index_cache.get(species)
        if indices is None:
            indices = [(species_ == i).nonzero().flatten() for i in range(len(self.mlps))]
            if not scripting:
                self._species_index_cache[species] = indices

        output = aev_flat.new_zeros((self.n_group,) + species_.shape)
        for i, mlp in enumerate(self.mlps):
            midx = indices[i]
            if midx.shape[0] > 0:
                input_ = aev_flat.index_select(0, midx)
                output.index_copy_(1, midx, mlp(input_))
        return output.view((self.n_group,) + species.shape)


class BatchedEnsemble(torch.nn.Module):
    """Drop-in replacement for :class:`Ensemble` (of :class:`ANIModel`
    members) that evaluates every ensemble member for each element as one
    batched matmul (:class:`StackedElementMLP`) instead of ``n_ensemble``
    separate per-element ``nn.Sequential`` calls -- see
    :func:`enable_batched_ensemble`. All ensemble members must share the
    same per-element architecture (true for every ``BuiltinEnsemble*`` model
    in this package: members differ only in trained weights).

    Loses :meth:`BuiltinEnsemble.__getitem__`/``atomic_energies(average=False)``
    (single-member access), since members are no longer stored as separate
    submodules after conversion; not used anywhere in this package outside
    those two methods.
    """

    def __init__(self, ensemble: Ensemble):
        super().__init__()
        n_members = len(ensemble)
        n_elements = len(ensemble[0])
        per_element_members = [
            [ensemble[m][str(i)] for m in range(n_members)] for i in range(n_elements)]
        self.group = BatchedElementGroup(per_element_members)
        self.size = n_members

    def forward(self, species_input: Tuple[Tensor, Tensor],  # type: ignore
                cell: Optional[Tensor] = None,
                pbc: Optional[Tensor] = None) -> SpeciesEnergies:
        species, aev = species_input
        assert species.shape == aev.shape[:-1]
        per_member_atomic = self.group(species, aev)  # (n_ensemble,) + species.shape
        atomic_energies = per_member_atomic.mean(dim=0)  # ensemble average, per atom
        return SpeciesEnergies(species, torch.sum(atomic_energies, dim=1))


def enable_batched_ensemble(model: torch.nn.Module) -> int:
    """Replace every :class:`Ensemble` reachable from ``model`` with an
    equivalent :class:`BatchedEnsemble`, in place. Mathematically identical
    (up to float32 summation-order rounding): the average over ensemble
    members and the sum over atoms are both linear and commute, so
    averaging per-atom first and summing after (what this does) equals
    summing per member and averaging after (what :class:`Ensemble` does).

    Cuts kernel-launch count for the ANI ensemble roughly ``n_ensemble``-fold
    (e.g. 8x for the *_2x models' 8-member ensembles), which matters for
    small/medium non-periodic MD where per-step wall time is dominated by
    the sheer number of small kernel launches rather than by compute.

    Returns the number of :class:`Ensemble` instances replaced.
    """
    targets = [name for name, module in model.named_modules() if isinstance(module, Ensemble)]
    for name in targets:
        *parent_path, attr = name.split('.')
        parent = model
        for p in parent_path:
            parent = getattr(parent, p)
        setattr(parent, attr, BatchedEnsemble(getattr(parent, attr)))
    return len(targets)


class Sequential(torch.nn.ModuleList):
    """Modified Sequential module that accept Tuple type as input"""

    def __init__(self, *modules):
        super().__init__(modules)

    def forward(self, input_: Tuple[Tensor, Tensor],  # type: ignore
                cell: Optional[Tensor] = None,
                pbc: Optional[Tensor] = None):
        for module in self:
            input_ = module(input_, cell=cell, pbc=pbc)
        return input_


class Gaussian(torch.nn.Module):
    """Gaussian activation"""
    def forward(self, x: Tensor) -> Tensor:
        return torch.exp(- x * x)


class SpeciesConverter(torch.nn.Module):
    """Converts tensors with species labeled as atomic numbers into tensors
    labeled with internal torchani indices according to a custom ordering
    scheme. It takes a custom species ordering as initialization parameter. If
    the class is initialized with ['H', 'C', 'N', 'O'] for example, it will
    convert a tensor [1, 1, 6, 7, 1, 8] into a tensor [0, 0, 1, 2, 0, 3]

    Arguments:
        species (:class:`collections.abc.Sequence` of :class:`str`):
        sequence of all supported species, in order (it is recommended to order
        according to atomic number).
    """
    conv_tensor: Tensor

    def __init__(self, species):
        super().__init__()
        rev_idx = {s: k for k, s in enumerate(utils.PERIODIC_TABLE)}
        maxidx = max(rev_idx.values())
        self.register_buffer('conv_tensor', torch.full((maxidx + 2,), -1, dtype=torch.long))
        for i, s in enumerate(species):
            self.conv_tensor[rev_idx[s]] = i

    def forward(self, input_: Tuple[Tensor, Tensor],
                cell: Optional[Tensor] = None,
                pbc: Optional[Tensor] = None):
        """Convert species from periodic table element index to 0, 1, 2, 3, ... indexing"""
        species, coordinates = input_
        converted_species = self.conv_tensor[species]

        # check if unknown species are included
        if converted_species[species.ne(-1)].lt(0).any():
            raise ValueError(f'Unknown species found in {species}')

        return SpeciesCoordinates(converted_species.to(species.device), coordinates)
