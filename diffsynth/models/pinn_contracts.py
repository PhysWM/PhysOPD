from collections import OrderedDict

import torch


FIELD_CONTRACT_VERSION = "explicit_attribute_bank_v2"
EXPERT_FIELD_RECIPE_VERSION = "explicit_attribute_recipes_v2"


PHYSICS_ATTRIBUTE_CONTRACT = OrderedDict(
    [
        ("d", 4),      # displacement / deformation state
        ("u", 4),      # velocity / motion field
        ("p", 2),      # pressure
        ("rho", 2),    # density / compressibility
        ("T", 2),      # temperature / energy
        ("alpha", 2),  # phase / occupancy
        ("eps", 4),    # strain
        ("sigma", 4),  # stress
        ("j", 4),      # contact / impulse
        ("D", 2),      # damage / fracture
        ("psi", 2),    # wave / optical field
    ]
)


def _build_attribute_slices(contract):
    slices = OrderedDict()
    offset = 0
    for name, width in contract.items():
        slices[name] = slice(offset, offset + int(width))
        offset += int(width)
    return slices


PHYSICS_ATTRIBUTE_SLICES = _build_attribute_slices(PHYSICS_ATTRIBUTE_CONTRACT)
PHYSICS_ATTR_DIM = sum(int(width) for width in PHYSICS_ATTRIBUTE_CONTRACT.values())


PHENOMENON_LABELS = [
    "Rigid Body",
    "Elastic",
    "Fluid",
    "Compressible Flow",
    "Phase Change",
    "Collision/Contact",
    "Granular",
    "Fracture",
    "Thermal",
    "Optical",
]


EXPERT_FIELD_RECIPES = OrderedDict(
    [
        ("Rigid Body", ("d", "u", "eps", "sigma")),
        ("Elastic", ("d", "u", "eps", "sigma")),
        ("Fluid", ("u", "p", "rho")),
        ("Compressible Flow", ("u", "p", "rho")),
        ("Phase Change", ("u", "rho", "T", "alpha")),
        ("Collision/Contact", ("d", "u", "j")),
        ("Granular", ("u", "rho", "alpha", "j")),
        ("Fracture", ("d", "u", "eps", "sigma", "j", "D")),
        ("Thermal", ("u", "T")),
        ("Optical", ("psi", "alpha")),
    ]
)


def attribute_mask(field_names, device=None, dtype=None):
    device = device if device is not None else "cpu"
    dtype = dtype if dtype is not None else torch.float32
    mask = torch.zeros(PHYSICS_ATTR_DIM, device=device, dtype=dtype)
    for field_name in field_names:
        field_slice = PHYSICS_ATTRIBUTE_SLICES[field_name]
        mask[field_slice] = 1.0
    return mask


def field_indices(field_names):
    indices = []
    for field_name in field_names:
        field_slice = PHYSICS_ATTRIBUTE_SLICES[field_name]
        indices.extend(range(field_slice.start, field_slice.stop))
    return indices


def split_attribute_bank(attribute_bank):
    if not isinstance(attribute_bank, torch.Tensor):
        raise TypeError(f"Expected attribute bank tensor, got {type(attribute_bank)!r}.")
    fields = OrderedDict()
    for field_name, field_slice in PHYSICS_ATTRIBUTE_SLICES.items():
        fields[field_name] = attribute_bank[:, field_slice]
    return fields


def select_fields(attribute_bank, field_names):
    fields = split_attribute_bank(attribute_bank)
    return OrderedDict((field_name, fields[field_name]) for field_name in field_names)
