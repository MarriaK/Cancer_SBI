"""Datasets, splits and DataLoader builders.

Two dataset families live here and are deliberately kept apart:

* :mod:`cancer_sbi.data.clone_sets` -- ``CNASimsDataset``, used by CloneMLP-NPE
  and CloneAtt-NPE. A trial is a *set* of clone profiles with frequencies. A sim
  missing any trial file is dropped.
* :mod:`cancer_sbi.data.dominant_clone` -- ``SimulationDataset``, used by
  DominantClone-NPE. A trial is the single dominant clone's profile. A sim
  missing trials is NaN-padded and kept.

They therefore train on different subsets of the same simulation directory
(trap 10). Do not unify them.

Importing this module pulls in torch; it has no other side effects.
"""

from cancer_sbi.data.clone_sets import (
    CNASimsDataset,
    discover_sim_trials,
    load_gz_pickle,
    load_pickle,
    top_frequent_rows_tensor,
)
from cancer_sbi.data.dominant_clone import SimulationDataset, collate_skip_none
from cancer_sbi.data.loaders import (
    build_clone_set_dataloaders,
    build_dominant_clone_dataloaders,
)
from cancer_sbi.data.splits import (
    complete_sim_ids,
    create_split,
    list_sim_ids,
    load_split,
    save_split,
)

__all__ = [
    "CNASimsDataset",
    "SimulationDataset",
    "collate_skip_none",
    "discover_sim_trials",
    "load_gz_pickle",
    "load_pickle",
    "top_frequent_rows_tensor",
    "build_clone_set_dataloaders",
    "build_dominant_clone_dataloaders",
    "complete_sim_ids",
    "create_split",
    "list_sim_ids",
    "load_split",
    "save_split",
]
