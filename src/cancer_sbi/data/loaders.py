"""DataLoader builders, one per dataset family.

Ported from ``Base_NPE/utils.py:210-238`` (clone sets; byte-identical to
``SetTransformer_NPE/utils.py``) and ``Plain_NPE/utils.py:112-135``
(dominant clone).

There are deliberately **two** builders and no unified one. The two paths use
different dataset classes, different on-disk files, different padding policies
and -- only on the dominant-clone side -- a custom ``collate_fn``. A single
builder would have to branch on all four, and the temptation to share a default
is exactly how the two pipelines' sim filtering would drift into each other
(trap 10).

Both builders reproduce the originals' loader flags exactly: the training loader
shuffles, the test loader does not, and ``num_workers`` stays at torch's default
of 0.

Nothing here runs at import time.
"""

from typing import Sequence, Tuple

from torch.utils.data import DataLoader

from cancer_sbi.data.clone_sets import CNASimsDataset
from cancer_sbi.data.dominant_clone import SimulationDataset, collate_skip_none


def build_clone_set_dataloaders(
    root_dir: str,
    train_ids: Sequence[str],
    test_ids: Sequence[str],
    top_k: int = 100,
    batch_size: int = 8,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Build the train and test loaders for CloneMLP-NPE and CloneAtt-NPE.

    Args:
        root_dir: Directory containing ``sim*/``.
        train_ids: Simulation names for training, from
            :func:`cancer_sbi.data.splits.load_split`.
        test_ids: Simulation names for testing.
        top_k: Clones kept per trial. 100 in both published models.
        batch_size: Sims per batch. The original default is 8; both
            ``*/main.py:24`` pass 32.
        pin_memory: Passed through to both loaders.

    Returns:
        ``(train_loader, test_loader)``. Each batch is the 3-tuple
        ``(X_trials, trial_mask, y)`` with shapes ``(B, 25, top_k, 45)`` float32,
        ``(B, 25)`` bool and ``(B, 44)`` float32. The train loader shuffles; the
        test loader does not.

    Note:
        Six parameters of the original signature are gone because the original
        accepted them and never used them: ``num_trials_per_sim`` (never reached
        ``CNASimsDataset``, which used its own default of 25), ``splits`` and
        ``seed`` (only read by an unreachable fallback branch), ``shuffle``
        (``Base_NPE/utils.py:235`` hard-codes ``shuffle=True`` for the training
        loader regardless), ``num_workers`` (never passed to ``DataLoader``) and
        ``**dataset_kwargs`` (never forwarded). Construct
        :class:`~cancer_sbi.data.clone_sets.CNASimsDataset` directly if you need
        non-default dataset options.
    """
    # Preserved from Base_NPE/utils.py:232-233: only root_dir, top_k and sim_ids
    # are given, so every other dataset option keeps its class default --
    # in particular num_trials_per_sim=25, which drives the trap-10 filtering.
    train_dataset = CNASimsDataset(root_dir, top_k=top_k, sim_ids=train_ids)
    test_dataset = CNASimsDataset(root_dir, top_k=top_k, sim_ids=test_ids)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory
    )
    return train_loader, test_loader


def build_dominant_clone_dataloaders(
    root_dir: str,
    train_ids: Sequence[str],
    test_ids: Sequence[str],
    batch_size: int = 8,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Build the train and test loaders for DominantClone-NPE.

    Args:
        root_dir: Directory containing ``sim*/``.
        train_ids: Simulation names for training.
        test_ids: Simulation names for testing.
        batch_size: Sims per batch. The original default is 8;
            ``Plain_NPE/main.py:24`` passes 32.
        pin_memory: Passed through to both loaders.

    Returns:
        ``(train_loader, test_loader)``. Each batch is either ``None`` (when
        every sim in it was dropped) or the 2-tuple ``(theta, x)`` with shapes
        ``(B, 44)`` and ``(B, 25, 44)``, both float32. ``B`` can be smaller than
        ``batch_size`` because :func:`collate_skip_none` drops unusable sims.

    Note:
        ``num_trials_per_sim`` and ``shuffle`` are gone from the original
        signature: neither reached anything (``Plain_NPE/utils.py:129-133``
        constructs ``SimulationDataset`` without ``num_trials`` and hard-codes
        ``shuffle=True`` for the training loader). Construct
        :class:`~cancer_sbi.data.dominant_clone.SimulationDataset` directly if
        you need a different trial count.
    """
    # Preserved from Plain_NPE/utils.py:129-130: only root_dir and sim_ids are
    # given, so num_trials keeps its class default of 25 and use_bulk stays
    # False (i.e. CNratios_largest, the dominant clone).
    train_dataset = SimulationDataset(root_dir, sim_ids=train_ids)
    test_dataset = SimulationDataset(root_dir, sim_ids=test_ids)

    # Preserved from Plain_NPE/utils.py:132-133: collate_skip_none is what makes
    # the None samples from SimulationDataset.__getitem__ legal. Without it the
    # default collate raises on the first sim that has no parameters.pkl.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        collate_fn=collate_skip_none,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        collate_fn=collate_skip_none,
    )
    return train_loader, test_loader


__all__ = ["build_clone_set_dataloaders", "build_dominant_clone_dataloaders"]
