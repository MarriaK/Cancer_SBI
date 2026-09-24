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
shuffles and the test loader does not. ``num_workers`` now defaults to 0, which
is torch's default and the originals' behaviour, but can be raised: workers only
prefetch, and the shuffle stays on the main process's generator, so raising it
does not move the RNG stream or the results.

Each builder returns a **3-tuple** ``(train_loader, val_loader, test_loader)``.
``val_loader`` is ``None`` unless ``val_ids`` is given. That is
``MODEL_IMPROVEMENT_PLAN.md`` §5 step 1: the early-stopping set must not be the
test set.

Nothing here runs at import time.
"""

from typing import Optional, Sequence, Tuple

from torch.utils.data import DataLoader

from cancer_sbi.data.clone_sets import CNASimsDataset
from cancer_sbi.data.dominant_clone import SimulationDataset, collate_skip_none
from cancer_sbi.data.splits import complete_sim_ids


def _worker_kwargs(num_workers: int) -> dict:
    """The worker-related ``DataLoader`` flags, kept identical across all sites.

    Args:
        num_workers: Requested worker process count.

    Returns:
        ``{"num_workers": n}`` and nothing else.

    Raises:
        ValueError: If ``num_workers`` is negative.

    Note:
        ``persistent_workers`` was set here until 2026-09-24 and is deliberately
        gone. Persistent workers are torn down and rebuilt only when the loader
        is re-created, so from the *second* epoch onwards each worker carries the
        RNG state its previous epoch left it in, instead of the fresh per-epoch
        seed torch derives from the main process's generator. A seeded run with
        ``num_workers>0`` then diverges from the same run with ``num_workers=0``
        after epoch 1 -- which is exactly the neutrality the flag was added
        under. Re-spawning workers each epoch costs a second or two per epoch;
        the divergence costs the comparison.
    """
    if num_workers < 0:
        raise ValueError(f"num_workers must be >= 0, got {num_workers}")
    return {"num_workers": num_workers}


def _restrict_to_complete(
    root_dir: str, sim_ids: Optional[Sequence[str]], partition: str
) -> Optional[Sequence[str]]:
    """Apply the clone-set completeness rule to one partition and say so.

    Args:
        root_dir: Directory containing ``sim*/``.
        sim_ids: The partition's simulation names, or ``None`` (no such
            partition, e.g. an absent validation set).
        partition: ``"train"``, ``"val"`` or ``"test"``, for the printed line.

    Returns:
        The kept names, or ``None`` when ``sim_ids`` was ``None``.
    """
    if sim_ids is None:
        return None
    kept = complete_sim_ids(root_dir, sim_ids)
    print(
        f"[dominant_clone] require_all_trials: {partition} "
        f"{len(sim_ids)} -> {len(kept)}"
    )
    return kept


def build_clone_set_dataloaders(
    root_dir: str,
    train_ids: Sequence[str],
    test_ids: Sequence[str],
    top_k: int = 100,
    batch_size: int = 8,
    pin_memory: bool = False,
    *,
    val_ids: Optional[Sequence[str]] = None,
    num_workers: int = 0,
    cache_dir: Optional[str] = None,
    trial_subsample: Optional[int] = None,
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    """Build the train, validation and test loaders for CloneMLP/CloneAtt-NPE.

    Args:
        root_dir: Directory containing ``sim*/``.
        train_ids: Simulation names for training, from
            :func:`cancer_sbi.data.splits.load_split`.
        test_ids: Simulation names for testing.
        top_k: Clones kept per trial. 100 in both published models.
        batch_size: Sims per batch. The original default is 8; both
            ``*/main.py:24`` pass 32.
        pin_memory: Passed through to every loader.
        val_ids: Simulation names for early stopping, or ``None`` for no
            validation loader. Keyword-only.
        num_workers: ``DataLoader`` worker processes. 0 (the original
            behaviour) keeps loading on the main process. Workers are
            re-spawned every epoch, which is what keeps a seeded run's batches
            identical to the ``num_workers=0`` run's. Keyword-only.
        cache_dir: Optional pre-built clone cache (see
            ``src/utilities/build_clone_cache.py``). When given, every dataset
            reads its tensors from the cache instead of the gzipped trial files.
            Keyword-only.
        trial_subsample: Matrix-3 augmentation, ``None`` (the published
            behaviour) or the number of trials to draw per item. **It is given
            to the training dataset only.** Validation and test keep all 25
            trials, because that is the condition every reported number is
            measured under and because early stopping on a randomly-thinned
            validation set would compare each epoch against a different target.
            Keyword-only.

    Returns:
        ``(train_loader, val_loader, test_loader)``; ``val_loader`` is ``None``
        when ``val_ids`` is ``None``. Each batch is the 3-tuple
        ``(X_trials, trial_mask, y)`` with shapes ``(B, 25, top_k, 45)`` float32,
        ``(B, 25)`` bool and ``(B, 44)`` float32. The train loader shuffles; the
        validation and test loaders do not.

    Note:
        Four parameters of the original signature are still gone because the
        original accepted them and never used them: ``num_trials_per_sim``
        (never reached ``CNASimsDataset``, which used its own default of 25),
        ``splits`` and ``seed`` (only read by an unreachable fallback branch),
        ``shuffle`` (``Base_NPE/utils.py:235`` hard-codes ``shuffle=True`` for
        the training loader regardless) and ``**dataset_kwargs`` (never
        forwarded). ``num_workers`` is back, and is now actually passed to
        ``DataLoader``. Construct
        :class:`~cancer_sbi.data.clone_sets.CNASimsDataset` directly if you need
        non-default dataset options.

        There is deliberately no ``require_all_trials`` parameter here:
        ``CNASimsDataset`` already drops every sim that is missing a trial file,
        so this family's sim set *is* the restricted one. The flag exists only
        on the dominant-clone builder, which is the side that has to opt in.
    """
    # Preserved from Base_NPE/utils.py:232-233: only root_dir, top_k and sim_ids
    # are given, so every other dataset option keeps its class default --
    # in particular num_trials_per_sim=25, which drives the trap-10 filtering.
    train_dataset = CNASimsDataset(
        root_dir,
        top_k=top_k,
        sim_ids=train_ids,
        cache_dir=cache_dir,
        # The ONLY dataset that gets it -- see the argument's docstring.
        trial_subsample=trial_subsample,
    )
    test_dataset = CNASimsDataset(
        root_dir, top_k=top_k, sim_ids=test_ids, cache_dir=cache_dir
    )

    loader_kwargs = _worker_kwargs(num_workers)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        **loader_kwargs,
    )

    val_loader: Optional[DataLoader] = None
    if val_ids is not None:
        val_dataset = CNASimsDataset(
            root_dir, top_k=top_k, sim_ids=val_ids, cache_dir=cache_dir
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=pin_memory,
            **loader_kwargs,
        )

    return train_loader, val_loader, test_loader


def build_dominant_clone_dataloaders(
    root_dir: str,
    train_ids: Sequence[str],
    test_ids: Sequence[str],
    batch_size: int = 8,
    pin_memory: bool = False,
    *,
    val_ids: Optional[Sequence[str]] = None,
    num_workers: int = 0,
    cache_dir: Optional[str] = None,
    require_all_trials: bool = False,
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    """Build the train, validation and test loaders for DominantClone-NPE.

    Args:
        root_dir: Directory containing ``sim*/``.
        train_ids: Simulation names for training.
        test_ids: Simulation names for testing.
        batch_size: Sims per batch. The original default is 8;
            ``Plain_NPE/main.py:24`` passes 32.
        pin_memory: Passed through to every loader.
        val_ids: Simulation names for early stopping, or ``None``. Keyword-only.
        num_workers: ``DataLoader`` worker processes, default 0. Keyword-only.
        cache_dir: Accepted only so that the two builders can be called through
            one code path. **This family has no cache** -- the clone cache holds
            ``(25, top_k, 45)`` clone sets, not the dominant clone's
            ``(25, 44)`` -- so a non-``None`` value is reported and ignored
            rather than silently pretending to help. Keyword-only.
        require_all_trials: Keep only the sims the clone-set models keep, i.e.
            those with all 25 ``CNratios_all.pkl.gz`` files and a readable
            ``parameters.pkl`` (:func:`cancer_sbi.data.splits.complete_sim_ids`).
            ``False``, the default, is the published behaviour: trap 10 stands,
            the missing trials are NaN-padded and the sim is kept. ``True`` is
            what makes this model's train/val/test sets identical to
            CloneMLP's and CloneAtt's, and one line per partition says how many
            sims it dropped. Keyword-only.

    Returns:
        ``(train_loader, val_loader, test_loader)``; ``val_loader`` is ``None``
        when ``val_ids`` is ``None``. Each batch is either ``None`` (when
        every sim in it was dropped) or the 2-tuple ``(theta, x)`` with shapes
        ``(B, 44)`` and ``(B, 25, 44)``, both float32. ``B`` can be smaller than
        ``batch_size`` because :func:`collate_skip_none` drops unusable sims.

    Note:
        ``num_trials_per_sim`` and ``shuffle`` are still gone from the original
        signature: neither reached anything (``Plain_NPE/utils.py:129-133``
        constructs ``SimulationDataset`` without ``num_trials`` and hard-codes
        ``shuffle=True`` for the training loader). Construct
        :class:`~cancer_sbi.data.dominant_clone.SimulationDataset` directly if
        you need a different trial count.
    """
    if cache_dir is not None:
        print(
            "[build_dominant_clone_dataloaders] ignoring cache_dir="
            f"{cache_dir}: the clone cache holds (25, top_k, 45) clone sets, "
            "which this dataset does not read."
        )

    if require_all_trials:
        train_ids = _restrict_to_complete(root_dir, train_ids, "train")
        val_ids = _restrict_to_complete(root_dir, val_ids, "val")
        test_ids = _restrict_to_complete(root_dir, test_ids, "test")

    # Preserved from Plain_NPE/utils.py:129-130: only root_dir and sim_ids are
    # given, so num_trials keeps its class default of 25 and use_bulk stays
    # False (i.e. CNratios_largest, the dominant clone).
    train_dataset = SimulationDataset(root_dir, sim_ids=train_ids)
    test_dataset = SimulationDataset(root_dir, sim_ids=test_ids)

    loader_kwargs = _worker_kwargs(num_workers)

    # Preserved from Plain_NPE/utils.py:132-133: collate_skip_none is what makes
    # the None samples from SimulationDataset.__getitem__ legal. Without it the
    # default collate raises on the first sim that has no parameters.pkl.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        collate_fn=collate_skip_none,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        collate_fn=collate_skip_none,
        **loader_kwargs,
    )

    val_loader: Optional[DataLoader] = None
    if val_ids is not None:
        val_dataset = SimulationDataset(root_dir, sim_ids=val_ids)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=pin_memory,
            collate_fn=collate_skip_none,
            **loader_kwargs,
        )

    return train_loader, val_loader, test_loader


__all__ = ["build_clone_set_dataloaders", "build_dominant_clone_dataloaders"]
