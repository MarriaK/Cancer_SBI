"""Dominant-clone dataset for DominantClone-NPE.

Ported from ``Plain_NPE/utils.py:40-109``. This path does not summarise a trial
as a *set* of clones: it takes the single dominant clone's copy-number profile
(``CNratios_largest``) per trial, so one sim is a ``(num_trials, 44)`` matrix.

``Plain_NPE/utils.py`` also carries its own byte-identical copies of the pickle
helpers and ``discover_sim_trials``; they are imported from
:mod:`cancer_sbi.data.clone_sets` rather than duplicated here, which is the one
place the two pipelines genuinely agreed.

Nothing here runs at import time.
"""

import os
import pickle
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from cancer_sbi.data.clone_sets import load_pickle


class SimulationDataset(Dataset):
    """One item per simulation: its ``(T, 44)`` trial matrix and its parameters.

    Ported from ``Plain_NPE/utils.py:40-100``. Unlike
    :class:`~cancer_sbi.data.clone_sets.CNASimsDataset`, this class keeps sims
    with missing trials by NaN-padding them, and only rejects a sim when *every*
    trial is missing.
    """

    def __init__(
        self,
        root_dir: str,
        sim_ids: Optional[Sequence[str]] = None,
        num_trials: int = 25,
        use_bulk: bool = False,
    ) -> None:
        """Index the sims without reading any of their contents.

        Args:
            root_dir: Directory containing ``sim*/``.
            sim_ids: Sim *names* to use (e.g. ``["sim1", "sim2"]``); when
                ``None``, every ``sim*`` directory under ``root_dir`` is used.
            num_trials: Trials per sim, i.e. the ``T`` dimension of every item.
            use_bulk: Take ``CNratios_bulk`` (element 2 of the results tuple)
                instead of ``CNratios_largest`` (element 1).
        """
        self.root_dir = root_dir
        self.num_trials = num_trials
        self.use_bulk = use_bulk

        # Sorted by the digits in the name, so sim2 precedes sim10. Note this
        # sorts the *given* ids rather than re-discovering them, which is how the
        # split is applied (Plain_NPE/utils.py:47-53).
        if sim_ids is not None:
            all_dirs = sorted(sim_ids, key=lambda x: int("".join(filter(str.isdigit, x))))
        else:
            all_dirs = sorted(
                [d for d in os.listdir(root_dir) if d.startswith("sim")],
                key=lambda x: int("".join(filter(str.isdigit, x))),
            )

        self.sim_dirs: List[str] = list(all_dirs)

    def __len__(self) -> int:
        """Number of sims indexed.

        Returns:
            The count of simulation directories, before any content filtering --
            unreadable sims are dropped later, per batch, by
            :func:`collate_skip_none`.
        """
        return len(self.sim_dirs)

    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Return one simulation, or ``None`` if it is unusable.

        Args:
            idx: Index into ``self.sim_dirs``.

        Returns:
            ``None`` when ``parameters.pkl`` is missing or every trial is
            missing; otherwise a 2-tuple ``(theta, x_tensor)``:

            * ``theta``: ``(44,)`` float32 selection coefficients.
            * ``x_tensor``: ``(num_trials, 44)`` float32, NaN rows where a trial
              was missing or incomplete.

            The 2-tuple arity is the contract: ``Plain_NPE/model.py:148`` and
            ``:197`` unpack it as ``theta_batch, x_batch = batch``. ``None`` is
            legal only because :func:`collate_skip_none` filters it out.
        """
        sim_path = os.path.join(self.root_dir, self.sim_dirs[idx])

        # Preserved from Plain_NPE/utils.py:64-66: a sim with no parameters.pkl
        # becomes None and is dropped from its batch rather than raising.
        theta_path = os.path.join(sim_path, "parameters.pkl")
        if not os.path.exists(theta_path):
            return None
        theta_np = load_pickle(theta_path)  # shape (46,) or similar
        # Skip the first two (nuisance) parameters.
        theta = torch.as_tensor(theta_np[2:], dtype=torch.float32)

        trials: List[torch.Tensor] = []
        for trial_idx in range(1, self.num_trials + 1):
            trial_dir = os.path.join(sim_path, str(trial_idx))
            result_path = os.path.join(trial_dir, "results.pkl")

            result_np: Any = None
            if os.path.exists(result_path):
                with open(result_path, "rb") as handle:
                    data = pickle.load(handle)
                # A complete result is the 3-tuple
                # ([failed_sim, failed_call], CNratios_largest, CNratios_bulk).
                # Anything else (a partial write, an old format) leaves
                # result_np None and the trial is NaN-padded below.
                if isinstance(data, tuple) and len(data) == 3:
                    result_np = data[2] if self.use_bulk else data[1]

            if result_np is not None:
                trials.append(torch.tensor(np.asarray(result_np, dtype=np.float32)))
            else:
                # Preserved from Plain_NPE/utils.py:86-92. Trap 10: the
                # DominantClone path NaN-PADS a missing trial and keeps the sim,
                # where CNASimsDataset would have dropped the whole sim. That is
                # why this model trains on 718 sims and the other two on 639.
                # The fallback width of 44 only applies when trial 1 itself is
                # missing, since otherwise the shape is copied from trials[0].
                # This looks wrong but it is what the published model does;
                # changing it changes the results. See docs/REFACTOR_NOTES.md.
                if trials:
                    nan_tensor = torch.full_like(trials[0], float("nan"))
                else:
                    nan_tensor = torch.zeros(44, dtype=torch.float32).fill_(float("nan"))
                trials.append(nan_tensor)

        x_tensor = torch.stack(trials)  # (num_trials, feature_dim)

        # Preserved from Plain_NPE/utils.py:96-98: an all-NaN sim would make
        # DeepSet divide by a zero trial count, so it is dropped here instead.
        if torch.isnan(x_tensor).all():
            return None

        return theta, x_tensor


def collate_skip_none(batch: Sequence[Optional[Tuple[torch.Tensor, torch.Tensor]]]) -> Any:
    """Collate a batch after dropping the ``None`` samples.

    Ported from ``Plain_NPE/utils.py:104-109``. ``SimulationDataset.__getitem__``
    returns ``None`` for sims with no ``parameters.pkl`` and for sims whose every
    trial is missing; those must not reach ``default_collate``.

    Args:
        batch: Samples from :class:`SimulationDataset`, each either a
            ``(theta, x_tensor)`` pair or ``None``.

    Returns:
        ``None`` when the whole batch was dropped -- the training and validation
        loops test ``if batch is None: continue`` (``Plain_NPE/model.py:146-147``
        and ``:195-196``). Otherwise the default-collated 2-tuple
        ``(theta, x)`` with shapes ``(B, 44)`` and ``(B, num_trials, 44)``,
        where ``B`` is the number of surviving samples and can be smaller than
        the configured batch size.
    """
    kept = [sample for sample in batch if sample is not None]
    if not kept:
        return None
    return torch.utils.data.dataloader.default_collate(kept)


__all__ = ["SimulationDataset", "collate_skip_none"]
