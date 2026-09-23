"""Clone-set dataset for CloneMLP-NPE and CloneAtt-NPE.

Ported verbatim (behaviour-wise) from ``Base_NPE/utils.py``, which is
byte-identical to ``SetTransformer_NPE/utils.py``. Provides:

* the small pickle/discovery helpers the whole package shares,
* :func:`top_frequent_rows_tensor`, which summarises one trial's clone table as
  its ``top_k`` most frequent copy-number profiles plus a frequency column,
* :class:`CNASimsDataset`, which yields one *sim* (not one trial) per item.

Nothing here runs at import time.
"""

import gzip
import os
import pickle
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset


def load_gz_pickle(filepath: str) -> Any:
    """Load a gzip-compressed pickle.

    Args:
        filepath: Path to a ``.pkl.gz`` file.

    Returns:
        Whatever the pickle contains; for trial files a ``numpy.ndarray`` of
        shape ``(N_clones, 44)``, dtype float.

    Raises:
        FileNotFoundError: If ``filepath`` does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at: {filepath}")
    with gzip.open(filepath, "rb") as handle:
        return pickle.load(handle)


def load_pickle(filepath: str) -> Any:
    """Load a plain pickle.

    Args:
        filepath: Path to a ``.pkl`` file.

    Returns:
        Whatever the pickle contains; for ``parameters.pkl`` a 1-D array-like of
        length 46 (2 nuisance entries followed by the 44 selection coefficients).

    Raises:
        FileNotFoundError: If ``filepath`` does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found at: {filepath}")
    with open(filepath, "rb") as handle:
        return pickle.load(handle)


def discover_sim_trials(root_dir: str, sim_glob_regex: str = r"^sim\d+$") -> List[str]:
    """List the simulation directories under ``root_dir``, numerically sorted.

    Args:
        root_dir: Directory containing ``sim1/``, ``sim2/``, ...
        sim_glob_regex: Regex a directory name must match to count as a sim.

    Returns:
        Absolute-ish paths (``os.path.join(root_dir, name)``) sorted by the first
        integer in the basename, so ``sim2`` precedes ``sim10``.
    """
    sims: List[str] = []
    for name in os.listdir(root_dir):
        if re.match(sim_glob_regex, name):
            full = os.path.join(root_dir, name)
            if os.path.isdir(full):
                sims.append(full)
    sims.sort(key=lambda p: int(re.findall(r"\d+", os.path.basename(p))[0]))
    return sims


def top_frequent_rows_tensor(
    matrix: torch.Tensor,
    top_k: int = 100,
    normalize: bool = True,
    pad_to_exact_k: bool = False,
) -> torch.Tensor:
    """Keep the ``top_k`` most frequent clone rows and append their frequency.

    One trial's raw table lists one row per sequenced cell, so identical rows
    mean cells of the same clone. This collapses duplicates, ranks clones by
    cell count, keeps the top ``top_k`` and appends the (normalised) count as a
    45th column.

    Args:
        matrix: ``(N, M)`` tensor of clone profiles, float32 in practice, one row
            per cell. ``M`` is 44 for this data.
        top_k: How many distinct clone rows to keep.
        normalize: Divide the counts by ``N`` to get a probability-like weight.
        pad_to_exact_k: Pad with zero rows so the output always has ``top_k``
            rows, which is what the batched dataset needs.

    Returns:
        ``(K, M + 1)`` float tensor where ``K == top_k`` when ``pad_to_exact_k``
        else ``min(top_k, num_unique)``. Column ``M`` holds the frequency.

    Raises:
        ValueError: If ``matrix`` is empty.
    """
    if matrix.numel() == 0:
        raise ValueError("Empty matrix passed to top_frequent_rows_tensor.")

    unique_rows, counts = torch.unique(matrix, dim=0, return_counts=True)

    # Preserved from Base_NPE/utils.py:32. Trap 17: torch.argsort is NOT stable,
    # so the relative order of clones with equal cell counts is unspecified and
    # can decide which clones make the top 100. Adding stable=True would look
    # like a fix but it changes which clones enter the set, and therefore the
    # results. See docs/REFACTOR_NOTES.md.
    order = torch.argsort(counts, descending=True)
    unique_rows = unique_rows[order]
    counts = counts[order].float()

    if normalize:
        # Preserved from Base_NPE/utils.py:36-37. Trap 18: the divisor is the
        # TOTAL number of input rows, not the number of rows retained, so the
        # kept frequencies sum to less than 1 and are never renormalised. The
        # encoders then use these raw weights for pooling. This looks wrong but
        # it is what the published models do; renormalising changes the results.
        # See docs/REFACTOR_NOTES.md.
        counts = counts / matrix.size(0)

    # Take top_k
    top_k_kept = min(top_k, unique_rows.size(0))
    rows = unique_rows[:top_k_kept]
    freqs = counts[:top_k_kept].unsqueeze(1)
    out = torch.cat([rows, freqs], dim=1)

    if pad_to_exact_k and top_k_kept < top_k:
        # Preserved from Base_NPE/utils.py:45-47. Trap 8 (first sentinel):
        # missing clone rows are padded with ZEROS, while missing *trials* are
        # padded with NaN (see CNASimsDataset.__getitem__). Because the encoders
        # detect padding with `isnan(x).all(-1)`, these zero rows are processed
        # as if they were real clones with frequency 0. Do not switch this to NaN
        # and do not add clone-level masking; both change the results.
        # See docs/REFACTOR_NOTES.md.
        pad_rows = torch.zeros((top_k - top_k_kept, out.size(1)), dtype=out.dtype)
        out = torch.vstack([out, pad_rows])
    elif pad_to_exact_k and top_k_kept > top_k:
        # Preserved from Base_NPE/utils.py:48-49. Unreachable: top_k_kept is
        # min(top_k, n_unique), so it can never exceed top_k. Kept so this
        # function stays line-for-line comparable with the original.
        out = out[:top_k]

    return out


class CNASimsDataset(Dataset):
    """One item per simulation: its stack of trials and its 44 parameters.

    Ported from ``Base_NPE/utils.py:83-206``.

    Each item is a whole sim, so a mini-batch of 32 is 32 sims, each carrying
    ``num_trials_per_sim`` trials of ``top_k`` clones.
    """

    def __init__(
        self,
        root_dir: str,
        num_trials_per_sim: int = 25,
        top_k: int = 100,
        normalize_freq: bool = True,
        pad_to_exact_k: bool = False,
        sim_regex: str = r"^sim\d+$",
        trial_filename: str = "CNratios_all.pkl.gz",
        params_filename: str = "parameters.pkl",
        drop_missing: bool = True,
        sim_ids: Optional[Sequence[str]] = None,
    ) -> None:
        """Scan ``root_dir`` and index the usable sims.

        Args:
            root_dir: Directory containing ``sim*/``.
            num_trials_per_sim: Trials expected per sim; also the ``T`` dimension
                of every item and the minimum a sim must have to be kept.
            top_k: Clones kept per trial.
            normalize_freq: Passed to :func:`top_frequent_rows_tensor`.
            pad_to_exact_k: Stored but never read -- see the comment below.
            sim_regex: Which directory names count as a sim.
            trial_filename: File read inside ``sim<N>/<t>/``.
            params_filename: File holding theta inside ``sim<N>/``.
            drop_missing: Warn and skip on unreadable parameters / missing trial
                files instead of raising.
            sim_ids: Restrict to these sim *names* (e.g. ``["sim1", "sim2"]``),
                which is how the train/test split is applied.

        Raises:
            RuntimeError: If no sims match, none survive the ``sim_ids`` filter,
                or none survive the minimum-trials filter.
        """
        self.root_dir = root_dir
        self.num_trials = num_trials_per_sim
        self.top_k = top_k
        self.normalize = normalize_freq
        # Preserved from Base_NPE/utils.py:108 and :183-188: the constructor
        # stores pad_to_exact_k but _load_and_process_trial hard-codes
        # pad_to_exact_k=True, so this attribute is dead. Kept so the constructor
        # signature matches the original.
        self.pad_to_exact_k = pad_to_exact_k
        self.trial_filename = trial_filename
        self.params_filename = params_filename
        self.drop_missing = drop_missing

        all_sims = discover_sim_trials(root_dir, sim_regex)
        if not all_sims:
            raise RuntimeError(f"No sims found in {root_dir} matching /{sim_regex}/")

        if sim_ids is not None:
            allowed = set(sim_ids)
            self.sims = [
                sim_dir for sim_dir in all_sims
                if os.path.basename(sim_dir) in allowed
            ]
        else:
            self.sims = all_sims

        if not self.sims:
            raise RuntimeError("No sims left after applying sim_ids filter.")

        self.items: List[Dict[str, Any]] = []

        for sim_dir in self.sims:
            params_path = os.path.join(sim_dir, self.params_filename)
            try:
                y_np = load_pickle(params_path)  # shape (46,) or similar
            except Exception as exc:  # noqa: BLE001 - mirrors the original
                if self.drop_missing:
                    print(f"[WARN] Skipping {sim_dir}: failed to load parameters ({exc})")
                    continue
                raise

            # The first two entries are nuisance parameters; the 44 selection
            # coefficients are columns 2.. (Base_NPE/utils.py:146-147).
            y_tensor = torch.as_tensor(y_np[2:], dtype=torch.float32)

            # Spelling kept from Base_NPE/utils.py:149 ("avaiable_trials") only
            # as a local; the public key below is spelled correctly, exactly as
            # the original's dict key is.
            available_trials: List[int] = []
            for trial_idx in range(1, self.num_trials + 1):
                trial_path = os.path.join(sim_dir, str(trial_idx), self.trial_filename)
                if os.path.exists(trial_path):
                    available_trials.append(trial_idx)
                else:
                    if not self.drop_missing:
                        raise FileNotFoundError(f"Missing trial file: {trial_path}")

            # Preserved from Base_NPE/utils.py:158-161. Trap 10: a sim with even
            # one missing trial file is dropped entirely, which is why
            # CloneMLP/CloneAtt train on 639 sims and test on 152 while
            # DominantClone (which NaN-pads instead) gets 718 / 168. The two
            # dataset classes must NOT be unified. The warning text is also
            # inherited verbatim and is misleading: trials were found, just not
            # enough of them. See docs/REFACTOR_NOTES.md.
            if len(available_trials) < self.num_trials:
                print(f"[WARN] Skipping {sim_dir}: no available trials found.")
                continue

            self.items.append(
                {
                    "sim_dir": sim_dir,
                    "available_trials": available_trials,
                    "y": y_tensor,
                }
            )

        if not self.items:
            raise RuntimeError("No valid (sim, trial) pairs found after scanning all sims.")

        print(f"[CNASimsDataset] sims={len(self.items)} top_k={self.top_k}")

    def __len__(self) -> int:
        """Number of sims kept.

        Returns:
            The count of usable simulations.
        """
        return len(self.items)

    def _load_and_process_trial(self, sim_dir: str, trial_idx: int) -> torch.Tensor:
        """Read one trial file and summarise it as a fixed-size clone set.

        Args:
            sim_dir: Path to the sim directory.
            trial_idx: 1-based trial number.

        Returns:
            ``(top_k, 45)`` float32 tensor: 44 CNA features plus the frequency
            column, zero-padded to exactly ``top_k`` rows.
        """
        trial_path = os.path.join(sim_dir, str(trial_idx), self.trial_filename)
        arr = load_gz_pickle(trial_path)  # numpy (N_clones, 44)
        x = torch.as_tensor(arr, dtype=torch.float32)
        return top_frequent_rows_tensor(
            x,
            top_k=self.top_k,
            normalize=self.normalize,
            # Preserved from Base_NPE/utils.py:187: hard-coded True, ignoring
            # self.pad_to_exact_k. Batching needs a fixed K.
            pad_to_exact_k=True,
        )

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return one simulation.

        Args:
            i: Index into the kept sims.

        Returns:
            A 3-tuple, and the arity is part of the contract -- every training
            and evaluation loop unpacks it as ``for X, _, theta in loader``:

            * ``X_trials``: ``(num_trials, top_k, 45)`` float32. Slots for trials
              that were not loaded stay NaN.
            * ``trial_mask``: ``(num_trials,)`` bool, True where a trial was
              loaded. See the trap comment below.
            * ``y``: ``(44,)`` float32 selection coefficients.
        """
        item = self.items[i]
        sim_dir = item["sim_dir"]
        avail = item["available_trials"]
        y = item["y"]

        # Preserved from Base_NPE/utils.py:198. Trap 8 (second sentinel): missing
        # TRIALS are padded with NaN, unlike missing clone rows, which are padded
        # with zeros. The NaN sentinel is the one the encoders actually test for
        # (`isnan(x).all(-1)`), so only trial-level padding is ever masked.
        # See docs/REFACTOR_NOTES.md.
        x_trials = torch.full(
            (self.num_trials, self.top_k, 45), float("nan"), dtype=torch.float32
        )
        trial_mask = torch.zeros((self.num_trials,), dtype=torch.bool)

        for slot, trial_idx in enumerate(avail[: self.num_trials]):
            x_trials[slot] = self._load_and_process_trial(sim_dir, trial_idx)
            trial_mask[slot] = 1.0

        # Preserved from Base_NPE/utils.py:206. Trap 9: trial_mask is returned by
        # every consumer as `_` and never read -- the encoders recompute
        # validity from the NaNs instead. It stays in the tuple because the
        # 3-tuple shape is the contract shared with the training loops, and it
        # must NOT start being used. See docs/REFACTOR_NOTES.md.
        return x_trials, trial_mask, y


__all__ = [
    "load_gz_pickle",
    "load_pickle",
    "discover_sim_trials",
    "top_frequent_rows_tensor",
    "CNASimsDataset",
]
