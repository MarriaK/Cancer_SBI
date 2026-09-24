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
import hashlib
import inspect
import json
import os
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

#: File names inside a clone cache directory, shared with
#: ``src/utilities/build_clone_cache.py`` and ``verify_clone_cache.py`` so the
#: builder, the verifier and the reader cannot drift apart.
CACHE_X_FILENAME = "X.npy"
CACHE_THETA_FILENAME = "theta.npy"
CACHE_SIM_IDS_FILENAME = "sim_ids.npy"
CACHE_MANIFEST_FILENAME = "manifest.json"


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
        cache_dir: Optional[str] = None,
        trial_subsample: Optional[int] = None,
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
            cache_dir: Directory written by ``src/utilities/build_clone_cache.py``.
                When given, ``__getitem__`` reads the already-summarised tensors
                from ``X.npy`` / ``theta.npy`` instead of opening 25 gzipped
                files and re-running :func:`top_frequent_rows_tensor`. The scan
                below still runs, so the set of sims kept is decided by exactly
                the same filters either way.
            trial_subsample: Matrix-3 augmentation. ``None`` -- the default and
                the published behaviour -- returns all ``num_trials_per_sim``
                trials, in order. An int ``K`` returns a fresh random subset of
                ``K`` trials on every ``__getitem__``, so every epoch sees a
                different view of the same sim. ``K >= num_trials_per_sim`` is
                recorded but inert: there is nothing to choose, so the item is
                bit-identical to the ``None`` one and no random number is drawn.

                **Give this to the training dataset only.** The validation and
                test sets are the published evaluation condition -- 25 trials --
                and the loader builder enforces that (``data/loaders.py``).

                The draw comes from the ambient ``torch`` RNG
                (:func:`torch.randperm`), not from a per-dataset or per-index
                generator. That is what makes it *fresh each epoch* rather than
                a fixed function of the index, and it stays reproducible under
                ``num_workers > 0`` because torch derives each worker's seed,
                every epoch, from the main process's generator -- the same
                property ``_worker_kwargs`` dropped ``persistent_workers`` to
                protect. Two runs with the same ``--seed`` therefore draw the
                same subsets; two different seeds do not.

        Raises:
            RuntimeError: If no sims match, none survive the ``sim_ids`` filter,
                or none survive the minimum-trials filter.
            ValueError: If ``cache_dir`` was built by a different version of
                :func:`top_frequent_rows_tensor`, or with a different ``top_k``
                or trial count than this dataset asks for; or if
                ``trial_subsample`` is less than 1.
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

        if trial_subsample is not None and int(trial_subsample) < 1:
            raise ValueError(
                f"trial_subsample must be >= 1, got {trial_subsample!r}."
            )
        # Recorded as given, so a caller can read back what it asked for...
        self.trial_subsample = (
            int(trial_subsample) if trial_subsample is not None else None
        )
        # ...but K >= num_trials selects every trial, and "select every trial in
        # order" is exactly the published path. Collapsing it here rather than
        # in __getitem__ keeps that case bit-identical AND free of an RNG draw,
        # which is what makes a K=25 run comparable with a K=None one.
        self._subsample_k: Optional[int] = (
            self.trial_subsample
            if self.trial_subsample is not None
            and self.trial_subsample < self.num_trials
            else None
        )

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

        # The cache is attached after the scan, so the kept sims are chosen by
        # the same code whether or not a cache is in use.
        self.cache_dir = str(cache_dir) if cache_dir is not None else None
        self._cache_rows: Dict[str, int] = {}
        # Opened lazily in __getitem__ rather than here: a memmap opened in the
        # parent process is inherited by every DataLoader worker after a fork
        # and that is exactly the way to get corrupt reads.
        self._cache_x: Optional[np.ndarray] = None
        self._cache_theta: Optional[np.ndarray] = None
        if self.cache_dir is not None:
            self._attach_cache(self.cache_dir)

        print(
            f"[CNASimsDataset] sims={len(self.items)} top_k={self.top_k}"
            + (f" cache={self.cache_dir}" if self.cache_dir else "")
            + (
                f" trial_subsample={self._subsample_k}/{self.num_trials}"
                if self._subsample_k is not None
                else ""
            )
        )

    def _attach_cache(self, cache_dir: str) -> None:
        """Validate a clone cache and index it by sim name.

        Args:
            cache_dir: Directory holding ``manifest.json``, ``sim_ids.npy``,
                ``X.npy`` and ``theta.npy``.

        Raises:
            FileNotFoundError: If the manifest is missing.
            ValueError: If the manifest's recorded source hash of
                :func:`top_frequent_rows_tensor` differs from the live
                function's; if its ``top_k`` / ``num_trials`` differ from this
                dataset's; if its ``root`` resolves to a different directory
                than this dataset is reading; if ``sim_ids.npy`` and ``X.npy``
                disagree on how many sims the cache holds; or if any sim this
                dataset will serve is absent from the cache index. A stale or
                partial cache is the one failure mode that would silently
                change every published number, so all six raise rather than
                falling back to the slow path.
        """
        manifest_path = os.path.join(cache_dir, CACHE_MANIFEST_FILENAME)
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Cache manifest not found at: {manifest_path}")
        with open(manifest_path, "r") as handle:
            manifest = json.load(handle)

        live_hash = top_frequent_rows_source_sha1()
        cached_hash = manifest.get("top_frequent_rows_tensor_sha1")
        if cached_hash != live_hash:
            raise ValueError(
                f"Clone cache at {cache_dir} was built by a different version of "
                f"top_frequent_rows_tensor (manifest {cached_hash!r}, live "
                f"{live_hash!r}). Rebuild the cache; do not train on it."
            )
        if int(manifest.get("top_k", -1)) != int(self.top_k):
            raise ValueError(
                f"Clone cache at {cache_dir} has top_k={manifest.get('top_k')}, "
                f"dataset asks for top_k={self.top_k}."
            )
        if int(manifest.get("num_trials", -1)) != int(self.num_trials):
            raise ValueError(
                f"Clone cache at {cache_dir} has num_trials="
                f"{manifest.get('num_trials')}, dataset asks for "
                f"num_trials={self.num_trials}."
            )

        # The root is compared resolved, not as written: the builder stores an
        # absolute path and a caller may well pass "../data/..." for the same
        # directory. A cache built from a *different* tree would be silently
        # served here, which is the trap this closes.
        cached_root = manifest.get("root")
        if cached_root is None or Path(cached_root).resolve() != Path(self.root_dir).resolve():
            raise ValueError(
                f"Clone cache at {cache_dir} was built from root {cached_root!r}, "
                f"but this dataset reads {str(self.root_dir)!r}. The tensors would "
                f"not be this tree's data; rebuild the cache."
            )

        sim_ids = np.load(os.path.join(cache_dir, CACHE_SIM_IDS_FILENAME))
        # Header-only read: mmap_mode gives the shape without paging in 1.6 GB.
        x_shape = np.load(
            os.path.join(cache_dir, CACHE_X_FILENAME), mmap_mode="r"
        ).shape
        if len(sim_ids) != x_shape[0]:
            raise ValueError(
                f"Clone cache at {cache_dir} is inconsistent: "
                f"{CACHE_SIM_IDS_FILENAME} names {len(sim_ids)} sims but "
                f"{CACHE_X_FILENAME} has {x_shape[0]} rows. Every row would be "
                f"served under the wrong sim's name; rebuild the cache."
            )

        self._cache_rows = {str(name): int(row) for row, name in enumerate(sim_ids)}

        # Every sim this dataset will serve has to be in the cache. Checked here
        # rather than at the first __getitem__ that misses, because a partial
        # cache should fail before the job is queued, not four hours in -- and
        # under a DataLoader the KeyError from a worker is a good deal harder to
        # read than this message.
        missing = [
            name
            for name in (os.path.basename(item["sim_dir"]) for item in self.items)
            if name not in self._cache_rows
        ]
        if missing:
            shown = ", ".join(missing[:10])
            more = "" if len(missing) <= 10 else f", ... and {len(missing) - 10} more"
            raise ValueError(
                f"Clone cache at {cache_dir} is missing {len(missing)} of the "
                f"{len(self.items)} sims this dataset serves: {shown}{more}. "
                f"Rebuild the cache for this split."
            )

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

            * ``X_trials``: ``(num_trials, top_k, 45)`` float32, or
              ``(trial_subsample, top_k, 45)`` when subsampling is on. Slots for
              trials that were not loaded stay NaN.
            * ``trial_mask``: ``(num_trials,)`` bool (or ``(trial_subsample,)``),
              True where a trial was loaded. See the trap comment below.
            * ``y``: ``(44,)`` float32 selection coefficients.
        """
        item = self.items[i]
        sim_dir = item["sim_dir"]
        avail = item["available_trials"]
        y = item["y"]

        # Drawn here, above the cache branch, so the cached and the uncached
        # path consume the same one draw at the same point in the RNG stream and
        # therefore return the SAME subset for the same seed. With subsampling
        # off this is None and draws nothing at all.
        selection = self._draw_trial_indices()

        if self.cache_dir is not None:
            return self._getitem_cached(sim_dir, selection)

        if selection is not None:
            x_trials = torch.full(
                (len(selection), self.top_k, 45), float("nan"), dtype=torch.float32
            )
            trial_mask = torch.zeros((len(selection),), dtype=torch.bool)
            for out_slot, slot in enumerate(selection):
                # Only the chosen trials are read: with K=16 that is 16 gzipped
                # files per item instead of 25, which is the one place this
                # augmentation is also cheaper than the published path.
                x_trials[out_slot] = self._load_and_process_trial(
                    sim_dir, avail[slot]
                )
                trial_mask[out_slot] = 1.0
            return x_trials, trial_mask, y

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

    def _draw_trial_indices(self) -> Optional[List[int]]:
        """Pick which trial slots this item should carry, or ``None`` for all.

        Returns:
            ``None`` when subsampling is off -- the caller then takes the
            published all-trials path and no random number is drawn. Otherwise a
            sorted list of ``self._subsample_k`` distinct slot indices in
            ``[0, num_trials)``.

        Note:
            The indices are drawn with :func:`torch.randperm` on the **ambient**
            torch RNG, so the subset is fresh on every call (hence every epoch)
            and is reproducible from ``--seed`` alone -- including under
            ``num_workers > 0``, where torch re-derives each worker's seed from
            the main generator once per epoch.

            They are then **sorted**. Pooling over the trial dimension is a mean
            (``models/trials.py``), so the order cannot matter to the model;
            sorting makes a printed item readable and makes the cached and the
            uncached path comparable slot by slot.
        """
        if self._subsample_k is None:
            return None
        picks = torch.randperm(self.num_trials)[: self._subsample_k]
        return sorted(int(p) for p in picks)

    def _getitem_cached(
        self, sim_dir: str, selection: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return one simulation out of the clone cache.

        Args:
            sim_dir: Path to the sim directory; only its basename is used.
            selection: Trial slots to keep, from :meth:`_draw_trial_indices`, or
                ``None`` for every trial (the published path).

        Returns:
            The same 3-tuple as :meth:`__getitem__`. ``trial_mask`` is all True:
            the cache only ever contains sims that passed the "all
            ``num_trials`` trials present" filter, so there is no NaN-padded
            slot to mask.

        Raises:
            KeyError: If this sim is not in the cache. That means the cache was
                built from a different sim set, and quietly falling back to the
                slow path would hide it.
        """
        name = os.path.basename(sim_dir)
        row = self._cache_rows.get(name)
        if row is None:
            raise KeyError(
                f"{name} is not in the clone cache at {self.cache_dir} "
                f"({len(self._cache_rows)} sims cached). Rebuild the cache for "
                f"this split."
            )

        # Lazy open, once per process: a memmap opened before a DataLoader fork
        # would be shared by every worker, so it is opened on first use inside
        # the process that reads it.
        if self._cache_x is None:
            self._cache_x = np.load(
                os.path.join(self.cache_dir, CACHE_X_FILENAME), mmap_mode="r"
            )
            self._cache_theta = np.load(
                os.path.join(self.cache_dir, CACHE_THETA_FILENAME), mmap_mode="r"
            )

        # copy=True, not np.asarray: a slice of a mmap-opened array is read-only,
        # and torch.from_numpy on it yields a non-writable tensor plus a
        # UserWarning on every item. One copy per item is the price of a writable
        # tensor -- the values are identical either way.
        if selection is None:
            x_trials = torch.from_numpy(np.array(self._cache_x[row], copy=True))
            trial_mask = torch.ones((self.num_trials,), dtype=torch.bool)
        else:
            # Fancy-indexing the memmap reads only the chosen trial planes, and
            # the result is already a fresh array -- hence no second copy.
            x_trials = torch.from_numpy(
                np.asarray(self._cache_x[row][selection], dtype=np.float32)
            )
            trial_mask = torch.ones((len(selection),), dtype=torch.bool)
        y = torch.from_numpy(np.array(self._cache_theta[row], copy=True))
        return x_trials, trial_mask, y


def top_frequent_rows_source_sha1() -> str:
    """Hash the source of :func:`top_frequent_rows_tensor`.

    Returns:
        The SHA-1 hex digest of ``inspect.getsource(top_frequent_rows_tensor)``,
        UTF-8 encoded. The cache builder records it and every cache reader
        re-checks it, so editing that function -- including the trap-17 argsort
        or the trap-18 divisor -- invalidates every cache built before the edit
        instead of silently serving stale tensors.
    """
    src = inspect.getsource(top_frequent_rows_tensor)
    return hashlib.sha1(src.encode("utf-8")).hexdigest()


__all__ = [
    "load_gz_pickle",
    "load_pickle",
    "discover_sim_trials",
    "top_frequent_rows_tensor",
    "top_frequent_rows_source_sha1",
    "CNASimsDataset",
    "CACHE_X_FILENAME",
    "CACHE_THETA_FILENAME",
    "CACHE_SIM_IDS_FILENAME",
    "CACHE_MANIFEST_FILENAME",
]
