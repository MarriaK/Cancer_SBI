"""Create and load the fixed train/test split of simulations.

Ported from ``SetTransformer_NPE/data_preprocessing.py`` (creation) and from the
identical ``with open("train_test_split.pkl", "rb")`` blocks at the top of every
``*/main.py`` and every evaluation script (loading).

The split is over whole *simulations*, not trials: a sim's 25 trials all land on
the same side, so the test set contains no trial from a training tumour.

Nothing here runs at import time -- the original
``SetTransformer_NPE/data_preprocessing.py`` was a top-to-bottom script that read
a hard-coded ``"../Guassian_Normal/simulation_outputs"`` and wrote
``train_test_split.pkl`` into the current directory the moment it was imported.
"""

import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np

PathLike = Union[str, os.PathLike]


def list_sim_ids(root_dir: PathLike) -> List[str]:
    """List the simulation directory names in the order the split used.

    Args:
        root_dir: Directory containing ``sim*/``.

    Returns:
        Lexicographically sorted names, e.g. ``["sim1", "sim10", "sim100", ...]``.

    Note:
        This is a plain ``sorted()``, i.e. **lexicographic**, matching
        ``SetTransformer_NPE/data_preprocessing.py:9``. It is deliberately NOT
        the numeric sort used by
        :func:`cancer_sbi.data.clone_sets.discover_sim_trials`: the split was
        drawn from this lexicographic ordering, so reproducing it requires the
        same ordering. The datasets re-sort their own members afterwards, so the
        difference never reaches a batch.
    """
    return sorted(d for d in os.listdir(root_dir) if d.startswith("sim"))


def create_split(
    root_dir: PathLike,
    test_size: float = 0.2,
    random_state: int = 123,
) -> Tuple[np.ndarray, np.ndarray]:
    """Draw the deterministic train/test split over simulation ids.

    Args:
        root_dir: Directory containing ``sim*/``.
        test_size: Fraction held out, 0.2 in the original.
        random_state: Seed handed to scikit-learn. 123 in the original, which is
            what makes the split identical on every machine.

    Returns:
        ``(train_ids, test_ids)`` as numpy arrays of ``str``, exactly the objects
        that were pickled by ``SetTransformer_NPE/data_preprocessing.py:13-19``.

    Note:
        Uses ``sklearn.model_selection.train_test_split``, imported here rather
        than at module scope so that the rest of the package does not need
        scikit-learn installed. Reimplementing the shuffle by hand would not
        reproduce scikit-learn's exact permutation and would silently change
        which sims are in the test set.
    """
    from sklearn.model_selection import train_test_split  # local: see docstring

    sim_ids = np.array(list_sim_ids(root_dir))
    train_ids, test_ids = train_test_split(
        sim_ids, test_size=test_size, random_state=random_state
    )
    return train_ids, test_ids


def save_split(path: PathLike, train_ids: Sequence[str], test_ids: Sequence[str]) -> Path:
    """Pickle a split to disk in the original's layout.

    Args:
        path: Destination ``.pkl`` file. Parent directories are created.
        train_ids: Simulation names for training.
        test_ids: Simulation names for testing.

    Returns:
        The path written, as a :class:`pathlib.Path`.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {"train_ids": train_ids, "test_ids": test_ids}
    with out_path.open("wb") as handle:
        pickle.dump(payload, handle)
    return out_path


def load_split(path: PathLike) -> Tuple[Sequence[str], Sequence[str]]:
    """Load a pickled train/test split.

    Args:
        path: The split pickle, holding ``{"train_ids": ..., "test_ids": ...}``.

    Returns:
        ``(train_ids, test_ids)`` exactly as stored -- numpy arrays of ``str``
        for the pickles currently on disk. They are passed straight to the
        dataset classes, which only ever build a ``set`` of them or sort them.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        KeyError: If the pickle lacks ``train_ids`` or ``test_ids``.

    Note:
        Preserved from Plain_NPE/main.py:16 and Plain_NPE/z-score_violin.py:55,
        which read ``"../Base_NPE/train_test_split.pkl"`` while Base and
        SetTransformer each read their own local ``train_test_split.pkl``.
        Trap 15: DominantClone has no split file of its own and borrows Base's.
        The two committed pickles are byte-identical today, so the three models
        do share one split -- but that is a coincidence of the files, not
        something the code guarantees. The path therefore stays an explicit
        argument here instead of being defaulted per model.
        See docs/REFACTOR_NOTES.md.
    """
    split_path = Path(path)
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found at: {split_path}")
    with split_path.open("rb") as handle:
        split = pickle.load(handle)
    return split["train_ids"], split["test_ids"]


__all__ = ["list_sim_ids", "create_split", "save_split", "load_split"]
