"""Command-line entry points, and the little bits they share.

Three commands replace the scripts that used to be run by hand from inside a
model folder:

    ===========================================  =================================
    old                                          new
    ===========================================  =================================
    ``cd SetTransformer_NPE && python data_preprocessing.py``
                                                 ``python -m cancer_sbi.cli.make_split``
    ``cd Base_NPE && python main.py``            ``python -m cancer_sbi.cli.train``
    ``cd Base_NPE && python z-score_violin.py``  ``python -m cancer_sbi.cli.evaluate``
    ===========================================  =================================

Two rules the old scripts broke and these do not:

* **No path is implied by the working directory.** The originals opened
  ``"train_test_split.pkl"`` and read a relative simulation directory, so they
  only worked when run from inside their own folder. Here every path is an
  argument, with an environment variable as the fallback.
* **Nothing heavy is imported until the arguments parse.** Each command imports
  torch, sbi and the rest of the package *inside* ``main()``, so ``--help``
  works on a machine that cannot import torch -- which is exactly the machine
  this package was written on.

This module itself imports nothing beyond the standard library and
:mod:`cancer_sbi.config`.
"""

import argparse
import os
from pathlib import Path
from typing import Optional

from cancer_sbi.config import PRESETS

#: Environment variable consulted when ``--data-root`` is not given.
DATA_ROOT_ENV = "CANCER_SBI_DATA_ROOT"

#: Environment variable consulted when ``--split`` is not given.
SPLIT_ENV = "CANCER_SBI_SPLIT"

#: Environment variable consulted when ``--out`` is not given.
OUT_ROOT_ENV = "CANCER_SBI_RUNS"

#: Default parent directory for run outputs, relative to where you run the
#: command. Shown in ``--help`` so it is never a surprise.
DEFAULT_RUNS_DIRNAME = "runs"


def add_model_argument(parser: argparse.ArgumentParser) -> None:
    """Add ``--model``, the choice of published model.

    Args:
        parser: The parser to extend.
    """
    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(PRESETS),
        help=(
            "Which published model to use. "
            "clonemlp = CloneMLP-NPE (was Base_NPE/), "
            "cloneatt = CloneAtt-NPE (was SetTransformer_NPE/), "
            "dominantclone = DominantClone-NPE (was Plain_NPE/), "
            "armtoken = ArmToken-NPE (new in matrix 5; no original folder). "
            "The choice fixes the encoder, the optimiser, the early-stopping "
            "rules and the checkpoint directory name."
        ),
    )


def add_data_root_argument(parser: argparse.ArgumentParser) -> None:
    """Add ``--data-root``, the simulation directory.

    Args:
        parser: The parser to extend.
    """
    parser.add_argument(
        "--data-root",
        type=Path,
        default=os.environ.get(DATA_ROOT_ENV),
        help=(
            "Directory holding the simulator's output, i.e. the folder that "
            "contains sim1/, sim2/, ... Each sim folder holds parameters.pkl "
            "and one subfolder per trial. The old scripts hard-coded a path "
            "relative to the model folder they were run from; there is no "
            f"default here. Falls back to ${DATA_ROOT_ENV}."
        ),
    )


def add_split_argument(parser: argparse.ArgumentParser, required_help: str = "") -> None:
    """Add ``--split``, the train/test split pickle.

    Args:
        parser: The parser to extend.
        required_help: Extra sentence appended to the help text.
    """
    parser.add_argument(
        "--split",
        type=Path,
        default=os.environ.get(SPLIT_ENV),
        help=(
            "Pickle holding {'train_ids': [...], 'test_ids': [...]}, as written "
            "by 'python -m cancer_sbi.cli.make_split'. All three models were "
            "trained on the same split file, so use the same one for all three "
            f"or the test sets stop being comparable. Falls back to ${SPLIT_ENV}. "
            + required_help
        ).strip(),
    )


def add_device_argument(parser: argparse.ArgumentParser) -> None:
    """Add ``--device``.

    Args:
        parser: The parser to extend.
    """
    parser.add_argument(
        "--device",
        default="auto",
        help=(
            "Device to use: 'auto' (the default) picks cuda when it is "
            "available and cpu otherwise, which is what every original script "
            "did on its first line. 'cpu' and 'cuda' force the choice."
        ),
    )


def resolve_device(name: str) -> str:
    """Turn ``--device`` into a concrete device string.

    Args:
        name: ``"auto"``, ``"cpu"``, ``"cuda"``, ``"cuda:1"``, ...

    Returns:
        ``"cuda"`` or ``"cpu"`` for ``"auto"``; anything else unchanged.

    Note:
        Imports torch lazily so that ``--help`` does not need it.
    """
    if name != "auto":
        return name
    import torch  # local import: see the module docstring.

    return "cuda" if torch.cuda.is_available() else "cpu"


def require_path(value: Optional[Path], flag: str, env_var: str) -> Path:
    """Fail with a readable message when a required path was not supplied.

    Args:
        value: What argparse produced, possibly ``None``.
        flag: The flag's name, for the message.
        env_var: The environment variable that can supply it instead.

    Returns:
        ``value`` as a :class:`~pathlib.Path`.

    Raises:
        SystemExit: With exit code 2, argparse's own code for a usage error.
    """
    if value is None:
        raise SystemExit(
            f"error: {flag} is required (or set ${env_var} in your environment)."
        )
    return Path(value)


def default_run_dir(model: str) -> Path:
    """Where a run writes when ``--out`` is not given.

    Args:
        model: The preset name.

    Returns:
        ``$CANCER_SBI_RUNS/<model>`` if that variable is set, else
        ``./runs/<model>`` relative to the current directory.
    """
    root = os.environ.get(OUT_ROOT_ENV, DEFAULT_RUNS_DIRNAME)
    return Path(root) / model


__all__ = [
    "DATA_ROOT_ENV",
    "SPLIT_ENV",
    "OUT_ROOT_ENV",
    "DEFAULT_RUNS_DIRNAME",
    "add_model_argument",
    "add_data_root_argument",
    "add_split_argument",
    "add_device_argument",
    "resolve_device",
    "require_path",
    "default_run_dir",
]
