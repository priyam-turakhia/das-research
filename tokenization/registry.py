"""Central registry: tokenizer dispatch and shared on-disk config schema.

Single source of truth for:
- The set of tokenizer types and the class each name maps to.
- The `tokenizer_config.json` schema written by every tokenizer's `save()`.
- Detecting a tokenizer's type from a saved directory.
- Loading a tokenizer of any type from a saved directory.

Tokenizer modules import only the schema helpers (`write_config`, `read_config`,
`CONFIG_VERSION`) — `TOKENIZER_CLASSES` and `load_tokenizer` are built lazily to
avoid an import cycle.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CONFIG_VERSION = 1
CONFIG_FILENAME = "tokenizer_config.json"

TOKENIZER_TYPES = (
    "spm_bpe",
    "spm_unigram",
    "morfessor",
    "morfessor_semi",
    "morph_bpe",
)


def write_config(
    path: str | Path,
    tokenizer_type: str,
    vocab_size: int,
    **extra: Any,
) -> None:
    """Write the standard tokenizer_config.json at `path`.

    Required fields: tokenizer_type, vocab_size, version.
    `extra` allows tokenizer-specific HF integration keys (e.g. auto_map).
    """
    if tokenizer_type not in TOKENIZER_TYPES:
        raise ValueError(
            f"Unknown tokenizer_type {tokenizer_type!r}; expected one of {TOKENIZER_TYPES}"
        )
    config: dict[str, Any] = {
        "tokenizer_type": tokenizer_type,
        "vocab_size": vocab_size,
        "version": CONFIG_VERSION,
    }
    config.update(extra)
    config_path = Path(path) / CONFIG_FILENAME
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def read_config(path: str | Path) -> dict[str, Any]:
    """Read tokenizer_config.json. Returns {} if the file does not exist."""
    config_path = Path(path) / CONFIG_FILENAME
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def detect_tokenizer_type(model_path: str | Path) -> str:
    """Detect a tokenizer's type from its saved directory.

    Reads `tokenizer_type` from the v1 config. Falls back to legacy keys
    (`tokenizer_class`, `model_type`) and finally to file-marker probing so
    pre-migration model directories still load.
    """
    model_path = Path(model_path)
    config = read_config(model_path)

    if "tokenizer_type" in config:
        ttype = config["tokenizer_type"]
        if ttype not in TOKENIZER_TYPES:
            raise ValueError(f"Unknown tokenizer_type in config: {ttype!r}")
        return ttype

    # Legacy compat: older saves wrote tokenizer_class + model_type instead.
    legacy_class = config.get("tokenizer_class", "")
    legacy_model = config.get("model_type", "")
    if legacy_class == "MorphBPETokenizer" or legacy_model == "morph_bpe":
        return "morph_bpe"
    if "Unigram" in legacy_class or legacy_model == "unigram":
        return "spm_unigram"
    if "BPE" in legacy_class or legacy_model == "bpe":
        return "spm_bpe"
    if "SemiSupervisedMorfessor" in legacy_class or legacy_model == "morfessor_semi":
        return "morfessor_semi"
    if "Morfessor" in legacy_class or legacy_model == "morfessor":
        return "morfessor"

    # Last-resort: probe for marker files.
    if (model_path / "morfessor_model.pkl").exists() and (model_path / "spm.model").exists():
        return "morph_bpe"
    if (model_path / "model.pkl").exists():
        return "morfessor"
    if (model_path / "spm.model").exists():
        return "spm_bpe"

    raise ValueError(f"Cannot detect tokenizer type from {model_path}")


def get_tokenizer_classes() -> dict[str, type]:
    """Lazily build the name->class map. Imports happen here to avoid cycles."""
    from tokenization.morfessor import MorfessorTokenizer
    from tokenization.morfessor_semi import SemiSupervisedMorfessorTokenizer
    from tokenization.morph_bpe import MorphBPETokenizer
    from tokenization.spm_bpe import SPMBPETokenizer
    from tokenization.spm_unigram import SPMUnigramTokenizer

    return {
        "spm_bpe": SPMBPETokenizer,
        "spm_unigram": SPMUnigramTokenizer,
        "morfessor": MorfessorTokenizer,
        "morfessor_semi": SemiSupervisedMorfessorTokenizer,
        "morph_bpe": MorphBPETokenizer,
    }


def load_tokenizer(model_path: str | Path, tokenizer_type: str | None = None):
    """Instantiate the right tokenizer class and call its `load()`.

    If `tokenizer_type` is None it is detected from the saved directory.
    """
    model_path = Path(model_path)
    if tokenizer_type is None:
        tokenizer_type = detect_tokenizer_type(model_path)

    classes = get_tokenizer_classes()
    if tokenizer_type not in classes:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

    logger.info(f"Loading {tokenizer_type} tokenizer from {model_path}")
    tokenizer = classes[tokenizer_type]()
    tokenizer.load(str(model_path))
    return tokenizer
