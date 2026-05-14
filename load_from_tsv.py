"""
load_from_tsv.py
----------------
Reusable loader for the gold-standard NER TSV format produced from the
LASLA/LiLa pipeline.

Expected TSV columns (tab-separated, with header row):
    id | form | lila:lemma | lila:token | BIO_gold

Returns a list of sentences. Each sentence is a dict:
    {
        "tokens": list of str        # the 'form' values
        "ids":    list of int/str    # the 'id' values  
        "meta":   list of dict       # full row data (lemma, token URI, etc.)
        "gold":   list of str        # BIO_gold labels, one per token
    }

Sentence boundaries are detected by a reset in the numeric `id` column
(i.e. when id goes from N back to a value <= previous id, a new sentence
has started). This matches the LASLA TSV convention.

You can also call load_gold_tsv(..., presplit=True) if your TSV already
has blank lines between sentences.
"""

import csv
from pathlib import Path
from typing import List, Dict, Any


def _parse_id(raw: str) -> int | str:
    """Return int if possible, else str (for multiword tokens like '7-8')."""
    try:
        return int(raw.strip())
    except ValueError:
        return raw.strip()


def load_gold_tsv(
    filepath: str | Path,
    presplit: bool = False,
    delimiter: str = "\t",
) -> List[Dict[str, Any]]:
    """
    Load a gold NER TSV file into a list of sentence dicts.

    Parameters
    ----------
    filepath  : path to the .tsv file
    presplit  : if True, use blank lines as sentence boundaries instead of
                id-reset detection (use when your TSV already has blank lines)
    delimiter : column separator (default: tab)

    Returns
    -------
    List of sentence dicts, each with keys:
        'tokens' : list[str]   — word forms (input to the pipeline)
        'ids'    : list        — token ids from the TSV
        'gold'   : list[str]   — gold BIO labels
        'meta'   : list[dict]  — all columns for each token
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    sentences = []
    current: Dict[str, list] = {"tokens": [], "ids": [], "gold": [], "meta": []}

    def _flush(current):
        if current["tokens"]:
            sentences.append({k: list(v) for k, v in current.items()})
        return {"tokens": [], "ids": [], "gold": [], "meta": []}

    with filepath.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)

        if reader.fieldnames is None:
            raise ValueError("TSV file is empty or missing a header row.")

        # Normalise column names
        reader.fieldnames = [f.strip() for f in reader.fieldnames]

        required = {"id", "form", "BIO_gold"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"TSV missing required columns: {missing}")

        prev_numeric_id = None

        for raw_row in reader:
            row = {k.strip(): (v.strip() if v else "") for k, v in raw_row.items() if k}

            # ── Blank-line sentence boundary (presplit mode) ────────────
            if presplit:
                if all(v == "" for v in row.values()):
                    current = _flush(current)
                    prev_numeric_id = None
                    continue

            # ── Skip comment / metadata lines ───────────────────────────
            raw_id = row.get("id", "")
            if raw_id.startswith("#") or raw_id == "":
                continue

            token_id = _parse_id(raw_id)

            # ── Id-reset sentence boundary (default mode) ───────────────
            if not presplit:
                numeric_id = token_id if isinstance(token_id, int) else None
                if (
                    numeric_id is not None
                    and prev_numeric_id is not None
                    and numeric_id <= prev_numeric_id
                ):
                    current = _flush(current)
                prev_numeric_id = numeric_id if numeric_id is not None else prev_numeric_id

            # ── Append token ─────────────────────────────────────────────
            current["tokens"].append(row["form"])
            current["ids"].append(token_id)
            current["gold"].append(row["BIO_gold"])
            current["meta"].append(row)

    _flush(current)  # flush final sentence

    if not sentences:
        raise ValueError("No sentences found — check delimiter and file format.")

    return sentences


def sentences_to_pipeline_input(
    sentences: List[Dict[str, Any]],
    as_strings: bool = False,
) -> List[List[str]] | List[str]:
    """
    Convert loaded sentences into pipeline input format.
 
    Parameters
    ----------
    sentences  : output of load_gold_tsv()
    as_strings : if False (default), returns a list of token lists —
                     [['Quousque', 'tandem', ...], [...]]
                 if True, returns a list of joined strings —
                     ['Quousque tandem ...', '...']
                 Use as_strings=True with split_on_words=False in the pipeline,
                 and as_strings=False with split_on_words=True.
    """
    if as_strings:
        return [" ".join(sent["tokens"]) for sent in sentences]
    return [sent["tokens"] for sent in sentences]