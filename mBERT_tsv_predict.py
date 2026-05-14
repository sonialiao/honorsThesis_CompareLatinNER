"""
mbert_tsv_predict.py
--------------------
Runs the mBERT (wikineural-multilingual-ner) NER pipeline over a gold TSV
file and saves a predictions TSV ready for evaluation.

Usage
-----
    python mbert_tsv_predict.py

Output TSV columns
------------------
    id | form | lila:lemma | lila:token | BIO_gold | BIO_pred

The key difference from latinBERT: the HuggingFace `pipeline` with
`grouped_entities=True` returns entity *spans*, not per-token labels.
We convert those back to BIO tags aligned to the original token list.
"""

import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import csv
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*Python version.*google.*")
warnings.filterwarnings("ignore", message=".*IProgress not found.*")
warnings.filterwarnings("ignore", message=".*`resume_download` is deprecated.*")
warnings.filterwarnings("ignore", message=".*`grouped_entities` is deprecated.*")

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

from load_from_tsv import load_gold_tsv, sentences_to_pipeline_input


# ── Output columns ────────────────────────────────────────────────────────────

OUTPUT_FIELDS = ["id", "form", "lila:lemma", "lila:token", "BIO_gold", "BIO_pred"]

# ── Label mapping ─────────────────────────────────────────────────────────────
# WikiNEuRal uses PER / ORG / LOC / MISC — map to the tag scheme in gold data
LABEL_MAP = {
    "PER":  "PERS",   # match gold label convention: PERSONS
    "ORG":  "GRP",    # match gold label convention: GROUPS
    "LOC":  "GEO",    # match gold label convention: GEOGRAPHY
        # NOTE: Philippica has LOC labels, correct all into GEO in evaluation script
    "MISC": "MISC",   # gold label has no MISC
}


def spans_to_bio(tokens: list[str], entities: list[dict]) -> list[str]:
    """
    Convert grouped-entity spans returned by the HuggingFace pipeline back
    into a BIO tag list aligned to `tokens`.

    The pipeline returns character-level offsets (start/end) relative to the
    joined input string.  We reconstruct those offsets from the token list so
    we can match each token to an entity span.

    Parameters
    ----------
    tokens   : original token list for the sentence
    entities : list of dicts from nlp(), each with keys
               'entity_group', 'start', 'end', 'word', 'score'

    Returns
    -------
    List of BIO tag strings, one per token.
    """
    # Build character offset ranges for each token in the joined string.
    # Tokens are joined with single spaces (same as sentences_to_pipeline_input).
    token_spans = []
    pos = 0
    for tok in tokens:
        token_spans.append((pos, pos + len(tok)))
        pos += len(tok) + 1  # +1 for the space separator

    bio_tags = ["O"] * len(tokens)

    for ent in entities:
        ent_start = ent["start"]
        ent_end   = ent["end"]
        label     = LABEL_MAP.get(ent["entity_group"], ent["entity_group"])

        first = True
        for i, (tok_start, tok_end) in enumerate(token_spans):
            # Token overlaps with entity span
            if tok_end > ent_start and tok_start < ent_end:
                bio_tags[i] = f"B-{label}" if first else f"I-{label}"
                first = False

    return bio_tags


def run_and_save(
    input_tsv: str | Path,
    output_tsv: str | Path,
    nlp,
    batch_size: int = 32,
) -> None:
    """
    Run the mBERT pipeline over every sentence in input_tsv and write
    predictions to output_tsv.

    Parameters
    ----------
    input_tsv  : path to the gold TSV
    output_tsv : path for the output predictions TSV
    nlp        : a HuggingFace token-classification pipeline instance
    batch_size : number of sentences to process per pipeline call
    """
    sentences = load_gold_tsv(input_tsv)
    # mBERT pipeline takes plain strings
    pipeline_input = sentences_to_pipeline_input(sentences, as_strings=True)

    print(f"Loaded {len(sentences)} sentences from {input_tsv}")

    output_tsv = Path(output_tsv)
    output_tsv.parent.mkdir(parents=True, exist_ok=True)

    total_tokens = 0
    mismatches = 0

    with output_tsv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDS, delimiter="\t")
        writer.writeheader()

        for batch_start in range(0, len(sentences), batch_size):
            batch_sents  = sentences[batch_start : batch_start + batch_size]
            batch_input  = pipeline_input[batch_start : batch_start + batch_size]

            # Pipeline accepts a list of strings and returns a list of entity lists
            batch_entities = nlp(batch_input)

            # When batch size == 1 the pipeline may return a flat list of dicts;
            # normalise to always be a list-of-lists
            if batch_input and isinstance(batch_entities[0], dict):
                batch_entities = [batch_entities]

            for sent, entities in zip(batch_sents, batch_entities):
                pred_labels = spans_to_bio(sent["tokens"], entities)

                if len(pred_labels) != len(sent["tokens"]):
                    mismatches += 1
                    print(
                        f"  WARNING: token/label mismatch at id={sent['ids'][0]} "
                        f"(tokens={len(sent['tokens'])}, labels={len(pred_labels)})"
                    )
                    pred_labels = _align_labels(pred_labels, len(sent["tokens"]))

                for i, token_meta in enumerate(sent["meta"]):
                    writer.writerow({
                        "id":         token_meta.get("id", ""),
                        "form":       token_meta.get("form", ""),
                        "lila:lemma": token_meta.get("lila:lemma", ""),
                        "lila:token": token_meta.get("lila:token", ""),
                        "BIO_gold":   sent["gold"][i],
                        "BIO_pred":   pred_labels[i],
                    })
                    total_tokens += 1

            print(
                f"  Processed sentences "
                f"{batch_start + 1}-{batch_start + len(batch_sents)} / {len(sentences)}"
            )

    print(f"\nDone. {total_tokens} tokens written to {output_tsv}")
    if mismatches:
        print(f"  ({mismatches} sentences had token/label mismatches — check warnings above)")


def _align_labels(labels: list, expected_len: int) -> list:
    """Pad with 'O' or truncate so label list matches expected length."""
    if len(labels) < expected_len:
        return labels + ["O"] * (expected_len - len(labels))
    return labels[:expected_len]


# ── Constants & Main ──────────────────────────────────────────────────────────

PREDICTION_ITERATION = "0"
INPUT_TEXTS = [
    'Cicero_PhilippicaOratio_CicPhi01_GOLD.tsv', 
    'Juvenal_sat_1_3_GOLD.tsv', 
    'Tacitus_TacHistoriae_TacHist1_GOLD.tsv'
]

INPUT_DIR = "Ner-Latin-RANLP/Latin_Gold_Data/"
OUTPUT_DIR = f"predictions_{PREDICTION_ITERATION}_mBERT/"
MODEL_NAME  = "Babelscape/wikineural-multilingual-ner"
BATCH_SIZE = 32

if __name__ == "__main__":

    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
    nlp       = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)

    for text in INPUT_TEXTS:
        input_path = f"{INPUT_DIR}{text}"
        output_path = f"{OUTPUT_DIR}{text}"

        # print(f"input path = {input_path}")
        # print(f"output path = {output_path}")
        
        run_and_save(
            input_tsv=input_path,
            output_tsv=output_path,
            nlp=nlp,
            batch_size=BATCH_SIZE,
        )