"""
run_pipeline.py
---------------
Runs the LatinBERT NER pipeline over a gold TSV file and saves a
predictions TSV ready for evaluation.

Usage
-----
    python run_pipeline.py \
        --input  path/to/gold.tsv \
        --output path/to/predictions.tsv \
        --model  Herodotos_trained_lat_BERT_hypopt_params \
        --encoder ../../latin-bert/models/subword_tokenizer_latin/latin.subword.encoder

Output TSV columns
------------------
    id | form | lila:lemma | lila:token | BIO_gold | BIO_pred

The output file is directly usable by an evaluation script — each row
has both the gold label and the model prediction for that token.
"""

import csv
import warnings
from pathlib import Path

import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

warnings.filterwarnings("ignore", message=".*Python version.*google.*")
warnings.filterwarnings("ignore", message=".*IProgress not found.*")

from transformers import AutoModelForTokenClassification

from LatinNERpipeline import LatinNerPipeline, tokenizer
from load_from_tsv import load_gold_tsv, sentences_to_pipeline_input


# ── Output columns ────────────────────────────────────────────────────────────

OUTPUT_FIELDS = ["id", "form", "lila:lemma", "lila:token", "BIO_gold", "BIO_pred"]
    
def run_and_save(
    input_tsv: str | Path,
    output_tsv: str | Path,
    pipeline: LatinNerPipeline,
    batch_size: int = 32,
) -> None:
    """
    Run the pipeline over every sentence in input_tsv and write
    predictions to output_tsv.

    Parameters
    ----------
    input_tsv  : path to the gold TSV
    output_tsv : path for the output predictions TSV
    pipeline   : a ready LatinNerPipeline instance
    batch_size : number of sentences to process per pipeline call
    """
    sentences = load_gold_tsv(input_tsv)
    pipeline_input = sentences_to_pipeline_input(sentences)

    print(f"Loaded {len(sentences)} sentences from {input_tsv}")

    output_tsv = Path(output_tsv)
    output_tsv.parent.mkdir(parents=True, exist_ok=True)

    total_tokens = 0
    mismatches = 0

    with output_tsv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDS, delimiter="\t")
        writer.writeheader()

        # Process in batches
        for batch_start in range(0, len(sentences), batch_size):
            batch_sents = sentences[batch_start : batch_start + batch_size]
            batch_input = pipeline_input[batch_start : batch_start + batch_size]

            predictions = pipeline(batch_input, split_on_words=True)

            for sent, pred in zip(batch_sents, predictions):
                pred_labels = pred["labels"]

                # Guard: label count should match token count
                if len(pred_labels) != len(sent["tokens"]):
                    mismatches += 1
                    print(
                        f"  WARNING: token/label count mismatch in sentence "
                        f"starting at id={sent['ids'][0]}  "
                        f"(tokens={len(sent['tokens'])}, labels={len(pred_labels)})"
                    )
                    # Pad or truncate so the row count stays consistent
                    pred_labels = _align_labels(pred_labels, len(sent["tokens"]))

                for i, token_meta in enumerate(sent["meta"]):
                    row = {
                        "id":          token_meta.get("id", ""),
                        "form":        token_meta.get("form", ""),
                        "lila:lemma":  token_meta.get("lila:lemma", ""),
                        "lila:token":  token_meta.get("lila:token", ""),
                        "BIO_gold":    sent["gold"][i],
                        "BIO_pred":    pred_labels[i],
                    }
                    writer.writerow(row)
                    total_tokens += 1

            print(
                f"  Processed sentences "
                f"{batch_start + 1}–{batch_start + len(batch_sents)} / {len(sentences)}"
            )

    print(f"\nDone. {total_tokens} tokens written to {output_tsv}")
    if mismatches:
        print(f"  ({mismatches} sentences had token/label mismatches — check warnings above)")


def _align_labels(labels: list, expected_len: int) -> list:
    """Pad with 'O' or truncate so label list matches expected length."""
    if len(labels) < expected_len:
        return labels + ["O"] * (expected_len - len(labels))
    return labels[:expected_len]


# === Constants & Main ===

PREDICTION_ITERATION = "0"
INPUT_TEXTS = [
    'Cicero_PhilippicaOratio_CicPhi01_GOLD.tsv', 
    'Juvenal_sat_1_3_GOLD.tsv', 
    'Tacitus_TacHistoriae_TacHist1_GOLD.tsv'
]

INPUT_DIR = "Ner-Latin-RANLP/Latin_Gold_Data/"
OUTPUT_DIR = f"predictions_{PREDICTION_ITERATION}_latinBERT/"
MODEL_NAME = "Herodotos_trained_lat_BERT_hypopt_params"
MODEL_PATH = "./Ner-Latin-RANLP/code/Herodotos_trained_lat_BERT_hypopt_params"
BATCH_SIZE = 32

if __name__ == "__main__":

    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
    pipeline =  LatinNerPipeline(model=model, tokenizer=tokenizer)

    for text in INPUT_TEXTS:
        input_path = f"{INPUT_DIR}{text}"
        output_path = f"{OUTPUT_DIR}{text}"

        # print(f"input path = {input_path}")
        # print(f"output path = {output_path}")
        
        run_and_save(
            input_tsv=input_path,
            output_tsv=output_path,
            pipeline=pipeline,
            batch_size=BATCH_SIZE,
        )