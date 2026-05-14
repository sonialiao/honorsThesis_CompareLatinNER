"""
llmner_tsv_predict.py
---------------------
Runs the llmNER zero-shot NER pipeline over gold TSV files and saves
predictions TSVs ready for evaluation.

Usage
-----
    python llmner_tsv_predict.py

    Requires downloading and installing the llmner package:
        pip install "git+https://github.com/plncmm/llmner.git"
    
    Backend options (set BACKEND below):
 
    "ministral"     — free, local. Requires Ollama running:
                       ollama pull ministral-3:14b && ollama serve
 
    "claude"     — Anthropic API. Requires litellm as a proxy:
                       pip install litellm
                       litellm --model anthropic/claude-opus-4-6
                   litellm starts on http://localhost:4000 and translates
                   OpenAI-style calls to Anthropic's API automatically.
                   Set ANTHROPIC_API_KEY in environment
 
    "openai"     — OpenAI API directly. Set OPENAI_API_KEY in environment.
 
Output TSV columns
------------------
    id | form | lila:lemma | lila:token | BIO_gold | BIO_pred
"""
# ── Imports ───────────────────────────────────────────────────────────────────

import os
import csv
from pathlib import Path

from llmner import ZeroShotNer, FewShotNer # pyright: ignore[reportAttributeAccessIssue]
from llmner.data import AnnotatedDocument, Annotation # pyright: ignore[reportAttributeAccessIssue]

from load_from_tsv import load_gold_tsv, sentences_to_pipeline_input

# # For token counting, uncomment:
# import openai # llmner uses this internally
# from types import MethodType

# ── Model Configuration ────────────────────────────────────────────────────

BACKEND = "claude"   # "ministral" | "claude" | "openai"

if BACKEND == "ministral":
    MODEL = "ministral-3:14b"
    os.environ["OPENAI_API_KEY"]  = "INNOVAFERTANIMUSMUTANTASDICEREFORMAS"  # placeholder
    os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"

elif BACKEND == "claude":
    # Using Claude's "OpenAI-style" endpoint for LLM NER
    MODEL = "claude-opus-4-6"

    os.environ["OPENAI_API_KEY"]  = os.environ.get("ANTHROPIC_API_KEY", "")
    os.environ["OPENAI_API_BASE"] = "https://api.anthropic.com/v1"
 
elif BACKEND == "openai":
    MODEL = "gpt-4o-mini"
    # set OPENAI_API_KEY in environment before running (see LLM NER docs)
 
else:
    raise ValueError(f"Unknown BACKEND: {BACKEND!r}  (choose ministral / claude / openai)")

# ── Constants ────────────────────────────────────────────────────────────

PREDICTION_ITERATION = "0"  # update manually to iterate on predictions
MODE = "few-shot"           # "zero-shot" | "few-shot"

INPUT_TEXTS = [
    "Cicero_PhilippicaOratio_CicPhi01_GOLD.tsv",
    "Juvenal_sat_1_3_GOLD.tsv",
    "Tacitus_TacHistoriae_TacHist1_GOLD.tsv",
]
INPUT_DIR  = "Ner-Latin-RANLP/Latin_Gold_Data/"

OUTPUT_DIR = f"predictions_{PREDICTION_ITERATION}_{BACKEND}/{MODE}/"
OUTPUT_FIELDS = ["id", "form", "lila:lemma", "lila:token", "BIO_gold", "BIO_pred"]

BATCH_SIZE = 32

# ── Entity definitions ────────────────────────────────────────────────────────
# llmNER uses natural-language descriptions.
# The LABEL_MAP then maps those keys to the BIO tag used in the gold data.

ENTITIES = {
    "person":       "A personal name, e.g. Cicero, Caesar, Catilina",
    "group":        "A people, tribe, army, or collective group, e.g. Romani, Belgae, Senatus",
    "geography":     "A geographic location: city, region, river, mountain, e.g. Roma, Gallia, Rhenus",
}

LABEL_MAP = {
    "person":       "PERS",
    "group":        "GRP",
    "geography":     "GEO",
}

EXAMPLES = [
    # Pliny Epistulae 6.16
    AnnotatedDocument(
        text="Erat Miseni classemque imperio praesens regebat.",
        annotations={
            Annotation(start=5, end=10, label="geography"),
        },
    ),
    # Pliny Epistulae 10.96
    AnnotatedDocument(
        text="Cognitionibus de Christianis interfui numquam: ideo nescio quid et quatenus aut puniri soleat aut quaeri.",
        annotations={
            Annotation(start=17, end=27, label="group"),
        },
    ),
    # Virgil Aeneid 1.6-7
    AnnotatedDocument(
        text="inferretque deōs Latiō, genus unde Latīnum, Albānīque patrēs, atque altae moenia Rōmae",
        annotations={
            Annotation(start=17, end=21, label="geography"),
            Annotation(start=35, end=41, label="group"),
            Annotation(start=44, end=52, label="group"),
            Annotation(start=81, end=86, label="geography"),
        },
    ),
    # Virgil Aeneid 7.1
    AnnotatedDocument(
        text="Tu quoque litoribus nostris, Aeneia nutrix,",
        annotations={
            Annotation(start=29, end=34, label="person"),
        },
    ),
    # Livy Ab Urbe Condita 28.1
    AnnotatedDocument(
        text="Cum transitu Hasdrubalis quantum in Italiam declinauerat belli tantum leuatae Hispaniae uiderentur, renatum ibi subito par priori bellum est.",
        annotations={
            Annotation(start=13, end=23, label="person"),
            Annotation(start=36, end=42, label="geography"),
            Annotation(start=78, end=86, label="geography")
        },
    ),
    # Livy Ab Urbe Condita 28.3
    AnnotatedDocument(
        text="Scipio ubi animaduertit dissipatum passim bellum, et circumferre ad singulas urbes arma diutini magis quam magni esse operis, retro uertit iter.",
        annotations={
            Annotation(start=0, end=5, label="person")
        },
    ),
]

# ── BIO conversion ────────────────────────────────────────────────────────────

def spans_to_bio(tokens: list[str], text: str, annotations) -> list[str]:
    """
    Convert llmNER Annotation objects into a BIO tag list aligned to `tokens`.

    Annotation objects expose: .start  .end  .label
    (.end is exclusive, matching Python slice convention)

    We use a char-index lookup (same approach as the notebook) rather than
    overlap arithmetic, since llmNER's offsets are relative to the joined
    string produced by sentences_to_pipeline_input(..., as_strings=True).

    Parameters
    ----------
    tokens      : original token list for the sentence
    text        : the joined string that was passed to model.predict()
    annotations : set/list of Annotation objects from the llmNER doc

    Returns
    -------
    List of BIO tag strings, one per token.
    """
    # Build char → (label, is_begin) from sorted annotations
    char_label: dict[int, tuple[str, bool]] = {}
    for ann in sorted(annotations, key=lambda a: a.start):
        short = LABEL_MAP.get(ann.label, ann.label.upper())
        for i in range(ann.start, ann.end):
            char_label[i] = (short, i == ann.start)

    bio_tags: list[str] = []
    cursor = 0
    for token in tokens:
        start = text.index(token, cursor)
        end   = start + len(token)
        cursor = end

        tag = "O"
        for i in range(start, end):
            if i in char_label:
                label, is_begin = char_label[i]
                tag = f"B-{label}" if is_begin else f"I-{label}"
                break

        bio_tags.append(tag)

    return bio_tags



# ── Token Counter ───────────────────────────────────────────────────────────────
# class TokenTracker:
#     def __init__(self, input_rate=0.20, output_rate=0.60):
#         self.total_input = 0
#         self.total_output = 0
#         self.input_rate = input_rate   # $ per 1M tokens
#         self.output_rate = output_rate # $ per 1M tokens

#     def log_usage(self, response):
#         """Captures usage from the OpenAI-compatible response object."""
#         if hasattr(response, 'usage'):
#             self.total_input += response.usage.prompt_tokens
#             self.total_output += response.usage.completion_tokens

#     def get_summary(self):
#         total_cost = (self.total_input / 1_000_000 * self.input_rate) + \
#                      (self.total_output / 1_000_000 * self.output_rate)
#         return {
#             "input_tokens": self.total_input,
#             "output_tokens": self.total_output,
#             "total_tokens": self.total_input + self.total_output,
#             "estimated_cost_usd": round(total_cost, 6)
#         }

# # Initialize tracker
# tracker = TokenTracker(input_rate=0.20, output_rate=0.60)

# # Monkey-patching the OpenAI client to intercept usage data
# # This works for both OpenAI v0.x and v1.x
# try:
#     # For OpenAI v1.x+
#     original_create = openai.resources.chat.completions.Completions.create
#     def patched_create(self, *args, **kwargs):
#         response = original_create(self, *args, **kwargs)
#         tracker.log_usage(response)
#         return response
#     openai.resources.chat.completions.Completions.create = patched_create
# except AttributeError:
#     # For OpenAI v0.x
#     original_create = openai.ChatCompletion.create
#     def patched_create(*args, **kwargs):
#         response = original_create(*args, **kwargs)
#         tracker.log_usage(response)
#         return response
#     openai.ChatCompletion.create = patched_create



# ── Core runner ───────────────────────────────────────────────────────────────

def run_and_save(
    input_tsv:  str | Path,
    output_tsv: str | Path,
    model:      ZeroShotNer,
    batch_size: int = 32,
) -> None:
    """
    Run the llmNER pipeline over every sentence in input_tsv and write
    predictions to output_tsv.

    Parameters
    ----------
    input_tsv  : path to the gold TSV
    output_tsv : path for the output predictions TSV
    model      : a contextualized ZeroShotNer instance
    batch_size : number of sentences to pass to model.predict() at once
    """
    sentences      = load_gold_tsv(input_tsv)
    pipeline_input = sentences_to_pipeline_input(sentences, as_strings=True)

    print(f"Loaded {len(sentences)} sentences from {input_tsv}")

    output_tsv = Path(output_tsv)
    output_tsv.parent.mkdir(parents=True, exist_ok=True)

    total_tokens = 0
    mismatches   = 0

    with output_tsv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDS, delimiter="\t")
        writer.writeheader()

        for batch_start in range(0, len(sentences), batch_size):
            batch_sents = sentences[batch_start : batch_start + batch_size]
            batch_texts = pipeline_input[batch_start : batch_start + batch_size]

            # model.predict() returns one Document per input string
            annotated_docs = model.predict(batch_texts)

            for sent, text, doc in zip(batch_sents, batch_texts, annotated_docs):
                pred_labels = spans_to_bio(sent["tokens"], text, doc.annotations)

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


# ── Main Function ──────────────────────────────────────────────────────────
if __name__ == "__main__":

    if MODE == "zero-shot":
        print(f"Loading model: {MODEL} ({BACKEND}) for zero-shot NER")
        ner_model = ZeroShotNer(model=MODEL)
        ner_model.contextualize(entities=ENTITIES)
        
    elif MODE == "few-shot":
        print(f"Loading model: {MODEL} ({BACKEND}) for few-shot NER")
        ner_model = FewShotNer(model=MODEL)
        ner_model.contextualize(entities=ENTITIES, examples=EXAMPLES)
    else:
        raise ValueError(f"Invalid MODE: {MODE} (choose zero-shot or few-shot)")

    for text in INPUT_TEXTS:
        input_path  = f"{INPUT_DIR}{text}"
        output_path = f"{OUTPUT_DIR}{text}"

        run_and_save(
            input_tsv=input_path,
            output_tsv=output_path,
            model=ner_model,
            batch_size=BATCH_SIZE,
        )
    
    # === uncomment for token cost summary ===
    # summary = tracker.get_summary()
    # print("\n" + "="*40)
    # print("FINAL TOKEN & COST REPORT")
    # print("-" * 40)
    # print(f"Input Tokens:  {summary['input_tokens']:,}")
    # print(f"Output Tokens: {summary['output_tokens']:,}")
    # print(f"Total Tokens:  {summary['total_tokens']:,}")
    # print(f"Estimated Cost: ${summary['estimated_cost_usd']:.6f}")
    # print("="*40)