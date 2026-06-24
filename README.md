# Comparing Named Entity Recognition Approaches for Classical Latin
Undergraduate Honors Thesis — Sonia Liao, University of Massachusetts Amherst, Computer Science Major / Classical Civilization Minor, Class of 2026

[Link to Manuscript (UMass Login to PATH portal)](https://honorspaths.honors.umass.edu/chc-paths/completion_forms_document_content/4577/thesis)

---

## Models

### 1. LatinBERT (Fine-tuned)
Pipeline adapted from [NER-AncientLanguages/Ner-Latin-RANLP](https://github.com/NER-AncientLanguages/Ner-Latin-RANLP), the codebase accompanying:

> Beersmans et al. (2023). *Training and Evaluation of Named Entity Recognition Models for Classical Latin.*

The underlying LatinBERT model (Bamman & Burns, 2020) was pre-trained on Classical Latin data and subsequently fine-tuned for NER on the Herodotos Project's annotations (Erdmann et al., 2019). This pipeline serves as the Latin-specific baseline. See `latinBERT_tsv_predict.py`.

### 2. Multilingual BERT (mBERT / WikiNEuRal)
Uses [`Babelscape/wikineural-multilingual-ner`](https://huggingface.co/Babelscape/wikineural-multilingual-ner), accessed via the HuggingFace Transformers library, described in:

> Tedeschi et al. (2021). *WikiNEuRal: Combined Neural and Knowledge-based Silver Data Creation for Multilingual NER.* Findings of EMNLP 2021. [(ACL Anthology)](https://aclanthology.org/2021.findings-emnlp.215/)

Trained on Wikipedia articles in 9 modern languages — including Spanish, French, and Italian, all Romance descendants of Latin — this model tests cross-lingual transfer learning rather than Latin-specific training. See `mBERT_tsv_predict.py`.

### 3. LLM — Zero-shot and Few-shot Prompting
Prompting pipeline adapted from [plncmm/llmner](https://github.com/plncmm/llmner):

> Villena et al. (2024). *LLM NER: A Python library for zero-shot NER with large language models.*

Three models were tested using this framework:

| Model | Parameters | Notes |
|---|---|---|
| **Mistral-NeMo** | 12B | Open-weight, Mistral AI × NVIDIA (2024). Run locally via [Ollama](https://ollama.com). ([HuggingFace](https://huggingface.co/mistralai/Mistral-Nemo-Base-2407)) |
| **Ministral 3** | 14B | Open-weight, Mistral AI (early 2026). One of the newest locally-deployable open-weight models. |
| **Claude Opus 4.7** | — | Proprietary, Anthropic (April 2026). State-of-the-art on difficult reasoning tasks. |

See `llmNER_tsv_predict.py` for zero-shot inference and `llmNER_fewShot_min_ex.ipynb` for few-shot runs.

---

## Dataset

Gold standard evaluation data from Beersmans et al. (2023), sourced from the [LASLA database](http://www.lasla.uliege.be/). Three texts are included:

- Tacitus, *Historiae* Book 1
- Cicero, *Orationes Philippicae* No. 1
- Juvenal, *Saturae* Parts 1–3

Tokens are annotated in the **BIO scheme** with three entity types: `PRS` (persons), `GEO` (geographical places), and `GRP` (peoples/groups). The Herodotos Project dataset (Erdmann et al., 2019), used to train the LatinBERT pipeline, was intentionally excluded from evaluation to avoid data leakage.

Gold data is in `Ner-Latin-RANLP/Latin_Gold_Data/`. Silver training data covering a broader range of Classical Latin authors is in `Ner-Latin-RANLP/Latin_Silver_Data/`.

---

## Setup & Dependencies

### 1. Clone this repo
```bash
git clone https://github.com/sonialiao/honorsThesis_CompareLatinNER.git
cd honorsThesis_CompareLatinNER
```

### 2. Pull third-party repositories
These are excluded from this repo and must be cloned separately into the project root:

```bash
# LatinBERT model and scripts
git clone https://github.com/dbamman/latin-bert.git
```
```bash
# NER pipeline and gold/silver data
git clone https://github.com/NER-AncientLanguages/Ner-Latin-RANLP.git
```

Then download the LatinBERT model weights into `latin-bert/models/`:
```bash
cd latin-bert
bash scripts/download.sh
cd ..
```

### 3. Load the subword text encoder
`subword_text_encoder.py` is a standalone extraction from the deprecated
[`tensor2tensor`](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/data_generators/text_encoder.py)
library, refactored for Python 3.10 compatibility. You need to pull this in as well.

```bash
wget -O subword_text_encoder.py https://raw.githubusercontent.com/tensorflow/tensor2tensor/master/tensor2tensor/data_generators/text_encoder.py
```

### 4. Install dependencies
**Python 3.10 required.**

```bash
pip install -r requirements.txt
```

> Note: if you have `uv` installed:
>   - rebuild the virtual environment with `uv sync`

### 5. LLM models (for the prompting pipeline)
Install [Ollama](https://ollama.com), then pull the open-weight models:
```bash
ollama pull mistral-nemo
ollama pull ministral   # check Ollama's model library for the exact tag
```

For Claude Opus 4.7, set your Anthropic API key:
```bash
export ANTHROPIC_API_KEY=your_key_here
```

### Expected directory structure after setup
```
honorsThesis_CompareLatinNER/
├── latin-bert/                  # cloned in step 2
├── Ner-Latin-RANLP/             # cloned in step 2
├── LatinNERpipeline.py          # included in this repo (refactored)
├── LatinBERT_min_example.ipynb
├── latinBERT_tsv_predict.py
├── LatinNERpipeline.py
├── llmNER_fewShot_min_ex.ipynb
├── llmNER_min_example.ipynb
├── llmNER_tsv_predict.py
├── load_from_tsv.py
├── mBERT_min_example.ipynb
├── mBERT_tsv_predict.py
├── predictions_0_*/
├── results_evaluation.ipynb
└── subword_text_encoder.py      # loaded in step 3
```

---

## Modifications: 

The `LatinNERpipeline.py` file in this repo is functionally identical to the one in `Ner-Latin-RANLP/code`, with some changes added to allow it to run under a (slightly) newer version of Python (3.10).

Subword text encoder was pulled from [here](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/data_generators/text_encoder.py) due to module dependency issues that could not be teased out.

Header rows were added to `Juvenal_sat_1_3_GOLD.tsv` and `Tacitus_TacHistoriae_TacHist1_GOLD.tsv` to enable automated loading. The script exposes two helper functions — one returning sentences as joined strings, one as lists of tokens — to accommodate the different input formats expected by each model.

---

## Results

Prediction outputs for all models are in the `predictions_0_*/` directories. Evaluation metrics and cross-model analysis are in `results_evaluation.ipynb`.

---

## Third-party Code

| Component | Source |
|---|---|
| LatinBERT NER pipeline | [NER-AncientLanguages/Ner-Latin-RANLP](https://github.com/NER-AncientLanguages/Ner-Latin-RANLP) |
| LLM NER prompting framework | [plncmm/llmner](https://github.com/plncmm/llmner) |
| Subword text encoder | [tensorflow/tensor2tensor](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/data_generators/text_encoder.py) |
| WikiNEuRal mBERT model | [Babelscape/wikineural-multilingual-ner](https://huggingface.co/Babelscape/wikineural-multilingual-ner) |

---

## Updates: June 2026

- changed package management from Anaconda to uv, which is written in Rust and a more modern tool
    - `requirements.txt` has been replaced by the combination of `pyproject.toml` and `uv.lock`, the file has been retained for records but uv shouldn't need it anymore

With `uv`, after installation and set up:
    - run python scripts with `uv run <script name>`
    - for Jupyter Notebooks, select the kernel directly located in `.venv`
    - all packages will be auto-installed and managed by uv via `pyproject.toml` and `uv.lock`
