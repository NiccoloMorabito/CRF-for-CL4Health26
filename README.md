# Two-Stage LLM Pipeline for Clinical Case Report Form (CRF) Filling

[![Paper](https://img.shields.io/badge/ACL%20Anthology-2026.bionlp--2.1-b31b1b)](https://aclanthology.org/2026.bionlp-2.1/)
[![DOI](https://img.shields.io/badge/DOI-10.18653%2Fv1%2F2026.bionlp--2.1-blue)](https://doi.org/10.18653/v1/2026.bionlp-2.1)

Code for my system for the **CRF Filling Shared Task 2026** (CL4Health), described in:

> **NM at CRF Filling 2026: A Two-Stage LLM Pipeline for Clinical CRF Population**
> Niccolò Morabito. *Proceedings of the BioNLP 2026 (Shared Tasks)*, San Diego, July 2026, pp. 1–5. Association for Computational Linguistics.
> [Paper (PDF)](https://aclanthology.org/2026.bionlp-2.1.pdf) · [ACL Anthology](https://aclanthology.org/2026.bionlp-2.1/)

## The problem

Clinical research needs structured data, but most patient information lives in free-text notes. A **Case Report Form (CRF)** is the structured questionnaire researchers fill in by hand, reading each note.

The task: given a clinical note about a patient with **dyspnea** (shortness of breath), automatically fill a CRF of **134 items** (symptoms, findings, lab values...), choosing for each item one value among a predefined list of valid options. Notes come in **English and Italian** (parallel datasets).

What makes it hard:

- **Extreme sparsity**: ~95% of the CRF items are `unknown` for any given note. A system must say something only when the text states it explicitly. Over-predicting is punished, so precision matters as much as coverage.
- **Almost no supervision**: only **10** gold note–CRF pairs are available for the target form.
- **No inference allowed**: negations, suspicions, plans and history must not be turned into positive values.

## The approach

Separate *finding* information from *trusting* it.

```
clinical note ──► [Stage 1: extraction]  ──► candidate CRF (JSON) ──► [Stage 2: verification] ──► final CRF
                  Gemini 3 Flash, 3-shot                              a different, more conservative LLM
                  recall-oriented                                     keeps only fields with explicit evidence
```

1. **Extraction.** A few-shot prompted LLM (Gemini 3 Flash, 3 of the 10 gold examples) reads the note once and returns a single JSON object with all the fields it can populate. Output is constrained to the valid options of each field (a Pydantic model built dynamically from the task's option lists).
2. **Verification.** A *different* LLM receives the note and the candidate CRF and removes every field that is not backed by clear, explicit textual evidence. It is forbidden to add fields or change values: it can only filter.
3. **Choosing the verifier.** The verifier is not the strongest model but the most *conservative* one, i.e. the one that produces fewer predictions and fewer false positives when used for extraction (Llama 4 Maverick 17B for English, GPT-OSS 120B for Italian). Such models have lower recall when used alone, but that conservatism is exactly what is useful for filtering.

The full prompts are in [`src/prompts.py`](src/prompts.py) and in the paper's appendix.

## Results

Experiments on the development set show that the verification stage substantially reduces unsupported predictions while preserving most of the correct extractions, improving macro F1 over extraction alone. On the official test set the system reaches a **macro F1 of 0.56 in both English and Italian**. Details and per-language tables are in the [paper](https://aclanthology.org/2026.bionlp-2.1.pdf).

## Repository

```
src/
  prompts.py          extraction and verification system prompts
  models.py           builds the CRF schema (Pydantic, Literal options) from the task's valid options
  crf_utils.py        provider-agnostic LLM calls (Gemini, OpenAI, Groq, Gemma), few-shot message building
  dataset_utils.py    loading of the shared-task datasets from Hugging Face (NLP-FBK)
  *.ipynb             experiment notebooks (extraction, post-elaboration/verification, Colab runs)
  prepare_submission.py
CRF-filling-CL4Health2026/
  scoring.py          local macro-F1 scoring (from the organizers)
  check_submission_format.py   validator + ZIP for Codabench (from the organizers)
  your_submission_data*/       predictions of every model/configuration tried
```

### Run it

```bash
pip install -r src/requirements.txt
# put your own API keys in src/api_keys.py (git-ignored), never in notebooks
```

The shared-task datasets are downloaded from the [NLP-FBK collection on Hugging Face](https://huggingface.co/collections/NLP-FBK/crf-filling-sharedtask-cl4health2026-6752d58a1768411d9961db1d). Score a submission with:

```bash
python3 CRF-filling-CL4Health2026/scoring.py --submission_path <file>.jsonl --language en
```

## Citation

```bibtex
@inproceedings{morabito-2026-nm,
    title = "{NM} at {CRF} Filling 2026: A Two-Stage {LLM} Pipeline for Clinical {CRF} Population",
    author = "Morabito, Niccol{\`o}",
    booktitle = "Proceedings of the {B}io{NLP} 2026 (Shared Tasks)",
    month = jul,
    year = "2026",
    address = "San Diego, California, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.bionlp-2.1/",
    doi = "10.18653/v1/2026.bionlp-2.1",
    pages = "1--5",
}
```
