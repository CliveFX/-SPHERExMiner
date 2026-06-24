# LLM Filter Experiment

This is a pretend/experimental layer for using a local LLM as a late-stage
triage filter over LuxQuarry/SPHEREx candidate spectra.

The point is not to replace the GPU detector. The point is to test whether an
LLM can read compact structured evidence packets and produce useful skeptical
reviews:

- Does this target contain an anomalous narrowband excess?
- Is the evidence more consistent with a real injected/beam-like signal or an
  artifact?
- Which flags, support counts, detector patterns, or competing peaks matter?
- Which candidates deserve human review?

The script uses the OpenAI-compatible Ollama endpoint on `spark-09dd:11434`.

## Files

- `llm_filter_experiment.py` builds packets, sends prompts, and scores replies.
- `prompts/` contains generated prompt text.
- `data/` contains generated serialized spectra, challenge packets, and truth
  ledgers.
- `responses/` contains model replies and scoring summaries.

## Basic Usage

Build the 100 injected spectra library and 5/10/25/100 challenge packets:

```bash
python LLMfilter/llm_filter_experiment.py build
```

Run a smoke test against Qwen 3.5 27B:

```bash
python LLMfilter/llm_filter_experiment.py smoke --challenge-size 5
```

Run prompt variants over a few challenge sets:

```bash
python LLMfilter/llm_filter_experiment.py run-examples --sizes 5 10
```

Available local models can be checked with:

```bash
curl http://spark-09dd:11434/v1/models
```

## Current Caveat

The challenge truth is held out in separate truth-ledger files. The prompt sees
known examples plus unlabeled challenge spectra. This is a triage experiment, not
a calibrated classifier yet.

One intentional challenge class is a paired decoy: the known-example section may
show an injected version of a target, while the challenge section contains the
same target from the baseline/uninjected run. This tests whether the model is
reasoning from spectral evidence or simply keying off target identity.
