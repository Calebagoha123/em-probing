# Methods Scaffold

## Section goal

Write this section so that another student could rerun the study from scratch.

Use past tense.
Prefer short declarative sentences.
Follow the Model Essay 4 style:
- start with an experimental-design paragraph
- define the data and model organism clearly
- explain the monitor construction
- explain evaluation and baselines
- make the metrics and splits explicit

## Recommended section structure

### 3.1 Experimental design and problem statement

This subsection should explain the whole study in one page or less.

Suggested first sentence:

> We evaluated whether a residual-stream monitor of realized emergent misalignment remained more stable across prompt wrappers than behavioral misalignment rate on the same model organism.

Then define the design in plain language:
- choose one EM model organism and its checkpoints
- generate responses for the same prompt set under multiple prompt wrappers
- externally judge each response for alignment
- extract response-conditioned residual-stream activations
- fit a linear monitor at the final misaligned checkpoint on held-out splits
- compare monitor behavior against behavioral misalignment rate across prompt wrappers

You should also explicitly state what the paper did not test in the primary experiment:
- no claim of pre-generation early warning
- no claim of causal localization
- checkpoint-transfer analysis reserved for later extension

### 3.2 Data and model

This subsection should answer:
- what model did you use?
- what checkpoints?
- what prompts?
- what prompt wrappers?
- what labels?

Include:
- base model name
- fine-tuned checkpoint repository or source
- exact prompt set source
- number of prompts
- prompt wrapper conditions
- response sampling settings
- judge model and label thresholds

Be explicit about the final dataset object.

Example wording shape:

> For each checkpoint and prompt-wrapper condition, we generated responses to the Turner et al. evaluation prompts. Each response was scored by an external judge model for alignment and coherence, then converted into a ternary label {aligned, misaligned, invalid}. Only aligned and misaligned responses were used for monitor fitting and evaluation.

### 3.3 Activation extraction

State exactly what activation you extracted.

For the primary paper:
- response-conditioned activations
- residual stream / hidden state at the final token of the full prompt-response sequence
- one vector per layer per example

You must also justify this choice:
- it aligned the monitor with realized behavior
- it did not attempt to claim pre-behavior prediction

If you later add a pre-generation appendix, keep it out of the main Methods subsection or put it in a brief “additional analysis” note.

### 3.4 Monitor construction

This subsection is the heart of the paper.

Define:
- what “monitor” means
- what layer selection means
- what direction-fitting procedure means
- how the held-out split works

A clean version:
- split prompts into train / validation / test by unique prompt
- on train: compute a mean-difference direction between misaligned and aligned activations for each layer
- on validation: select the layer and threshold that best separate the two classes
- freeze both layer and threshold
- on test: evaluate the monitor on neutral, hhh, and evil conditions without further tuning

Why this matters:
- it answers the circularity concern directly
- it shows you are not just rediscovering the same labels on the same data

### 3.5 Baselines and comparison targets

You need at least one clear comparison target.

Primary baseline:
- behavioral misalignment rate by prompt wrapper

Optional secondary comparison:
- simple probe or monitor fit separately per condition, if you want to show what is gained by freezing one monitor

Keep the baseline section simple.
The main comparison is internal monitor robustness versus behavioral robustness.

### 3.6 Evaluation metrics

Define each metric and why it matters.

Recommended metrics:
- AUROC on held-out aligned vs misaligned examples
  - threshold-independent discrimination quality
- accuracy or balanced accuracy at a frozen threshold
  - easier to interpret operationally
- mean monitor score by prompt condition
  - captures score drift across wrappers
- behavioral misalignment rate by prompt condition
  - direct comparison target
- robustness gap or range across wrappers
  - the main primary-question statistic

You should define prompt robustness operationally.

Example:

> We defined a monitor as more prompt-robust than behavior if its held-out discrimination and score distribution varied less across wrapper conditions than the corresponding behavioral misalignment rate.

If you want a sharper metric:
- report the range across conditions
- or the standard deviation across conditions
- or the maximum absolute deviation from neutral

Pick one primary robustness metric and keep the others secondary.

### 3.7 Validation and anti-circularity controls

This subsection should be explicit.

Include:
- prompt-level split rather than row-level split
- no tuning on test prompts
- external judge rather than the target model self-grading
- optional manual audit of a small stratified sample

Suggested sentence:

> To avoid circularity, we fit the monitor only on training prompts, selected layer and threshold on validation prompts, and reported all prompt-robustness results on held-out prompts that were never used during monitor construction.

## Flash figure guidance

Include one simple figure in the Methods section.

Best option:
- a single workflow diagram

Recommended boxes:
1. EM checkpoints
2. prompt set x wrapper conditions
3. response generation
4. external judging
5. response-conditioned activation extraction
6. held-out monitor fitting
7. evaluation under neutral / hhh / evil
8. comparison: monitor robustness vs behavioral robustness

Caption should be self-sufficient:
- name the model organism
- define the wrappers
- define what the monitor was fit on
- define what was compared

## Methods checklist

Before leaving Methods, confirm that a reader can answer:
- exactly which model and checkpoints were used
- exactly which prompts and wrappers were used
- exactly how labels were assigned
- exactly which activation was extracted
- exactly how the monitor was fit
- exactly how train/validation/test were split
- exactly what the baseline was
- exactly what metric decided the main result

## Minimal subsection template

### 3 Methods
### 3.1 Experimental design and problem statement
### 3.2 Data and model
### 3.3 Response generation and labeling
### 3.4 Activation extraction
### 3.5 Residual-stream monitor construction
### 3.6 Baselines and evaluation metrics
### 3.7 Validation protocol and anti-circularity controls
