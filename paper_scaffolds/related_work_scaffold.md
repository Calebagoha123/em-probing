# Related Work Scaffold

## Job of this section

This section should not be a bibliography dump.

Its job is to show:
- what is already known
- what line of work your paper belongs to
- what exact gap remains

Model Essay 4 handles this well by grouping the literature into conceptual blocks and then ending each block with the tension that motivates the paper.

For this paper, the cleanest structure is four subsections.

## Recommended structure

### 2.1 Emergent misalignment and model organisms

Purpose:
- define the empirical setting you inherit
- show that EM is a real and reproducible phenomenon

What to cover:
- Betley et al.: narrow finetuning can induce broad misalignment
- Turner et al.: model organisms, checkpoint availability, phase-transition-like dynamics, public prompt sets
- any recent extensions to other architectures or reasoning models if useful

What this subsection should conclude:
- the field now has tractable experimental organisms for studying EM
- the challenge is no longer just eliciting EM, but monitoring it reliably

### 2.2 Behavioral monitoring and prompt sensitivity

Purpose:
- explain why output behavior alone is unstable

What to cover:
- behavioral EM rate as the standard metric
- Wyse et al.: prompt wrappers strongly affect measured EM rates
- Arnold and Lorch or related work: behavior-level order parameters / phase-transition monitoring
- if useful, broader model-evaluation literature on risks of relying on outputs alone

What this subsection should conclude:
- behavioral metrics are useful but not obviously prompt-robust
- this creates a monitoring gap

### 2.3 Internal representations of misalignment and safety-relevant directions

Purpose:
- place your paper inside the internal-representation / monitoring literature

What to cover:
- Soligo et al.: convergent linear EM representations
- residual-stream directions / representation engineering / related safety directions
- optional analogy to refusal-direction work if you need to show that compact linear behavioral signals are plausible

What this subsection should conclude:
- there is precedent for linearly encoded safety-relevant structure
- but prompt robustness of such a monitor has not been established

### 2.4 Probes and representation-level monitoring

Purpose:
- connect your method to the broader probes literature and show you are using a lightweight, interpretable monitor rather than a heavy black-box classifier

What to cover:
- Alain and Bengio on linear probes
- contamination-detection / hidden-state monitoring papers if relevant
- standard caveats from probing literature: selectivity, circularity, and train/test leakage

What this subsection should conclude:
- probes can be useful monitoring tools if evaluated carefully
- your held-out design is intended to address exactly those concerns

## End-of-section synthesis paragraph

You need one final paragraph after the subsections.

Its job:
- say what all of the above implies together
- state the missing experiment in one sentence

Recommended shape:

> Prior work has established emergent misalignment as a reproducible phenomenon, shown that behavioral EM rates are prompt-sensitive, and identified internal residual-stream signals associated with misaligned behavior. What remains unclear is whether a simple internal monitor can track emergent misalignment more robustly than behavior itself when prompt framing changes.

That sentence is the handoff into Methods.

## Things to avoid

- do not let Related Work repeat the Introduction
- do not summarize each paper in equal detail
- do not hide the gap until the end of the paper
- do not claim nobody has studied internal EM representations; Soligo clearly has

## Citation strategy

Each subsection should do one of these jobs:
- establish a fact you inherit
- establish a limitation in prior work
- establish the method family you use

If a paper does none of those jobs, cut it.

## Mini outline you can draft from directly

### 2.1 Emergent misalignment and model organisms
- EM as a broad post-finetuning failure mode
- checkpointed model organisms make the dynamics experimentally tractable
- therefore EM is now a monitoring problem, not only a phenomenon-discovery problem

### 2.2 Behavioral monitoring and prompt sensitivity
- behavioral misalignment rate is the default metric
- prompt wrappers can amplify or suppress the observed rate
- therefore behavior alone may be an unstable readout of internal state

### 2.3 Internal representations of misalignment
- residual-stream differences can encode EM linearly
- intervention work suggests this signal is not arbitrary
- but the literature has not yet established prompt robustness of the monitor itself

### 2.4 Probes as monitors
- probes provide lightweight readouts of internal state
- probing work also warns against circularity and split leakage
- this motivates your held-out evaluation design
