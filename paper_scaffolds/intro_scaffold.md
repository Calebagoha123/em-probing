# Introduction Scaffold

## Working job of the introduction

Move from a broad AI safety / evaluation problem to the specific claim of the paper:

**Behavioral emergent misalignment measurements are prompt-sensitive, so we test whether a held-out residual-stream monitor of realized emergent misalignment is more prompt-robust than behavioral misalignment rate.**

This section should read like Model Essay 4:
- start with a broad problem that matters
- narrow to the concrete failure of current evaluation
- identify the gap in prior work
- end with your exact research question and contributions

## Paragraph map

### Paragraph 1: Wide shot

Open with the broader stakes of evaluating advanced language models.

What this paragraph should do:
- establish that frontier LLM evaluation matters because these systems are increasingly deployed in high-stakes settings
- note that behavioral evaluation is often the default way to assess safety-relevant properties
- introduce the idea that output behavior may not cleanly reveal internal model state

What not to do:
- do not start with emergent misalignment immediately
- do not start with definitions or paper summaries

Possible sentence job:
- “As language models become more capable and are deployed in increasingly consequential settings, reliable safety evaluation becomes a core technical problem rather than a peripheral concern.”

### Paragraph 2: Introduce emergent misalignment as the concrete phenomenon

Now zoom in to emergent misalignment.

What this paragraph should do:
- define emergent misalignment briefly
- explain why it is a particularly useful test case for monitoring
- mention that model-organism setups make it experimentally tractable

Key move:
- present EM not just as a strange behavior, but as a monitoring problem with checkpointed training dynamics

### Paragraph 3: Why behavior alone is inadequate

This is where Wyse enters.

What this paragraph should do:
- state that behavioral EM rates vary substantially under different system prompt wrappers
- explain the consequence: behavior is not obviously a stable proxy for the underlying state of the fine-tuned model
- motivate the need for a lower-level monitor

This is the paper's central problem statement.

Good framing:
- a model may look aligned under one wrapper and misaligned under another
- this weakens purely behavioral monitoring claims

### Paragraph 4: Prior work gets close but does not answer the question

Use this paragraph to build the gap.

What this paragraph should do:
- mention model organisms / checkpoint studies
- mention Soligo-style internal representation directions
- mention behavioral phase-transition work
- say that none of these directly test whether an internal EM monitor is itself prompt-robust on held-out prompts

Important:
- the gap is not “nobody studied EM”
- the gap is not “nobody used probes”
- the gap is specifically “nobody has shown that a simple internal EM monitor is more prompt-robust than behavior”

### Paragraph 5: Research question and study design

State the paper's main question in one sentence.

Recommended wording:

> This paper asks whether a held-out residual-stream monitor of realized emergent misalignment, fit at the final misaligned checkpoint, remains more stable across prompt wrappers than behavioral misalignment rate.

Then add one sentence explaining the core design:
- generate responses under multiple prompt wrappers
- label behavior externally
- extract response-conditioned activations
- fit a residual-stream monitor on held-out splits
- compare monitor robustness against behavioral robustness

### Paragraph 6: Contributions

End with a numbered list or compact prose list of contributions.

Recommended contribution structure:

1. We frame prompt sensitivity as a challenge for EM monitoring rather than merely for EM elicitation.
2. We test whether a simple held-out residual-stream monitor is more prompt-robust than behavioral misalignment rate.
3. We provide a lightweight monitoring pipeline on a public EM model organism that can be extended later to checkpoint-transfer analysis.

Keep the contribution claims modest.

Do not claim:
- mechanistic proof
- causal localization unless your intervention results become strong
- early warning in the introduction unless you later show true pre-behavior transfer cleanly

## Introduction checklist

Before moving on, confirm that the introduction:
- starts broad and ends specific
- has one clear central problem
- names the gap in one sentence
- states one exact research question
- distinguishes your paper from Soligo, Wyse, and behavioral phase-transition work
- avoids overselling early warning

## Optional closing sentence

If you want a strong final line for the introduction:

> If behavioral misalignment can be masked by prompt framing, then the practical question is no longer only whether models become misaligned, but whether we can monitor that state in a way that is less fragile than behavior itself.
