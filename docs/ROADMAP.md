# Roadmap

The order of work, from making the suite solid to running it everywhere. Each step is useful on its own. Each test
is pre-registered before any row is scored.

## 1. Make the suite its own package (the split)
- Move `eval.bench`, `eval.bindprobes`, `eval.binding` and `eval.riders` here. kotodama's checkpoint loader becomes a
  `koto:` plugin, and the bench imports nothing from kotodama.
- The bench owns the closed-set scorer. kotodama's server keeps its own copy with a parity test against the bench's.
- The rules for contributions and documentation come from kotodama and anamnesis. So do the gates: test retention,
  timelessness, referents, import closure and legibility.
- A `python -m kotodama.eval.binding` shim keeps existing launch scripts working.
- Repository: private under anima-research, then public after the docs pass.

## 2. Close the open validity questions
- **joinlite:** the I / you queries open with "Right now". Test that opening the way the third-person "Now" was
  tested. A joinlite without "Now" is a new battery version.
- **formupdate:** in prose, the update cost reverses sign. Test whether a recency cue causes it: "… is put in it"
  makes the latest value the gold answer in C. If it does, that grounds a formupdate-v3.
- **Riders:** report each register on its own, and add a balanced few-shot condition (`fair`). Tag the banked
  items by the kind of update they hold, swap or reset.
- **Reliability and floors:** for every slice, record split-half reliability, between-model spread, and the share of
  models at chance or ceiling. A future test of whether slices group by construct runs only on informative slices, corrected for their reliability.
- **Fluid controls:** an in-context n-gram baseline is scored on every fluid family. Every panel model is above it.
  It solves F4's period rows read after a motif token, and the registry flags that subset. F1 and F2 are invariant
  to the corpus frequency of their nonce words. `PRIOR-ART.md` §2 lists the prior work.

## 3. Report the floor's format load, and test it against scale
The 0-shot readings are format- and frame-loaded wherever they were tested:
- C0's 0-shot rung spreads by 24 to 48 points across frames on 8/8 models;
- bindprobes attr gains 19 to 46 points from worked examples on 8/8.

Every report states this beside the 0-shot floor. The open question is whether it shrinks with scale. On the
1–8B panel it does not: C0's mean spread is 34.8 points below 2B and 31.3 at 7–8B. Only three models on that panel
are above 2B.
The test is a pre-registered regression of each 0-shot frame spread on log parameters and log tokens, run on the
full base zoo, including bases of 14B and above.

## 4. Run the solid suite on every base model available
- Score the whole audited suite on every accessible base model, with the clean readings, the floors and their flags.
- Score the slices that sit at chance on most of the 0.5–8B panel on a large base model before reading that floor
  as a property of the eval. Llama-3.1-405B base lifts the deixis addressee slice (κ .60) and leaves swapdyn's
  0-shot addressee at chance, which now waits on a validity review of its 0-shot frame. A second large family is
  next. The registry records both (`PANEL_RELIABILITY`).
- Add looped models with a controllable loop count (Ouro first) for a depth axis. The panel's own kotodama-tsumugi
  is looped, so a depth claim needs them.

## 5. A protocol for instruct models
Chat models need changes to the closed-set method that scores base-model prompts. The protocol needs:
- a chat rendering of each battery, in which the user turn holds the context and the question, and the assistant
  turn opens with the answer frame;
- the same closed candidate set;
- the rendering audit applied to the chat rendering itself, so instruct scores carry their own flags.

The first validity check is a base model and its instruct model from one family, such as Qwen2.5-7B.

## 6. The public face
The Binding Atlas is rebuilt from the registry and organised by what each eval measures. Each section shows the
clean readings for every model in the zoo, beside the 0-shot floor. The page opens with the results that hold.
