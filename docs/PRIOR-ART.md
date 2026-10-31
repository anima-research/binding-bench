# Prior art: what this benchmark builds on

The benchmark's claims should rest on published work wherever published work exists. This file maps each part of
the benchmark to the papers that ground it. For each paper, the relation column gives one of four words. The
benchmark **follows** the paper, **extends** it, **differs** from it, or **contrasts** with it. The file also records
how each citation was checked.

Novelty is marked **◆** and kept narrow. A ◆ entry names something that no cited paper does. It stays a candidate
until a targeted search finds no precedent.

**Checked** means that the paper was read through alphaXiv and that its text confirms the stated finding.
**Listed** means that the ID resolves and that the summary comes from a secondary note. A listed paper is not read yet.

## 1. Scoring and evaluation method

| paper | finding (checked text) | relation |
|---|---|---|
| Sclar, Choi, Tsvetkov, Suhr 2023, *Quantifying language models' sensitivity to spurious features in prompt design* (arXiv 2310.11324, ICLR 2024). Checked | Meaning-preserving format changes move few-shot accuracy by up to 76 points (LLaMA-2-13B). The spread is not removed by scale, more shots or instruction tuning. Format performance correlates only weakly between models, so a single fixed format can confound model comparisons. The paper recommends reporting a spread over plausible formats. | **follows**: the rendering audit's spread rule and its model-relative flags apply this recommendation to binding batteries. **extends**: it pairs the spread with off-construct levers and a causal exemplar test, and turns flags into registry rows. |
| Min et al. 2022, *Rethinking the role of demonstrations* (arXiv 2202.12837). Checked | Replacing demonstration labels with uniformly random labels costs only 0–5 points on classification and multi-choice tasks across 12 models. Format, label space and input distribution drive most of the gain. | **contrasts** with the flipped-exemplar test: flipping cell-matched exemplars *systematically* to the misread answer collapses accuracy (Qwen2.5-7B 98 → 2). Both results fit one reading: exemplars teach format and answer role more than the mapping. |
| Wei et al. 2023, *Larger language models do in-context learning differently* (arXiv 2303.03846). Checked | Flipped labels and semantically unrelated labels separate the in-context mapping from semantic priors; larger models override priors with flipped labels more readily. Classification only. | **follows**: the direct precedent for the flip test. Their scale prediction is checkable on the banked flip rows. |
| Ravulapalli 2026, *In-context binding capacity in language models* (arXiv 2609.30634). Checked | K lines `e: o`, D distractor tokens, query `The task for e is:`; top logit over the obligation pool; k\* (threshold) and K50 (logistic fit in log K, right-censored at 64). Pythia's k\* is 3–6 from 410M to 12B, and on continuous K50 Pythia 160M–2.8B sits inside a size law K50 = cN^0.82. The paper varies only K and D, not format. | **follows**: capacity-v1 uses the same design with names as entities. **differs**: on this benchmark's harness Pythia 160M–6.9B is flat (K50 2.2–3.1) and the size-law exponent is .38. ◆ The format dependence of K50 (in the prose frame OLMo-2-7B goes from >64 to 17.5 and Llama-3.1-8B from >64 to 37.6) is not studied in the paper. |

## 2. Fluid in-context learning against a crystallized control

| paper | finding (checked text) | relation |
|---|---|---|
| Pan et al. 2023, *What in-context learning "learns" in-context: disentangling task recognition and task learning* (arXiv 2305.09731). Checked | Separates task recognition (random labels) from task learning (abstract labels) across GPT-3, LLaMA and OPT, 350M–175B, with 8/16/32 shots; task learning emerges with scale. Classification only. | the nearest precedent for the fluid-vs-prior split. **differs**: its prior condition is randomized labels, not a matched recall item. |
| Niu et al. 2025, *Illusion or algorithm?* (arXiv 2505.11004, TMLR). Checked | Random-token pattern tasks beside tasks needing parametric knowledge, on the Pythia suite with checkpoints and Llama-3.x. ICL accuracy correlates with token frequency (r −.65 to −.91). | the nearest precedent for novel rule plus knowledge control across base models. Its frequency result motivates a token-frequency invariance test for F1/F2. |
| Chen et al. 2024, *ICLEval* (arXiv 2406.14955). Checked | Copying and rule learning measured apart from knowledge, 1.1B–65B; copying peaks early in pretraining, rule learning rises until about 200B tokens. | **follows**: separating ICL from knowledge, tracked over training. **differs**: knowledge is compared through separate benchmarks, not in the same frame. |
| Gupta, Sanders, Solar-Lezama, *Randomly sampled language reasoning problems* (arXiv 2501.02825). Checked | Random 3-state DFAs; LLMs (69–80) score below 4–6-gram baselines, and scale does not clearly help. | suggests a control this benchmark lacks: a surface-statistics (n-gram / copy) baseline for F4 and F1/F2. |
| Olsson et al. 2022, *In-context learning and induction heads* (arXiv 2209.11895); Liu et al., *Infini-gram* (arXiv 2401.17377). Listed | Induction heads copy the token that followed an earlier occurrence of the current prefix; the ∞-gram predicts from the longest suffix that occurred before. | **follows**: the fluid families' surface baseline is this copy rule run over the prompt alone (an in-context ∞-gram). It solves F4's period rows read after a motif token and nothing else. |
| Ghosh et al. 2026, *Fine-tuning vs in-context learning: a formal language view* (arXiv 2604.23267). Checked | Nonce probabilistic grammars scored by log-prob discrimination; ICL varies by family and is sensitive to the tokens used. | **follows**: judge-free log-prob scoring of nonce rules. Supports a token-vocabulary invariance check. |
| de Wynter, *Is in-context learning learning?* (arXiv 2509.10414). Checked | Formal-language tasks over arbitrary alphabets; peak accuracy at 50–100 shots; breaks under distribution shift. Four instruct models. | the fluid ladders are difficulty ladders with a censored midpoint, so a short shot range is reported as censoring, not as failure. |
| Yang et al., *DRE-Bench* (arXiv 2506.02648, ICLR 2026). Checked | Frames fluid vs crystallized intelligence explicitly, with procedurally generated ARC-style grids; 32B-and-up instruct and reasoning models. | framing precedent; not language, no in-benchmark crystallized control. |
| Sánchez et al., *Linguini* (arXiv 2409.12126). Checked | Linguistics-olympiad puzzles answerable from context; real languages; base and instruct models under 25%. | its transliteration ablation is a template for a rendering check on F1. |
| Tanzer et al., *MTOB* (arXiv 2309.16575). Checked | Learning Kalamang from one grammar book. | explicit learning from a document, not few-shot induction; out of scope. |

◆ **Candidate:** the fluid families learn new rules in language registers: morphology, symbol algebra, rebinding
chains and sequence induction. Each family is read against a crystallized control in the *same frame*. The panel holds eight base-model
families from 1B to 8B. A frame-matched check makes "same frame" a measured property: C0 stays
within 18 points over six frames at shots 6 and 3, on 8/8 models.

## 3. Binding and entity tracking

| paper | finding | relation |
|---|---|---|
| Guo et al. 2026, *When context changes: update failures in LLMs* (CICM, arXiv 2609.38866). Checked | Repeated overwrites leave models answering an old value while probes still read the current one (84.8% Qwen, 82.2% Llama); attention drifts to old values. Multiple choice hides old-value errors relative to free generation. | **follows**: F3 and formupdate include the stale value as a candidate and report the stale-pick rate. |
| Oh & Demberg 2026, *A retrieval-conditioned rebinding circuit for dynamic entity tracking* (arXiv 2606.08644). Checked | After one swap, a head circuit remaps the binding at retrieval instead of rebuilding state (Gemma, Llama, instruct). | mechanism-side precedent for the rebinding leg; the benchmark's chains go beyond one swap. |
| Ravulapalli 2026, *Legible failures* (arXiv 2609.11216). Checked | On wrong K50 trials, linear probes beat 1/K by +0.196 on average (14 of 16 models); near baseline on mutable and scoped bindings in Pythia-1.4B. | a near-chance log-prob reading does not show the binding is absent; relevant to every floored slice. |
| Kim & Schuster 2023, *Entity tracking in language models* (arXiv 2305.02363). Listed | The boxes task. | **follows**: riders is a boxes × operations task in four registers. |
| Feng & Steinhardt 2023, *How do language models bind entities in context?* (arXiv 2310.17191). Listed | Binding IDs, with a Pythia size sweep. | mechanism background. |
| Prakash et al. 2024, entity-tracking circuit under fine-tuning (arXiv 2402.14811); Dai 2024 (arXiv 2409.05448); Gur-Arieh 2025 (arXiv 2510.06182); Tang 2026 (arXiv 2605.30233); *Slot machines* 2026 (arXiv 2604.21139); Wu 2025 (arXiv 2505.20896). Listed | Entity-tracking mechanisms, ordering IDs, mixed retrieval strategies, parallel update aggregation, multi-entity slots, variable binding learned from scratch. | to read before any mechanism claim. |

## 4. State tracking and the algebra of updates

| paper | finding | relation |
|---|---|---|
| Sarkar, Mukherjee, Sato 2026, *Relational composition as parallel transport* (arXiv 2610.34945). Checked | Under per-entity frames, attention between fresh entities must go through a context-supplied transport; representation theory fixes the width and depth a mechanism needs for a given group. Shown on small finite groups and parsed CLUTRR / StepGame, not on pretrained LMs. | a vocabulary for why binding fresh names is hard, and a reason to split update scores into invertible updates (swaps) and resets (overwrites). Almost every update in the current batteries is a reset; riders `swap` is the main permutation. |
| Li, Guo, Andreas 2025, *(How) do language models track state?* (arXiv 2503.02854). Checked | S3 / S5 permutation tracking; associative vs parity-associative algorithms; pretraining shifts which forms. | template for a pure-swap holder-count family. |
| Zhang & Li 2026, *Tracking states or tracking cosets?* (arXiv 2609.29951). Checked | Transformers plateau at 1 / (abelianization class size) and track quotients, even on S3 / S4. | predicts partial scores on swap families; argues against a clean 4 → 5-holder step. |
| Liu et al. 2023 (arXiv 2210.10749); Merrill et al. 2024 (arXiv 2404.08819). Listed | Transformers learn shortcuts to automata; fixed-depth models cannot do S5 word problems. | background for any depth claim. |

## 5. Depth and looped models

The kotodama-tsumugi model on the panel is itself looped: a 20-layer core runs twice, for 72 effective layers.
A claim about depth therefore needs looped models whose loop count can be set.

| paper | finding (checked text) | relation |
|---|---|---|
| Zhu et al. 2025, *Ouro* (arXiv 2510.25741). Checked | 1.4B and 2.6B looped LMs trained at T = 4 with a loss at every step; SmolLM2 tokenizer; MMLU rises from T = 1 to T = 4; per-loop KV caches required. | the cleanest depth-vs-parameters test (matched effective depth across sizes). |
| Geiping et al. 2025, *Huginn* (arXiv 2502.05171). Checked | Recurrent-depth 3.5B; r = 1 is broken and r = 4 is at chance on MMLU. | a loop sweep needs r ≥ 8 and seed averaging. |
| McLeish et al., retrofitted recurrence (arXiv 2511.07384); Parcae (arXiv 2604.12946); BeNice! KV sharing across loops (arXiv 2610.31633). Checked | Retrofitted ~1B recurrent models with 1,024-token training context; a looped size ladder; KV-sharing compatibility per model. | candidate panel members for a depth axis. |

## Novelty candidates (◆), each to be searched before it is claimed
1. The rendering audit as an instrument. It combines the spread across renderings, off-construct levers and a causal test of the exemplars. Every flag is scoped to a panel and stored in a registry beside the pinned batteries.
2. K50's dependence on the binding lines' frame, and the disagreement with the published size law on Pythia.
3. Same-frame crystallized control for language-register fluid families, checked across six frames on eight base families.
4. The stages by which base models learn who *you* is, for static and dynamic person deixis, across a zoo of base models. Its prior art is not searched yet.
