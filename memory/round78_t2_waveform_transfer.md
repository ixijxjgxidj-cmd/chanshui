# Round 78: STEAD multi-task waveform transfer and blend audit

## Scope and isolation

- Remote execution only: zzai `/root/projects/t2_wave_blend_20260816/`.
- Public STEAD cache only for pretraining: 68,872 P-window waveforms with magnitude, distance and depth metadata, grouped by `source_id`.
- Downstream assets are R1/R2 train-side `outputs/t2_round44d/raw_train.npz`, each with 140 three-component 6,000-sample records.
- No 08 archive, file, derived feature, prediction or annotation was opened. Round 58 holdout is a frozen comparator only, never used for selection.

## Existing independent evidence

`outputs/t2_round52b_ft/ft52b.json` uses 5-fold OOF and three seeds. Relative to scratch (FTC average 119.49), STEAD single-task pretraining reached 135.87 and multi-task magnitude/distance/depth pretraining reached 136.84 (SE 0.83). This confirms train-side representation transfer but is below the frozen Round 58 holdout score 149.29, so it cannot replace Round 58.

## Round 78 blend audit

A fixed alpha grid from 0.00 to 1.00 in increments of 0.05 blended historical seven-member Round 56 OOF center predictions and multi-task waveform OOF predictions. Scores were recomputed against labels in `raw_train.npz`. An initial accidental comparison against label-like arrays in `round56_oof.npz` was detected and discarded before any decision.

- Best in-sample alpha: 0.25.
- Alpha 0.00: R1/R2 average score 134.46.
- Alpha 0.25: R1/R2 average score 134.82.
- Gain: +0.37 only, with alpha selected on the same OOF.

## Verdict

**Not admitted.** The selected blend weight is in-sample and has no independent validation. It is not used for a release, no holdout is reopened, and the Round 58 configuration remains frozen. The narrow research result is that waveform transfer shows weak complementary R2-side signal; a nested OOF protocol is required before reconsidering blending.

## Literature review: 15 directly relevant works

The corpus is `/root/5.6+chanshui1/outputs/t2_round52_lit/lit52_ranked.json`, generated from arXiv metadata for station magnitude, P-wave early warning, transfer, multi-task learning, JEPA and regression distillation.

1. Mousavi and Beroza (2020), *A Machine Learning Approach for Earthquake Magnitude Estimation*: absolute amplitude and waveform context matter.
2. Munchmeyer et al. (2021), arXiv:2101.02010: regional transfer reduces systematic magnitude bias but domain mismatch remains central.
3. SeisT (2023), arXiv:2310.01037: joint seismic tasks improve shared representations.
4. FisH (2024), arXiv:2408.06629: phase, location and magnitude share useful waveform features.
5. PhaseNet, Zhu and Beroza (2019), arXiv:1803.03211: convolutional phase representations transfer across tasks.
6. EQTransformer, Mousavi et al. (2020), arXiv:1909.06396: shared encoders work for detection and phase tasks.
7. Earthquake Transformer, Mousavi et al. (2020), arXiv:2004.00586: sequence context helps early waveform inference.
8. STEAD, Mousavi et al. (2019), arXiv:1810.10669: source metadata permits group-separated validation.
9. INSTANCE, Michelini et al. (2021), arXiv:2101.06465: large heterogeneous corpora support transfer testing.
10. SeisBench, Woollam et al. (2022), arXiv:2210.11114: reproducible seismic dataset separation.
11. BYOL, Grill et al. (2020), arXiv:2006.07733: label-free pretraining reference, deferred because STEAD labels exist.
12. JEPA, Assran et al. (2023), arXiv:2301.08243: predictive representations are promising if target-like unlabeled waveforms are found.
13. Hinton et al. (2015), arXiv:1503.02531: knowledge distillation does not justify in-sample blend selection.
14. Lakshminarayanan et al. (2017), arXiv:1612.01474: deep ensembles motivate retaining diversity diagnostics.
15. CORAL, Sun and Saenko (2016), arXiv:1607.01719: domain alignment is relevant but prior diagonal-CORAL showed no T3 gain.

## Next experiment

Only reopen this family with fully nested evaluation: in every outer R1/R2 fold, select blend alpha inside the remaining folds and evaluate the untouched outer fold. Round 58 holdout and all 08 assets remain excluded from fitting and selection.
