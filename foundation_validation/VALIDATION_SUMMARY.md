# Foundation Model Validation Summary



## Implemented validation

- Repeated held-out-line CV planning for full Geneformer fine-tuning:
  - `foundation_validation/repeated_line_cv_plan/fold_plan.json`
  - `foundation_validation/repeated_line_cv_plan/run_repeated_geneformer_cv.ps1`
  - 36 planned runs: 3 repeats across held-out-line folds for both `diffday` and `cell_state`.
- Per-class classification reports, confusion matrices, top confusions, and grouped metrics:
  - `foundation_validation/day_lineheldout/`
  - `foundation_validation/state_lineheldout/`
- Marker-gene sanity checks:
  - GSE175634 internal marker checks in `foundation_validation/day_lineheldout/`
  - External GSE130731 high-count iPS marker checks in `foundation_validation/external_gse130731_topcells/`
- External dataset testing:
  - Built and tokenized high-count GSE130731 iPS cells.
  - Ran the fine-tuned state and day classifier heads as prediction-distribution probes.
- Fast repeated held-out-line CV over exported embeddings:
  - `foundation_validation/embedding_group_cv/`

## Current headline results

### Internal held-out-line validation

- Day classifier: accuracy `0.7404`, macro F1 `0.7529`.
- State classifier: accuracy `0.8915`, macro F1 `0.8637`.
- Day errors are mainly adjacent or late-stage confusions, especially `day11 -> day15`, `day15 -> day11`, and `day7 -> day11`.
- State errors are mostly biologically neighboring populations, especially `CF -> PROG`, `IPSC -> MES`, `MES -> IPSC`, and `PROG -> CF`.

### Marker sanity

- GSE175634 marker coverage is strong for the selected marker panels.
- Clear expected checks passed for `IPSC`, `CM`, and `CF`.
- `CMES`, `MES`, and `PROG` need biological review because their simple marker-panel maxima do not always match their assigned labels.
- GSE130731 high-count external cells pass the iPSC marker sanity check.

### External GSE130731 probe

GSE130731 is useful as an external iPSC sanity/projection test, not as a full cardiac trajectory benchmark.

- State model on external high-count iPS cells:
  - `IPSC` predicted for only `3.7%`.
  - Most calls were `MES` (`47.5%`) and `CMES` (`17.3%`).
- Day model on external high-count iPS cells:
  - Mostly `day3` (`49.6%`), `day7` (`31.2%`), and `day5` (`13.0%`).
  - `day0` only `5.45%`.

This means the current model is internally promising but not externally robust enough to claim broad generalization.

## Calibration and rejection

Temperature calibration was fitted from the original held-out-line predictions:

- Day model:
  - Temperature: `1.02796`
  - Confidence threshold: `0.71030`
  - Accepted coverage: `56.20%`
  - Accepted accuracy: `90.06%`
- State model:
  - Temperature: `1.08425`
  - Confidence threshold: `0.49136`
  - Accepted coverage: `98.20%`
  - Accepted accuracy: `90.04%`

On external GSE130731 high-count iPS cells, calibrated rejection flags many day-model calls as low-confidence/OOD, but the original state model still confidently misclassifies many iPS cells as MES/CMES. This motivated explicit external iPSC anchoring.

## External iPSC anchor fine-tuning

Built an anchor-augmented Geneformer dataset:

- Internal GSE175634 cells: `230,786`
- External QC-passed GSE130731 iPS anchors: `4,000`
- Combined cells: `234,786`
- External anchor labels: `cell_state=IPSC`, `diffday=day0`

Anchored state model:

- Internal held-out-line accuracy: `0.8939`
- Internal held-out-line macro F1: `0.8692`
- External GSE130731 iPS prediction: `100% IPSC`

Anchored day model:

- Internal held-out-line accuracy: `0.7388`
- Internal held-out-line macro F1: `0.7508`
- External GSE130731 iPS prediction: `99.975% day0`

This is a useful improvement: external iPSC behavior is fixed without materially degrading held-out-line internal performance. It is still not proof of broad external trajectory generalization, because GSE130731 only anchors the early/iPSC domain.

## Domain/Batch Check

The domain-adversarial embedding check is implemented in `scripts/foundation/domain_adversarial_embedding_check.py`.

Current state-embedding probe result:

- Random-domain test target accuracy: `0.1876`
- Random-domain test domain accuracy: `0.0917`
- Held-out-line probe target accuracy: `0.1517`
- Held-out-line probe domain accuracy: `0.0858`

These low probe scores indicate that the exported embeddings alone are weak for these shallow/domain-adversarial checks. The classifier head is much stronger than the frozen exported embeddings, so final claims should rely on full fine-tuning CV rather than embedding-only probes.

## Full CV Runs

Started the full repeated held-out-line fine-tuning run from:

- `foundation_validation/repeated_line_cv_plan/run_full_repeated_cv.cmd`

Logs:

- `foundation_validation/repeated_line_cv_plan/full_repeated_cv_stdout.log`
- `foundation_validation/repeated_line_cv_plan/full_repeated_cv_stderr.log`

This run is resumable because `run_repeated_geneformer_cv.py` now supports `--skip-existing`.

## Recommended strengthening

- Run the generated full repeated held-out-line fine-tuning plan, not only the fast embedding CV.
- Add probability calibration and confidence rejection so external/OOD samples can be flagged instead of forced into a trajectory class.
- Fine-tune with external iPSC data as an explicit `IPSC`/early-domain anchor after careful QC.
- Add a batch/domain-adversarial validation check to separate real biological trajectory signal from dataset/source effects.
- Use leave-one-line-out or leave-one-donor-out CV for the final reported table.
- Add manual review for `CMES`, `MES`, and `PROG` marker definitions because these are the biologically ambiguous transition states.
