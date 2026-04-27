# Manual Marker Review: CMES, MES, and PROG

Generated on 2026-04-27 from `foundation_validation/day_lineheldout/gse175634_day_lineheldout_marker_scores_by_state.csv`.

## Reason for review

`CMES`, `MES`, and `PROG` are transitional states in the cardiomyocyte differentiation trajectory. They are expected to be less cleanly separable than terminal or clearer populations such as `IPSC`, `CM`, and `CF`.

The simple marker-panel sanity check supports this concern:

- `IPSC` top marker set: `IPSC`, passes.
- `CM` top marker set: `CM`, passes.
- `CF` top marker set: `CF`, passes.
- `MES` top marker set: `IPSC`, does not pass the simple MES expectation.
- `CMES` top marker set: `CF`, does not pass the simple CMES/MES/PROG expectation.
- `PROG` top marker set: `CF`, does not pass the simple PROG/CMES/CM expectation.

## Class-specific notes

### MES

The MES-labeled cells retain high IPSC marker signal in the current panel. This may represent early mesodermal transition biology, but it could also reflect label bleed, asynchronous differentiation, or a marker panel that is too small and too sensitive to pluripotency markers.

Recommended review:

- Check canonical mesoderm markers individually: `T`, `MIXL1`, `MESP1`, `EOMES`, `TBX6`.
- Compare marker timing across `day1`, `day3`, and `day5`.
- Inspect whether MES errors are concentrated in specific lines or samples.

### CMES

CMES has mixed marker behavior and is not cleanly captured by the current small panel. The highest simple score is CF, which is suspicious and may indicate that the CMES definition overlaps with stromal/fibroblast-like programs or that broad extracellular matrix genes are dominating the marker score.

Recommended review:

- Expand CMES markers beyond `MESP1`, `NKX2-5`, `ISL1`, `HAND1`, and `TBX5`.
- Check whether COL/DCN/LUM/POSTN expression is biological contamination, real EMT-like signal, or annotation ambiguity.
- Review confusion pairs involving `CMES -> MES`, `CMES -> PROG`, and `CMES -> UNK`.

### PROG

PROG also scores highest on the CF panel. This is a warning that the current PROG label may include heterogeneous progenitors, fibroblast-like intermediates, or cells with strong matrix programs.

Recommended review:

- Inspect `ISL1`, `NKX2-5`, `TBX5`, `HAND2`, and `MEF2C` individually instead of only the mean panel score.
- Compare PROG against CF and CM in UMAP/embedding space.
- Report PROG performance separately from terminal-cell claims.

## Conclusion

The current model results are biologically meaningful for broad state prediction, but claims involving `MES`, `CMES`, and `PROG` should be framed as transitional-state predictions requiring marker-level review. For final reporting, include both the classifier metrics and this marker caveat.
