@echo off
cd /d C:\Users\tumok\Documents\Projects\Final\HybridGnnRnn
echo Started full repeated CV at %DATE% %TIME% > foundation_validation\repeated_line_cv_plan\full_repeated_cv_stdout.log
C:\Users\tumok\anaconda3\python.exe scripts\foundation\run_repeated_geneformer_cv.py ^
  --tokenized-dataset foundation_model_data\geneformer\tokenized\gse175634_geneformer.dataset ^
  --model-dir models\geneformer\Geneformer\Geneformer-V1-10M ^
  --output-dir foundation_validation\repeated_line_cv_plan ^
  --targets diffday cell_state ^
  --repeats 3 ^
  --heldout-lines 3 ^
  --epochs 1 ^
  --max-cells-per-class 15000 ^
  --nproc 0 ^
  --ngpu 1 ^
  --run ^
  --skip-existing ^
  >> foundation_validation\repeated_line_cv_plan\full_repeated_cv_stdout.log 2>> foundation_validation\repeated_line_cv_plan\full_repeated_cv_stderr.log
echo Python exit code %ERRORLEVEL% >> foundation_validation\repeated_line_cv_plan\full_repeated_cv_stdout.log
echo Finished full repeated CV at %DATE% %TIME% >> foundation_validation\repeated_line_cv_plan\full_repeated_cv_stdout.log
