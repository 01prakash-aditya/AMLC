# PowerShell script to run the full ML pipeline sequentially

Write-Host "=========================================="
Write-Host "STARTING STAGE 2: PREPROCESSING"
Write-Host "=========================================="
python code\business_entity_resolution\src\stage02_preprocessing.py
if ($LASTEXITCODE -ne 0) { Write-Error "Stage 2 failed!"; exit 1 }

Write-Host "`n=========================================="
Write-Host "STARTING STAGE 3: BLOCKING"
Write-Host "=========================================="
python code\business_entity_resolution\src\stage03_blocking.py
if ($LASTEXITCODE -ne 0) { Write-Error "Stage 3 failed!"; exit 1 }

Write-Host "`n=========================================="
Write-Host "STARTING STAGE 4: FEATURE ENGINEERING"
Write-Host "=========================================="
python code\business_entity_resolution\src\stage04_feature_engineering.py
if ($LASTEXITCODE -ne 0) { Write-Error "Stage 4 failed!"; exit 1 }

Write-Host "`n=========================================="
Write-Host "STARTING STAGE 5: MODEL TRAINING"
Write-Host "=========================================="
python code\business_entity_resolution\src\stage05_model_training.py
if ($LASTEXITCODE -ne 0) { Write-Error "Stage 5 failed!"; exit 1 }

Write-Host "`n=========================================="
Write-Host "STARTING STAGE 6: INFERENCE"
Write-Host "=========================================="
python code\business_entity_resolution\src\stage06_inference.py
if ($LASTEXITCODE -ne 0) { Write-Error "Stage 6 failed!"; exit 1 }

Write-Host "`n=========================================="
Write-Host "PIPELINE COMPLETED SUCCESSFULLY!"
Write-Host "Outputs generated in output/ directory."
Write-Host "=========================================="
