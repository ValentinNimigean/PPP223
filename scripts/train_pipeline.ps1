param (
    [string]$Repo = ".",
    [string]$TeacherModel = "gpt-4o",
    [string]$StudentUrl = "",
    [string]$OllamaBaseModel = $env:OLLAMA_MODEL,
    [string]$SftBaseModel = "unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit",
    [switch]$SkipSynth,
    [switch]$SkipSft,
    [switch]$SkipDpo,
    [switch]$SkipEval
)

$ErrorActionPreference = "Stop"

if (-not $OllamaBaseModel) {
    $OllamaBaseModel = "qwen2.5-coder:1.5b"
}

# Activate virtual environment if present
if (Test-Path ".venv") {
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "Activating virtual environment (.venv)..." -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    . .venv/Scripts/Activate.ps1
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "PPP223 / Mobtrap Training Pipeline (Windows PowerShell)" -ForegroundColor Cyan
Write-Host "Repo:              $Repo" -ForegroundColor Cyan
Write-Host "Teacher Model:     $TeacherModel" -ForegroundColor Cyan
Write-Host "Ollama Eval Model: $OllamaBaseModel" -ForegroundColor Cyan
Write-Host "SFT Base Model:    $SftBaseModel" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Helper function to combine files if they exist
function Combine-Files {
    param (
        [string]$OutputFile,
        [string[]]$InputFiles
    )
    if (Test-Path $OutputFile) {
        Remove-Item $OutputFile -Force
    }
    $found = $false
    foreach ($file in $InputFiles) {
        if (Test-Path $file) {
            Get-Content $file | Add-Content $OutputFile
            $found = $true
        } else {
            Write-Host "Skipping missing file: $file" -ForegroundColor Yellow
        }
    }
    if (-not $found) {
        Write-Error "No input files found for $OutputFile"
    } else {
        $lines = (Get-Content $OutputFile).Count
        Write-Host "Wrote $OutputFile with $lines rows" -ForegroundColor Green
    }
}

# Ensure target directories exist
if (-not (Test-Path "training_data/sft")) {
    New-Item -ItemType Directory -Path "training_data/sft" | Out-Null
}
if (-not (Test-Path "training_data/preferences")) {
    New-Item -ItemType Directory -Path "training_data/preferences" | Out-Null
}
if (-not (Test-Path "evaluation_reports")) {
    New-Item -ItemType Directory -Path "evaluation_reports" | Out-Null
}

# STEP 0: Base Eval Before Training
if (-not $SkipEval) {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "STEP 0: Base eval before training" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    python eval/eval.py --repo $Repo --model $OllamaBaseModel --benchmark eval/benchmark_self.json --out evaluation_reports/eval_report_base.json --verbose
}

# STEP 1: Generate Synthetic SFT Data
if (-not $SkipSynth) {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "STEP 1: Generate synthetic SFT data" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    python data/synth.py --repo $Repo --out training_data/sft/synthetic_qa_auto.jsonl --model $TeacherModel
}

# STEP 2: Combine SFT Data
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "STEP 2: Combine SFT data" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Combine-Files -OutputFile "training_data/sft/synthetic_qa_combined.jsonl" -InputFiles @("training_data/sft/synthetic_qa_seed.jsonl", "training_data/sft/synthetic_qa_auto.jsonl")

python scripts/audit_training_data.py training_data/sft/synthetic_qa_combined.jsonl

# STEP 3 & 4: SFT Fine-Tuning
if (-not $SkipSft) {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "STEP 3: Dry-run SFT formatting" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    python model/finetune.py --data training_data/sft/synthetic_qa_combined.jsonl --model $SftBaseModel --out results_sft --dry-run --max-seq-length 2048

    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "STEP 4: SFT fine-tuning" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    
    $env:PYTORCH_ALLOC_CONF = "expandable_segments:True"
    python model/finetune.py --data training_data/sft/synthetic_qa_combined.jsonl --model $SftBaseModel --out results_sft --epochs 3 --learning-rate 1e-4 --lora-r 8 --lora-alpha 16 --eval-split 0.1
}

# STEP 5: Eval post-codefixes
if (-not $SkipEval) {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "STEP 5: Eval current Ollama model after code fixes" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "NOTE: Evaluating base local model. To evaluate the SFT model, compile and load the adapter first." -ForegroundColor DarkYellow
    python eval/eval.py --repo $Repo --model $OllamaBaseModel --benchmark eval/benchmark_self.json --out evaluation_reports/eval_report_post_codefix_or_base.json --verbose
}

# STEP 6-9: DPO Preference Alignment
if (-not $SkipDpo) {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "STEP 6: Generate preference data" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    
    if ($StudentUrl) {
        python data/pref_gen.py --input training_data/sft/synthetic_qa_combined.jsonl --output training_data/preferences/preference_data_auto.jsonl --model $TeacherModel --judge $TeacherModel --student-url $StudentUrl --student-model $OllamaBaseModel
    } else {
        python data/pref_gen.py --input training_data/sft/synthetic_qa_combined.jsonl --output training_data/preferences/preference_data_auto.jsonl --model $TeacherModel --judge $TeacherModel
    }

    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "STEP 7: Generate failure-driven DPO rows from eval reports" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    
    if (Test-Path "evaluation_reports/eval_report_base.json") {
        python data/failure_dpo_from_eval.py --report evaluation_reports/eval_report_base.json --out training_data/preferences/preference_data_failures_base.jsonl --threshold 0.75
    }
    if (Test-Path "evaluation_reports/eval_report_post_codefix_or_base.json") {
        python data/failure_dpo_from_eval.py --report evaluation_reports/eval_report_post_codefix_or_base.json --out training_data/preferences/preference_data_failures_post.jsonl --threshold 0.75
    }

    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "STEP 8: Combine preference data" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Combine-Files -OutputFile "training_data/preferences/preference_data_combined.jsonl" -InputFiles @("training_data/preferences/preference_data_auto.jsonl", "training_data/preferences/preference_data_failures_base.jsonl", "training_data/preferences/preference_data_failures_post.jsonl", "training_data/preferences/preference_data_failures_unseen_v2.jsonl", "training_data/preferences/preference_data_rlhf.jsonl")

    python scripts/audit_training_data.py training_data/preferences/preference_data_combined.jsonl

    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "STEP 8b: Validate combined DPO data" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    python scripts/validate_dpo.py training_data/preferences/preference_data_combined.jsonl

    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "STEP 9: DPO training" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    
    if (-not (Test-Path "results_sft/adapter")) {
        Write-Error "Missing results_sft/adapter. SFT adapter must exist before running DPO."
    }
    
    python model/dpo.py --dpo-data-path training_data/preferences/preference_data_combined.jsonl --out results_dpo --sft-adapter results_sft/adapter --max-steps 200
}

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "DONE: Fine-tuning pipeline finished successfully." -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
