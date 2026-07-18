param(
    [string]$Corpus = "datasets\teacher\corpus.tsv",
    [string]$TeacherRoot = "datasets\teacher",
    [string]$Output = "artifacts\ensemble-v2-large\ensemble-v2-large.pkl"
)

$ErrorActionPreference = "Stop"
$Project = Split-Path -Parent $PSScriptRoot
$Runner = Join-Path $Project ".venv\Scripts\tmgm-rt.exe"

Push-Location $Project
try {
    & $Runner train-ensemble `
        --corpus $Corpus `
        --teacher-root $TeacherRoot `
        --train-tracks 72 `
        --validation-tracks 18 `
        --frames-per-track 600 `
        --epochs 6 `
        --activity-members 3 `
        --onset-members 3 `
        --activity-clauses 192 `
        --onset-clauses 192 `
        --activity-threshold 96 `
        --onset-threshold 96 `
        --activity-specificity 5 `
        --onset-specificity 4 `
        --activity-negative-samples 8 `
        --onset-negative-samples 4 `
        --onset-fusion top2_mean `
        --onset-tolerance-frames 4 `
        --max-literals 40 `
        --seed 42 `
        --output $Output
}
finally {
    Pop-Location
}
