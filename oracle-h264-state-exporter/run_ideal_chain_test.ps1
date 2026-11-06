[CmdletBinding()]
param(
  [string]$SampleId = 'sample_0000',
  [string]$PipelineOutputRoot,
  [string]$ChainTestsRoot
)

$ErrorActionPreference = 'Stop'
$repoRoot = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($PipelineOutputRoot)) {
  $PipelineOutputRoot = Join-Path $repoRoot 'output'
}
if ([string]::IsNullOrWhiteSpace($ChainTestsRoot)) {
  $ChainTestsRoot = Join-Path $repoRoot 'chain_tests\ideal'
}
$pipelineRoot = [IO.Path]::GetFullPath($PipelineOutputRoot)
$datasetRoot = Join-Path $pipelineRoot 'training_dataset'
$manifestPath = Join-Path $datasetRoot 'manifests\all.jsonl'
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
  throw "Training manifest not found. Run RUN_ALL.bat first: $manifestPath"
}
$rows = @(
  Get-Content -LiteralPath $manifestPath |
  Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
  ForEach-Object { $_ | ConvertFrom-Json } |
  Where-Object { [string]$_.sample_id -eq $SampleId }
)
if ($rows.Count -eq 0) {
  throw "Sample was not found in the training manifest: $SampleId"
}
$targetStep = [int](($rows | ForEach-Object { [int]$_.target_step } | Measure-Object -Maximum).Maximum)
$targetRows = @($rows | Where-Object { [int]$_.input_step -eq $targetStep })
if ($targetRows.Count -ne 1) {
  throw "Expected exactly one final-step row for $SampleId, found $($targetRows.Count)."
}
$targetRow = $targetRows[0]
$targetRecordId = [string]$targetRow.target_record_id
if ([string]::IsNullOrWhiteSpace($targetRecordId)) {
  $targetRecordId = [string]$targetRow.record_id
}
$predictionPath = Join-Path $datasetRoot ([string]$targetRow.target_label_path)
$templatePath = Join-Path $pipelineRoot (Join-Path 'oracle_exports' (Join-Path $SampleId (Join-Path ('step_{0:D3}' -f $targetStep) 'oracle_state.bin')))
if (-not (Test-Path -LiteralPath $predictionPath -PathType Leaf)) {
  throw "Ideal target label not found: $predictionPath"
}
if (-not (Test-Path -LiteralPath $templatePath -PathType Leaf)) {
  throw "Final-step Oracle template not found: $templatePath"
}
$sampleOutputRoot = Join-Path ([IO.Path]::GetFullPath($ChainTestsRoot)) $SampleId
New-Item -ItemType Directory -Path $sampleOutputRoot -Force | Out-Null
$runOutput = Join-Path $sampleOutputRoot (Get-Date -Format 'yyyyMMdd-HHmmss-fff')
& (Join-Path $repoRoot 'encode_prediction.ps1') `
  -PredictionNpz $predictionPath `
  -OracleTemplateState $templatePath `
  -OutputDir $runOutput `
  -Fps ([double]$targetRow.fps) `
  -JmPackageRoot (Join-Path $repoRoot 'JM-backend')
$resultPath = Join-Path $runOutput 'prediction_replay_result.json'
$result = Get-Content -LiteralPath $resultPath -Raw | ConvertFrom-Json
if ($result.validation.standard_decode_valid -ne $true) {
  throw "Ideal chain test produced an invalid standard decode: $resultPath"
}
Write-Host ''
Write-Host 'Ideal chain test passed.'
Write-Host "  sample:     $SampleId"
Write-Host "  target step: $targetStep"
Write-Host "  prediction:  $predictionPath"
Write-Host "  output:      $runOutput"
