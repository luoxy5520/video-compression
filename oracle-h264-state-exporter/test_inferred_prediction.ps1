[CmdletBinding()]
param(
  [Parameter(Mandatory=$true)][string]$PredictionNpz,
  [string]$SampleId = 'sample_0000',
  [string]$PipelineOutputRoot,
  [string]$TestOutputRoot,
  [string]$JmPackageRoot
)

$ErrorActionPreference = 'Stop'
$repoRoot = $PSScriptRoot
$predictionPath = (Resolve-Path -LiteralPath $PredictionNpz).Path
if ([string]::IsNullOrWhiteSpace($PipelineOutputRoot)) {
  $PipelineOutputRoot = Join-Path $repoRoot 'output'
}
if ([string]::IsNullOrWhiteSpace($TestOutputRoot)) {
  $TestOutputRoot = Join-Path $repoRoot 'chain_tests\inferred'
}
if ([string]::IsNullOrWhiteSpace($JmPackageRoot)) {
  $bundledJm = Join-Path $repoRoot 'JM-backend'
  if (Test-Path -LiteralPath $bundledJm -PathType Container) {
    $JmPackageRoot = $bundledJm
  }
}

$pipelineRoot = [IO.Path]::GetFullPath($PipelineOutputRoot)
$templatePath = Join-Path $pipelineRoot ("oracle_exports\{0}\step_020\oracle_state.bin" -f $SampleId)
$metadataPath = Join-Path $pipelineRoot ("oracle_exports\{0}\step_020\metadata.json" -f $SampleId)
foreach ($required in @($templatePath,$metadataPath)) {
  if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
    throw "Required step-20 Oracle I template is missing: $required. Run RUN_ALL.bat first."
  }
}

$metadata = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
$runDir = Join-Path ([IO.Path]::GetFullPath($TestOutputRoot)) (Join-Path $SampleId $stamp)
$encodeArgs = @{
  PredictionNpz = $predictionPath
  OracleTemplateState = $templatePath
  OutputDir = $runDir
  Fps = [double]$metadata.input.fps
}
if (-not [string]::IsNullOrWhiteSpace($JmPackageRoot)) {
  $encodeArgs.JmPackageRoot = $JmPackageRoot
}
& (Join-Path $repoRoot 'encode_prediction.ps1') @encodeArgs

$resultPath = Join-Path $runDir 'prediction_replay_result.json'
$result = Get-Content -LiteralPath $resultPath -Raw | ConvertFrom-Json
$summary = [ordered]@{
  schema = 'oracle-h264-state-exporter/inferred-chain-test-v1'
  sample_id = $SampleId
  target_step = 20
  prediction = $predictionPath
  oracle_dependency = 'step-20 I-frame state only; every P-frame mode/MV/residual is replaced by the supplied prediction'
  run_dir = $runDir
  jm_matches_ffmpeg = $result.validation.jm_matches_ffmpeg
  standard_decode_valid = $result.validation.standard_decode_valid
  predicted_bitstream = $result.outputs.predicted_bitstream
  predicted_bitstream_sha256 = $result.outputs.predicted_bitstream_sha256
}
[IO.File]::WriteAllText(
  (Join-Path $runDir 'inferred_chain_summary.json'),
  ($summary | ConvertTo-Json -Depth 5),
  [Text.UTF8Encoding]::new($false)
)
if (-not $summary.standard_decode_valid) {
  throw "Inferred prediction chain test failed. Inspect $runDir"
}
Write-Host ''
Write-Host 'Inferred prediction chain test passed.'
Write-Host "  sample: $SampleId"
Write-Host '  Oracle dependency: step-20 I-frame state only'
Write-Host "  bitstream: $($summary.predicted_bitstream)"
Write-Host "  result: $(Join-Path $runDir 'inferred_chain_summary.json')"
