[CmdletBinding()]
param(
  [string]$SampleId = 'sample_0000',
  [string]$PipelineOutputRoot,
  [string]$TestOutputRoot,
  [string]$JmPackageRoot
)

$ErrorActionPreference = 'Stop'
$repoRoot = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($PipelineOutputRoot)) {
  $PipelineOutputRoot = Join-Path $repoRoot 'output'
}
if ([string]::IsNullOrWhiteSpace($TestOutputRoot)) {
  $TestOutputRoot = Join-Path $repoRoot 'chain_tests\ideal'
}
if ([string]::IsNullOrWhiteSpace($JmPackageRoot)) {
  $bundledJm = Join-Path $repoRoot 'JM-backend'
  if (Test-Path -LiteralPath $bundledJm -PathType Container) {
    $JmPackageRoot = $bundledJm
  }
}

$pipelineRoot = [IO.Path]::GetFullPath($PipelineOutputRoot)
$labelPath = Join-Path $pipelineRoot ("training_dataset\labels\{0}__step-20.npz" -f $SampleId)
$templatePath = Join-Path $pipelineRoot ("oracle_exports\{0}\step_020\oracle_state.bin" -f $SampleId)
$metadataPath = Join-Path $pipelineRoot ("oracle_exports\{0}\step_020\metadata.json" -f $SampleId)
$referenceBitstream = Join-Path $pipelineRoot ("oracle_exports\{0}\step_020\reference.264" -f $SampleId)
foreach ($required in @($labelPath,$templatePath,$metadataPath,$referenceBitstream)) {
  if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
    throw "Required step-20 artifact is missing: $required. Run RUN_ALL.bat first."
  }
}

$metadata = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
$runDir = Join-Path ([IO.Path]::GetFullPath($TestOutputRoot)) (Join-Path $SampleId $stamp)
$encodeArgs = @{
  PredictionNpz = $labelPath
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
$predictedStateHash = (Get-FileHash -LiteralPath (Join-Path $runDir 'predicted_state.bin') -Algorithm SHA256).Hash.ToLowerInvariant()
$templateStateHash = (Get-FileHash -LiteralPath $templatePath -Algorithm SHA256).Hash.ToLowerInvariant()
$predictedBitstreamHash = (Get-FileHash -LiteralPath (Join-Path $runDir 'predicted.264') -Algorithm SHA256).Hash.ToLowerInvariant()
$referenceBitstreamHash = (Get-FileHash -LiteralPath $referenceBitstream -Algorithm SHA256).Hash.ToLowerInvariant()
$summary = [ordered]@{
  schema = 'oracle-h264-state-exporter/ideal-chain-test-v1'
  sample_id = $SampleId
  target_step = 20
  interpretation = 'step-20 Oracle P labels used as a perfect model prediction; only the step-20 Oracle I state is retained from the template'
  run_dir = $runDir
  state_matches_oracle = ($predictedStateHash -eq $templateStateHash)
  bitstream_matches_oracle = ($predictedBitstreamHash -eq $referenceBitstreamHash)
  jm_matches_ffmpeg = $result.validation.jm_matches_ffmpeg
  standard_decode_valid = $result.validation.standard_decode_valid
}
[IO.File]::WriteAllText(
  (Join-Path $runDir 'ideal_chain_summary.json'),
  ($summary | ConvertTo-Json -Depth 5),
  [Text.UTF8Encoding]::new($false)
)
if (-not $summary.state_matches_oracle -or -not $summary.bitstream_matches_oracle -or -not $summary.standard_decode_valid) {
  throw "Ideal chain test failed. Inspect $runDir"
}
Write-Host ''
Write-Host 'Ideal prediction chain test passed.'
Write-Host "  sample: $SampleId"
Write-Host '  target/template step: 20'
Write-Host "  result: $(Join-Path $runDir 'ideal_chain_summary.json')"
