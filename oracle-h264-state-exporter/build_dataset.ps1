[CmdletBinding()]
param(
  [Parameter(Mandatory=$true)][string]$ExportsRoot,
  [Parameter(Mandatory=$true)][string]$OutputDir,
  [string]$FeatureRoot,
  [ValidateSet('prompt','sample')][string]$GroupBy = 'prompt',
  [ValidateRange(0,1)][double]$TrainRatio = 0.8,
  [ValidateRange(0,1)][double]$ValRatio = 0.1,
  [ValidateRange(0,1)][double]$TestRatio = 0.1,
  [int]$SplitSeed = 20261105,
  [string]$StepFilter,
  [string]$TargetStep = 'final',
  [ValidateSet('latent','hidden','both','legacy')][string]$FeatureKind = 'both',
  [ValidateSet('skip-p16','p16','variable','any')][string]$ExpectedModePreset = 'skip-p16',
  [switch]$RequireFeatures,
  [switch]$CopyFeatures,
  [switch]$CopyBackendArtifacts
)

$ErrorActionPreference = 'Stop'
$python = Get-Command python -ErrorAction SilentlyContinue
if ($null -eq $python) {
  throw 'python was not found in PATH.'
}

$repoRoot = $PSScriptRoot
$builder = Join-Path $repoRoot 'scripts\build_dataset.py'
$builderArgs = @(
  $builder,
  '--exports-root', $ExportsRoot,
  '--output-dir', $OutputDir,
  '--group-by', $GroupBy,
  '--train-ratio', ([string]::Format([Globalization.CultureInfo]::InvariantCulture, '{0}', $TrainRatio)),
  '--val-ratio', ([string]::Format([Globalization.CultureInfo]::InvariantCulture, '{0}', $ValRatio)),
  '--test-ratio', ([string]::Format([Globalization.CultureInfo]::InvariantCulture, '{0}', $TestRatio)),
  '--split-seed', "$SplitSeed",
  '--expected-mode-preset', $ExpectedModePreset,
  '--feature-kind', $FeatureKind,
  '--target-step', $TargetStep
)

if (-not [string]::IsNullOrWhiteSpace($FeatureRoot)) {
  $builderArgs += @('--feature-root', $FeatureRoot)
}
if (-not [string]::IsNullOrWhiteSpace($StepFilter)) {
  $builderArgs += @('--step-filter', $StepFilter)
}
if ($RequireFeatures) {
  $builderArgs += '--require-features'
}
if ($CopyFeatures) {
  $builderArgs += '--copy-features'
}
if ($CopyBackendArtifacts) {
  $builderArgs += '--copy-backend-artifacts'
}

& $python.Source @builderArgs
if ($LASTEXITCODE -ne 0) {
  throw "Dataset construction failed with exit code $LASTEXITCODE."
}
