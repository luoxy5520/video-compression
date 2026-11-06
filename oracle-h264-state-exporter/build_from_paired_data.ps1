[CmdletBinding()]
param(
  [Parameter(Mandatory=$true)][string]$PairedDataJson,
  [Parameter(Mandatory=$true)][string]$OutputRoot,
  [ValidateRange(0,51)][int]$Qp = 28,
  [ValidateRange(1,16)][int]$ReferenceFrames = 1,
  [int[]]$Steps,
  [ValidateRange(1,1000000)][int]$MaxRecords = [int]::MaxValue,
  [ValidateSet('prompt','sample')][string]$GroupBy = 'prompt',
  [ValidateRange(0,1)][double]$TrainRatio = 0.8,
  [ValidateRange(0,1)][double]$ValRatio = 0.1,
  [ValidateRange(0,1)][double]$TestRatio = 0.1,
  [int]$SplitSeed = 20261105,
  [switch]$Resume,
  [switch]$CopyFeatures,
  [switch]$CopyBackendArtifacts,
  [switch]$SkipDataset,
  [string]$JmPackageRoot
)

$ErrorActionPreference = 'Stop'
$repoRoot = $PSScriptRoot

function Resolve-ManifestPath([string]$Value, [string]$BaseDir) {
  if ([string]::IsNullOrWhiteSpace($Value)) { return $null }
  if ([IO.Path]::IsPathRooted($Value)) { return [IO.Path]::GetFullPath($Value) }
  return [IO.Path]::GetFullPath((Join-Path $BaseDir $Value))
}

function Get-OptionalString($Value) {
  if ($null -eq $Value) { return $null }
  $text = [string]$Value
  if ([string]::IsNullOrWhiteSpace($text)) { return $null }
  return $text
}

function Test-CompleteExport([string]$Directory) {
  foreach ($name in @('metadata.json','closure_result.json','oracle_labels.npz','oracle_state.bin')) {
    if (-not (Test-Path -LiteralPath (Join-Path $Directory $name) -PathType Leaf)) { return $false }
  }
  try {
    $closure = Get-Content -LiteralPath (Join-Path $Directory 'closure_result.json') -Raw | ConvertFrom-Json
    return ($closure.bitstream_bit_exact -eq $true -and
      $closure.jm_decoded_pixels_exact -eq $true -and
      $closure.ffmpeg_decoded_pixels_exact -eq $true)
  } catch {
    return $false
  }
}

$manifestPath = (Resolve-Path -LiteralPath $PairedDataJson).Path
$manifestDir = Split-Path -Parent $manifestPath
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ($null -eq $manifest.records) {
  throw 'paired_data JSON must contain a records array.'
}

$records = @($manifest.records)
if ($null -ne $Steps -and $Steps.Count -gt 0) {
  $records = @($records | Where-Object { [int]$_.step -in $Steps })
}
$records = @($records | Select-Object -First $MaxRecords)
if ($records.Count -eq 0) {
  throw 'No records remain after applying Steps/MaxRecords filters.'
}

$outputPath = [IO.Path]::GetFullPath($OutputRoot)
if (-not (Test-Path -LiteralPath $outputPath)) {
  New-Item -ItemType Directory -Path $outputPath | Out-Null
}
$exportsRoot = Join-Path $outputPath 'oracle_exports'
New-Item -ItemType Directory -Path $exportsRoot -Force | Out-Null

$localConfigPath = Join-Path $repoRoot '.oracle-state.local.json'
if ([string]::IsNullOrWhiteSpace($JmPackageRoot) -and -not (Test-Path -LiteralPath $localConfigPath -PathType Leaf)) {
  throw 'JM is not configured. Run setup_local.ps1 once or pass -JmPackageRoot.'
}

$seen = @{}
$completed = 0
$skipped = 0
foreach ($record in $records) {
  $sampleId = Get-OptionalString $record.sample_id
  if ([string]::IsNullOrWhiteSpace($sampleId)) { throw 'Every record must have sample_id.' }
  $step = [int]$record.step
  $recordKey = "$sampleId::$step"
  if ($seen.ContainsKey($recordKey)) { throw "Duplicate (sample_id, step): $recordKey" }
  $seen[$recordKey] = $true

  if ([string]$record.pixel_format -ne 'yuv420p') {
    throw "Only yuv420p is supported; $recordKey declares '$($record.pixel_format)'."
  }
  $rawYuv = Resolve-ManifestPath ([string]$record.raw_yuv) $manifestDir
  $latent = Resolve-ManifestPath (Get-OptionalString $record.latent) $manifestDir
  $hidden = Resolve-ManifestPath (Get-OptionalString $record.hidden) $manifestDir
  $finalLatent = Resolve-ManifestPath (Get-OptionalString $record.final_latent) $manifestDir
  foreach ($requiredPath in @($rawYuv,$latent,$hidden)) {
    if ([string]::IsNullOrWhiteSpace($requiredPath) -or -not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
      throw "Missing required input for ${recordKey}: $requiredPath"
    }
  }

  $recordOutput = Join-Path $exportsRoot (Join-Path $sampleId ('step_{0:D3}' -f $step))
  if (Test-Path -LiteralPath $recordOutput) {
    if ($Resume -and (Test-CompleteExport $recordOutput)) {
      Write-Host "[resume] $recordKey is already complete."
      $skipped++
      continue
    }
    if (-not $Resume) {
      throw "Output already exists but is not resumable: $recordOutput"
    }
    $recordFull = [IO.Path]::GetFullPath($recordOutput)
    $exportsFull = [IO.Path]::GetFullPath($exportsRoot).TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
    if (-not $recordFull.StartsWith($exportsFull,[StringComparison]::OrdinalIgnoreCase)) {
      throw "Refusing to archive an export outside the exports root: $recordFull"
    }
    $recordParent = Split-Path -Parent $recordFull
    $archiveName = (Split-Path -Leaf $recordFull) + '_incomplete_' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff')
    $archiveDir = Join-Path $recordParent $archiveName
    Write-Host "[resume] Archiving incomplete Oracle export to: $archiveDir"
    Move-Item -LiteralPath $recordFull -Destination $archiveDir
  }

  Write-Host "[$($completed + $skipped + 1)/$($records.Count)] Exporting $recordKey"
  $exportArgs = @{
    InputYuv = $rawYuv
    Width = [int]$record.width
    Height = [int]$record.height
    Frames = [int]$record.frames
    Fps = [double]$record.fps
    ColorMatrix = if ([string]::IsNullOrWhiteSpace([string]$record.color_matrix)) { 'unknown' } else { [string]$record.color_matrix }
    ColorRange = if ([string]::IsNullOrWhiteSpace([string]$record.color_range)) { 'unknown' } else { [string]$record.color_range }
    Qp = $Qp
    ModePreset = 'skip-p16'
    ReferenceFrames = $ReferenceFrames
    RDOptimization = 0
    SampleId = $sampleId
    DiffusionStep = [string]$step
    Prompt = (Get-OptionalString $record.prompt)
    Seed = (Get-OptionalString $record.seed)
    LatentRef = $latent
    HiddenRef = $hidden
    FinalLatentRef = $finalLatent
    Timestep = (Get-OptionalString $record.timestep)
    Sigma = (Get-OptionalString $record.sigma)
    HiddenBranch = (Get-OptionalString $record.hidden_branch)
    HiddenBlock = (Get-OptionalString $record.hidden_block)
    Checkpoint = (Get-OptionalString $record.checkpoint)
    Sampler = (Get-OptionalString $record.sampler)
    TotalSteps = (Get-OptionalString $record.total_steps)
    OutputDir = $recordOutput
  }
  if (-not [string]::IsNullOrWhiteSpace($JmPackageRoot)) {
    $exportArgs.JmPackageRoot = $JmPackageRoot
  }
  & (Join-Path $repoRoot 'export_oracle.ps1') @exportArgs
  if (-not (Test-CompleteExport $recordOutput)) {
    throw "Oracle export did not pass closure validation: $recordKey"
  }
  $completed++
}

if (-not $SkipDataset) {
  $datasetDir = Join-Path $outputPath 'training_dataset'
  if (Test-Path -LiteralPath $datasetDir) {
    $summaryPath = Join-Path $datasetDir 'dataset_summary.json'
    $datasetIsCurrent = $false
    if ($Resume -and (Test-Path -LiteralPath $summaryPath -PathType Leaf)) {
      try {
        $summary = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
        $allManifest = Join-Path $datasetDir 'manifests\all.jsonl'
        $datasetIsCurrent = (
          [string]$summary.schema -eq 'oracle-h264-state-exporter/dataset-v2-final-target' -and
          [string]$summary.configuration.target_step -eq 'final' -and
          [int]$summary.counts.accepted_records -eq $records.Count -and
          (Test-Path -LiteralPath $allManifest -PathType Leaf)
        )
      } catch {
        $datasetIsCurrent = $false
      }
    }
    if ($datasetIsCurrent) {
      Write-Host '[resume] Training dataset is already complete and uses final-step targets.'
      Write-Host "Completed exports: $completed; resumed exports: $skipped; dataset: $datasetDir"
      return
    }
    if (-not $Resume) {
      throw "Training dataset directory already exists: $datasetDir"
    }
    $datasetFull = [IO.Path]::GetFullPath($datasetDir)
    $outputFull = [IO.Path]::GetFullPath($outputPath).TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
    if (-not $datasetFull.StartsWith($outputFull,[StringComparison]::OrdinalIgnoreCase)) {
      throw "Refusing to archive a dataset outside the output root: $datasetFull"
    }
    $archiveName = 'training_dataset_stale_' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff')
    $archiveDir = Join-Path $outputPath $archiveName
    Write-Host "[resume] Archiving incomplete or old-schema dataset to: $archiveDir"
    Move-Item -LiteralPath $datasetFull -Destination $archiveDir
  }
  $datasetArgs = @{
    ExportsRoot = $exportsRoot
    OutputDir = $datasetDir
    FeatureRoot = $manifestDir
    FeatureKind = 'both'
    GroupBy = $GroupBy
    TrainRatio = $TrainRatio
    ValRatio = $ValRatio
    TestRatio = $TestRatio
    SplitSeed = $SplitSeed
    ExpectedModePreset = 'skip-p16'
    RequireFeatures = $true
    CopyFeatures = $CopyFeatures
    CopyBackendArtifacts = $CopyBackendArtifacts
  }
  & (Join-Path $repoRoot 'build_dataset.ps1') @datasetArgs
}

Write-Host ''
Write-Host 'Paired-data bridge completed.'
Write-Host "  records selected: $($records.Count)"
Write-Host "  exports created: $completed"
Write-Host "  exports resumed: $skipped"
Write-Host "  output root: $outputPath"
