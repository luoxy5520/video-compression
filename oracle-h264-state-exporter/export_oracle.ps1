[CmdletBinding(DefaultParameterSetName='Rgb')]
param(
  [Parameter(Mandatory=$true,ParameterSetName='Rgb')][string]$InputRgb,
  [Parameter(Mandatory=$true,ParameterSetName='Yuv')][string]$InputYuv,
  [Parameter(Mandatory=$true)][ValidateRange(16,16384)][int]$Width,
  [Parameter(Mandatory=$true)][ValidateRange(16,16384)][int]$Height,
  [Parameter(Mandatory=$true)][ValidateRange(2,1000000)][int]$Frames,
  [ValidateRange(1,240)][double]$Fps = 30,
  [Parameter(ParameterSetName='Rgb')][ValidateSet('rgb24','bgr24','rgba','bgra')][string]$PixelFormat = 'rgb24',
  [ValidateSet('bt601','bt709','unknown')][string]$ColorMatrix = 'unknown',
  [ValidateSet('limited','full','unknown')][string]$ColorRange = 'unknown',
  [ValidateRange(0,51)][int]$Qp = 28,
  [ValidateSet('skip-p16','p16','variable')][string]$ModePreset = 'skip-p16',
  [ValidateRange(1,16)][int]$ReferenceFrames = 2,
  [ValidateSet(0,1)][int]$RDOptimization = 0,
  [string]$SampleId,
  [string]$DiffusionStep,
  [string]$Prompt,
  [string]$Seed,
  [string]$FeatureRef,
  [string]$LatentRef,
  [string]$HiddenRef,
  [string]$FinalLatentRef,
  [string]$Timestep,
  [string]$Sigma,
  [string]$HiddenBranch,
  [string]$HiddenBlock,
  [string]$Checkpoint,
  [string]$Sampler,
  [string]$TotalSteps,
  [string]$OutputDir,
  [string]$JmPackageRoot
)

$ErrorActionPreference = 'Stop'
$repoRoot = $PSScriptRoot

function Get-Tool([string]$Name) {
  $tool = Get-Command $Name -ErrorAction SilentlyContinue
  if ($null -eq $tool) {
    throw "$Name was not found in PATH."
  }
  return $tool.Source
}

function Get-Sha256([string]$Path) {
  return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

if (($Width % 16) -ne 0 -or ($Height % 16) -ne 0) {
  throw 'Width and height must be divisible by 16 for the current JM state schema.'
}

$isYuvInput = $PSCmdlet.ParameterSetName -eq 'Yuv'
$inputPath = if ($isYuvInput) {
  (Resolve-Path -LiteralPath $InputYuv).Path
} else {
  (Resolve-Path -LiteralPath $InputRgb).Path
}
$actualInputBytes = (Get-Item -LiteralPath $inputPath).Length
$expectedYuvBytes = [int64]$Width * [int64]$Height * 3L / 2L * [int64]$Frames
if ($isYuvInput) {
  if ($actualInputBytes -ne $expectedYuvBytes) {
    throw "Raw YUV size mismatch: expected exactly $expectedYuvBytes bytes for $Frames frames of ${Width}x${Height} yuv420p, got $actualInputBytes bytes."
  }
} else {
  $bytesPerPixel = if ($PixelFormat -in @('rgb24','bgr24')) { 3L } else { 4L }
  $expectedRgbBytes = [int64]$Width * [int64]$Height * $bytesPerPixel * [int64]$Frames
  if ($actualInputBytes -ne $expectedRgbBytes) {
    throw "Raw RGB size mismatch: expected exactly $expectedRgbBytes bytes for $Frames frames of ${Width}x${Height} $PixelFormat, got $actualInputBytes bytes."
  }
}

if ([string]::IsNullOrWhiteSpace($JmPackageRoot)) {
  $localConfigPath = Join-Path $repoRoot '.oracle-state.local.json'
  if (-not (Test-Path -LiteralPath $localConfigPath -PathType Leaf)) {
    throw 'JM package is not configured. Run setup_local.ps1 first or pass -JmPackageRoot.'
  }
  $localConfig = Get-Content -LiteralPath $localConfigPath -Raw | ConvertFrom-Json
  $JmPackageRoot = $localConfig.jm_package_root
}
$jmRoot = (Resolve-Path -LiteralPath $JmPackageRoot).Path
$sourceClosureScript = Join-Path $jmRoot 'scripts\run_b1_closure.ps1'
if (-not (Test-Path -LiteralPath $sourceClosureScript -PathType Leaf)) {
  throw "JM closure script not found: $sourceClosureScript"
}

$ffmpeg = Get-Tool 'ffmpeg'
$ffprobe = Get-Tool 'ffprobe'
$python = Get-Tool 'python'

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
  $stem = [IO.Path]::GetFileNameWithoutExtension($inputPath)
  $OutputDir = Join-Path $repoRoot (Join-Path 'runs' ($stem + '-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff')))
}
$outputPath = [IO.Path]::GetFullPath($OutputDir)
if (Test-Path -LiteralPath $outputPath) {
  $existing = Get-ChildItem -LiteralPath $outputPath -Force -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($null -ne $existing) {
    throw "Output directory already exists and is not empty: $outputPath"
  }
} else {
  New-Item -ItemType Directory -Path $outputPath | Out-Null
}

$logsDir = Join-Path $outputPath 'logs'
$jmRunRoot = Join-Path $outputPath 'work\jm-runs'
New-Item -ItemType Directory -Path $logsDir,$jmRunRoot -Force | Out-Null

$effectiveJmRoot = $jmRoot
$closureModePreset = $ModePreset
if ($ModePreset -eq 'skip-p16') {
  # The validated upstream p16 preset disables skip. Build an isolated runtime
  # shadow package and enable skip without modifying the user's JM installation.
  $effectiveJmRoot = Join-Path $outputPath 'work\jm-skip-p16-runtime'
  New-Item -ItemType Directory -Path `
    (Join-Path $effectiveJmRoot 'scripts'), `
    (Join-Path $effectiveJmRoot 'bin'), `
    (Join-Path $effectiveJmRoot 'cfg') -Force | Out-Null
  Copy-Item -LiteralPath $sourceClosureScript -Destination (Join-Path $effectiveJmRoot 'scripts\run_b1_closure.ps1')
  foreach ($relativePath in @(
    'bin\lencod_state_b1.exe',
    'bin\ldecod_ref.exe',
    'cfg\encoder_baseline.cfg',
    'cfg\state_ip_p16.cfg',
    'cfg\state_ip_variable_multiref.cfg',
    'cfg\decoder.cfg'
  )) {
    Copy-Item -LiteralPath (Join-Path $jmRoot $relativePath) -Destination (Join-Path $effectiveJmRoot $relativePath)
  }
  $runtimeP16Cfg = Join-Path $effectiveJmRoot 'cfg\state_ip_p16.cfg'
  $runtimeCfgText = Get-Content -LiteralPath $runtimeP16Cfg -Raw
  $runtimeCfgText = $runtimeCfgText -replace '(?m)^PSliceSkip\s*=\s*0\s*$', 'PSliceSkip               = 1'
  if ($runtimeCfgText -notmatch '(?m)^PSliceSkip\s*=\s*1\s*$') {
    throw 'Failed to enable PSliceSkip in the isolated p16 runtime config.'
  }
  [IO.File]::WriteAllText($runtimeP16Cfg, $runtimeCfgText, [Text.UTF8Encoding]::new($false))
  $closureModePreset = 'p16'
}
$closureScript = Join-Path $effectiveJmRoot 'scripts\run_b1_closure.ps1'

$yuvPath = Join-Path $outputPath 'input_yuv420p.yuv'
$ffmpegArgs = @(
  '-hide_banner','-loglevel','error','-y',
  '-f','rawvideo',
  '-pixel_format',$PixelFormat,
  '-video_size',"${Width}x${Height}",
  '-framerate',([string]::Format([Globalization.CultureInfo]::InvariantCulture, '{0}', $Fps)),
  '-i',$inputPath,
  '-frames:v',"$Frames",
  '-an','-pix_fmt','yuv420p','-f','rawvideo',$yuvPath
)
if ($isYuvInput) {
  Write-Host '[1/4] Importing existing YUV420p without color conversion...'
  Copy-Item -LiteralPath $inputPath -Destination $yuvPath
  $ffmpegArgs = $null
} else {
  Write-Host '[1/4] Converting packed RGB to YUV420p...'
  & $ffmpeg @ffmpegArgs *> (Join-Path $logsDir 'ffmpeg_rgb_to_yuv.log')
  if ($LASTEXITCODE -ne 0) {
    throw 'FFmpeg RGB-to-YUV conversion failed. See logs/ffmpeg_rgb_to_yuv.log.'
  }
}
$actualYuvBytes = (Get-Item -LiteralPath $yuvPath).Length
if ($actualYuvBytes -ne $expectedYuvBytes) {
  throw "YUV size mismatch: expected $expectedYuvBytes bytes, got $actualYuvBytes bytes."
}

Write-Host '[2/4] Exporting and replay-validating Oracle JM state...'
$closureLog = Join-Path $logsDir 'jm_closure_console.log'
& $closureScript `
  -InputYuv $yuvPath `
  -Width $Width -Height $Height -Frames $Frames -Fps $Fps `
  -Qp $Qp -ModePreset $closureModePreset -ReferenceFrames $ReferenceFrames `
  -RDOptimization $RDOptimization -RunRoot $jmRunRoot *>&1 |
  Tee-Object -FilePath $closureLog | Out-Host

$jmRuns = @(Get-ChildItem -LiteralPath $jmRunRoot -Directory | Sort-Object LastWriteTime)
if ($jmRuns.Count -ne 1) {
  throw "Expected exactly one JM run under $jmRunRoot, found $($jmRuns.Count)."
}
$jmRun = $jmRuns[0].FullName
$dumpDir = Join-Path $jmRun 'b1-dump'
$replayDir = Join-Path $jmRun 'b1-replay'

$copyMap = [ordered]@{
  (Join-Path $dumpDir 'state_b1.bin') = (Join-Path $outputPath 'oracle_state.bin')
  (Join-Path $dumpDir 'reference_b1.264') = (Join-Path $outputPath 'reference.264')
  (Join-Path $dumpDir 'reference_recon.yuv') = (Join-Path $outputPath 'reference_recon.yuv')
  (Join-Path $replayDir 'replayed_b1.264') = (Join-Path $outputPath 'replayed.264')
  (Join-Path $jmRun 'b1_result.json') = (Join-Path $outputPath 'closure_result.json')
  (Join-Path $dumpDir 'dump.log') = (Join-Path $logsDir 'jm_dump.log')
  (Join-Path $replayDir 'replay.log') = (Join-Path $logsDir 'jm_replay.log')
}
foreach ($entry in $copyMap.GetEnumerator()) {
  if (-not (Test-Path -LiteralPath $entry.Key -PathType Leaf)) {
    throw "Expected JM output is missing: $($entry.Key)"
  }
  Copy-Item -LiteralPath $entry.Key -Destination $entry.Value
}

Write-Host '[3/4] Parsing backend state into training tensors...'
$parser = Join-Path $repoRoot 'scripts\parse_b1_state.py'
$npzPath = Join-Path $outputPath 'oracle_labels.npz'
$stateMetadataPath = Join-Path $outputPath 'state_metadata.json'
& $python $parser --input (Join-Path $outputPath 'oracle_state.bin') --output-npz $npzPath --metadata-json $stateMetadataPath
if ($LASTEXITCODE -ne 0) {
  throw 'State parsing failed.'
}

Write-Host '[4/4] Writing reproducibility metadata...'
$ffmpegVersion = (& $ffmpeg -version | Select-Object -First 1)
$ffprobeVersion = (& $ffprobe -version | Select-Object -First 1)
$pythonVersion = (& $python --version 2>&1 | Select-Object -First 1)
$closureResult = Get-Content -LiteralPath (Join-Path $outputPath 'closure_result.json') -Raw | ConvertFrom-Json

$metadata = [ordered]@{
  schema = 'oracle-h264-state-exporter/run-v1'
  created_at = (Get-Date).ToString('o')
  pairing = [ordered]@{
    sample_id = if ([string]::IsNullOrWhiteSpace($SampleId)) { $null } else { $SampleId }
    diffusion_step = if ([string]::IsNullOrWhiteSpace($DiffusionStep)) { $null } else { $DiffusionStep }
    prompt = if ([string]::IsNullOrWhiteSpace($Prompt)) { $null } else { $Prompt }
    seed = if ([string]::IsNullOrWhiteSpace($Seed)) { $null } else { $Seed }
    feature_ref = if ([string]::IsNullOrWhiteSpace($FeatureRef)) { $null } else { $FeatureRef }
    latent_ref = if ([string]::IsNullOrWhiteSpace($LatentRef)) { $null } else { $LatentRef }
    hidden_ref = if ([string]::IsNullOrWhiteSpace($HiddenRef)) { $null } else { $HiddenRef }
    final_latent_ref = if ([string]::IsNullOrWhiteSpace($FinalLatentRef)) { $null } else { $FinalLatentRef }
    timestep = if ([string]::IsNullOrWhiteSpace($Timestep)) { $null } else { $Timestep }
    sigma = if ([string]::IsNullOrWhiteSpace($Sigma)) { $null } else { $Sigma }
    hidden_branch = if ([string]::IsNullOrWhiteSpace($HiddenBranch)) { $null } else { $HiddenBranch }
    hidden_block = if ([string]::IsNullOrWhiteSpace($HiddenBlock)) { $null } else { $HiddenBlock }
    checkpoint = if ([string]::IsNullOrWhiteSpace($Checkpoint)) { $null } else { $Checkpoint }
    sampler = if ([string]::IsNullOrWhiteSpace($Sampler)) { $null } else { $Sampler }
    total_steps = if ([string]::IsNullOrWhiteSpace($TotalSteps)) { $null } else { $TotalSteps }
  }
  input = [ordered]@{
    path = $inputPath
    sha256 = Get-Sha256 $inputPath
    pixel_format = if ($isYuvInput) { 'yuv420p' } else { $PixelFormat }
    source_kind = if ($isYuvInput) { 'raw-yuv420p' } else { 'packed-rgb' }
    color_matrix = $ColorMatrix
    color_range = $ColorRange
    width = $Width
    height = $Height
    frames = $Frames
    fps = $Fps
    bytes = $actualInputBytes
  }
  rgb_to_yuv = [ordered]@{
    performed = -not $isYuvInput
    output_pixel_format = 'yuv420p'
    output_bytes = $actualYuvBytes
    output_sha256 = Get-Sha256 $yuvPath
    ffmpeg_arguments = $ffmpegArgs
  }
  h264 = [ordered]@{
    profile = 'Baseline'
    entropy = 'CAVLC'
    gop = '1I + P'
    qp = $Qp
    mode_preset = $ModePreset
    upstream_closure_preset = $closureModePreset
    reference_frames = $ReferenceFrames
    rd_optimization = $RDOptimization
    state_schema = 'JMZV1/version-7-B1-KMR'
  }
  validation = [ordered]@{
    bitstream_bit_exact = $closureResult.bitstream_bit_exact
    jm_decoded_pixels_exact = $closureResult.jm_decoded_pixels_exact
    ffmpeg_decoded_pixels_exact = $closureResult.ffmpeg_decoded_pixels_exact
  }
  tools = [ordered]@{
    ffmpeg = $ffmpegVersion
    ffprobe = $ffprobeVersion
    python = $pythonVersion
    jm_package_root = $jmRoot
    effective_jm_runtime_root = $effectiveJmRoot
  }
  artifacts = [ordered]@{
    oracle_state_sha256 = Get-Sha256 (Join-Path $outputPath 'oracle_state.bin')
    oracle_labels_sha256 = Get-Sha256 $npzPath
    reference_bitstream_sha256 = Get-Sha256 (Join-Path $outputPath 'reference.264')
    replayed_bitstream_sha256 = Get-Sha256 (Join-Path $outputPath 'replayed.264')
    reference_recon_sha256 = Get-Sha256 (Join-Path $outputPath 'reference_recon.yuv')
  }
}

[IO.File]::WriteAllText(
  (Join-Path $outputPath 'metadata.json'),
  ($metadata | ConvertTo-Json -Depth 8),
  [Text.UTF8Encoding]::new($false)
)

Write-Host ''
Write-Host 'Oracle export completed:'
Write-Host "  $outputPath"
Write-Host "  state:  $(Join-Path $outputPath 'oracle_state.bin')"
Write-Host "  labels: $(Join-Path $outputPath 'oracle_labels.npz')"
Write-Host "  meta:   $(Join-Path $outputPath 'metadata.json')"
