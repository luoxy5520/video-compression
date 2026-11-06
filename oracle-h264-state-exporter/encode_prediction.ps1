[CmdletBinding()]
param(
  [Parameter(Mandatory=$true)][string]$PredictionNpz,
  [Parameter(Mandatory=$true)][string]$OracleTemplateState,
  [Parameter(Mandatory=$true)][string]$OutputDir,
  [ValidateRange(1,240)][double]$Fps = 30,
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

$predictionPath = (Resolve-Path -LiteralPath $PredictionNpz).Path
$templatePath = (Resolve-Path -LiteralPath $OracleTemplateState).Path

if ([string]::IsNullOrWhiteSpace($JmPackageRoot)) {
  $localConfigPath = Join-Path $repoRoot '.oracle-state.local.json'
  if (-not (Test-Path -LiteralPath $localConfigPath -PathType Leaf)) {
    throw 'JM package is not configured. Run setup_local.ps1 first or pass -JmPackageRoot.'
  }
  $localConfig = Get-Content -LiteralPath $localConfigPath -Raw | ConvertFrom-Json
  $JmPackageRoot = $localConfig.jm_package_root
}
$jmRoot = (Resolve-Path -LiteralPath $JmPackageRoot).Path

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
$runtimeRoot = Join-Path $outputPath 'work\jm-skip-p16-runtime'
New-Item -ItemType Directory -Path $logsDir,(Join-Path $runtimeRoot 'bin'),(Join-Path $runtimeRoot 'cfg') -Force | Out-Null

$python = Get-Tool 'python'
$ffmpeg = Get-Tool 'ffmpeg'
$ffprobe = Get-Tool 'ffprobe'

Write-Host '[1/4] Converting predicted tensors to a replayable B1 state...'
$adapter = Join-Path $repoRoot 'scripts\prediction_to_b1_state.py'
$predictedState = Join-Path $outputPath 'predicted_state.bin'
$adapterResultPath = Join-Path $outputPath 'adapter_result.json'
& $python $adapter `
  --prediction $predictionPath `
  --oracle-template $templatePath `
  --output-state $predictedState `
  --result-json $adapterResultPath
if ($LASTEXITCODE -ne 0) {
  throw 'Prediction-to-state conversion failed.'
}
$adapterResult = Get-Content -LiteralPath $adapterResultPath -Raw | ConvertFrom-Json

if ($adapterResult.configuration.num_ref_frames -ne 1) {
  throw 'The current one-click backend requires exactly one reference frame.'
}
if ($adapterResult.configuration.profile_idc -ne 66 -or $adapterResult.configuration.symbol_mode -ne 0) {
  throw 'The current one-click backend requires Baseline Profile with CAVLC.'
}

Write-Host '[2/4] Preparing the isolated Skip/P16 JM replay runtime...'
$runtimeFiles = @(
  'bin\lencod_state_b1.exe',
  'bin\ldecod_ref.exe',
  'cfg\encoder_baseline.cfg',
  'cfg\state_ip_p16.cfg',
  'cfg\decoder.cfg'
)
foreach ($relativePath in $runtimeFiles) {
  $source = Join-Path $jmRoot $relativePath
  if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
    throw "Missing JM runtime artifact: $source"
  }
  Copy-Item -LiteralPath $source -Destination (Join-Path $runtimeRoot $relativePath)
}
$runtimeP16Cfg = Join-Path $runtimeRoot 'cfg\state_ip_p16.cfg'
$runtimeCfgText = Get-Content -LiteralPath $runtimeP16Cfg -Raw
$runtimeCfgText = $runtimeCfgText -replace '(?m)^PSliceSkip\s*=\s*0\s*$', 'PSliceSkip               = 1'
if ($runtimeCfgText -notmatch '(?m)^PSliceSkip\s*=\s*1\s*$') {
  throw 'Failed to enable PSliceSkip in the isolated p16 runtime config.'
}
[IO.File]::WriteAllText($runtimeP16Cfg, $runtimeCfgText, [Text.UTF8Encoding]::new($false))

$width = [int]$adapterResult.configuration.width
$height = [int]$adapterResult.configuration.height
$frames = [int]$adapterResult.configuration.frames
$qp = [int]$adapterResult.configuration.qp
$expectedBytes = [int64]$width * [int64]$height * 3L / 2L * [int64]$frames
$dummyYuv = Join-Path $outputPath 'work\dummy_zero.yuv'
$dummyStream = [IO.File]::Create($dummyYuv)
try {
  $dummyStream.SetLength($expectedBytes)
} finally {
  $dummyStream.Dispose()
}

Write-Host '[3/4] Replaying the predicted state into a standard H.264 bitstream...'
$encoder = Join-Path $runtimeRoot 'bin\lencod_state_b1.exe'
$baseCfg = Join-Path $runtimeRoot 'cfg\encoder_baseline.cfg'
$stateCfg = Join-Path $runtimeRoot 'cfg\state_ip_p16.cfg'
$bitstreamPath = Join-Path $outputPath 'predicted.264'
$encoderReconPath = Join-Path $outputPath 'predicted_encoder_recon.yuv'
$encoderLog = Join-Path $logsDir 'jm_prediction_replay.log'
$encoderArgs = @(
  '-d',$baseCfg,'-f',$stateCfg,
  '-p',"InputFile = $dummyYuv",
  '-p',"OutputFile = $bitstreamPath",
  '-p',"ReconFile = $encoderReconPath",
  '-p',"SourceWidth = $width",
  '-p',"SourceHeight = $height",
  '-p',"FramesToBeEncoded = $frames",
  '-p',"FrameRate = $Fps",
  '-p',"QPISlice = $qp",
  '-p',"QPPSlice = $qp",
  '-p','RDOptimization = 0',
  '-p','NumberReferenceFrames = 1',
  '-p','PList0References = 1'
)
$env:JM_STATE_MODE = 'replay'
$env:JM_STATE_FILE = $predictedState
Push-Location $outputPath
try {
  $encoderTimer = [Diagnostics.Stopwatch]::StartNew()
  & $encoder @encoderArgs *> $encoderLog
  $encoderExit = $LASTEXITCODE
  $encoderTimer.Stop()
  if ($encoderExit -ne 0) {
    throw "JM prediction replay failed with exit code $encoderExit. See $encoderLog."
  }
} finally {
  Pop-Location
  Remove-Item Env:JM_STATE_MODE -ErrorAction SilentlyContinue
  Remove-Item Env:JM_STATE_FILE -ErrorAction SilentlyContinue
}

Write-Host '[4/4] Verifying JM and FFmpeg standard decoding...'
$decoder = Join-Path $runtimeRoot 'bin\ldecod_ref.exe'
$decoderCfg = Join-Path $runtimeRoot 'cfg\decoder.cfg'
$jmDecodedPath = Join-Path $outputPath 'predicted_jm_decoded.yuv'
$jmDecoderLog = Join-Path $logsDir 'jm_decoder.log'
Push-Location $outputPath
try {
  $jmDecodeTimer = [Diagnostics.Stopwatch]::StartNew()
  & $decoder -d $decoderCfg -p "InputFile = $bitstreamPath" -p "OutputFile = $jmDecodedPath" -p 'RefFile = ""' *> $jmDecoderLog
  $jmDecodeExit = $LASTEXITCODE
  $jmDecodeTimer.Stop()
  if ($jmDecodeExit -ne 0) {
    throw "JM decoder failed with exit code $jmDecodeExit. See $jmDecoderLog."
  }
} finally {
  Pop-Location
}

$ffmpegDecodedPath = Join-Path $outputPath 'predicted_ffmpeg_decoded.yuv'
$ffmpegLog = Join-Path $logsDir 'ffmpeg_decoder.log'
$ffmpegTimer = [Diagnostics.Stopwatch]::StartNew()
& $ffmpeg -hide_banner -loglevel error -y -i $bitstreamPath -pix_fmt yuv420p -f rawvideo $ffmpegDecodedPath *> $ffmpegLog
$ffmpegExit = $LASTEXITCODE
$ffmpegTimer.Stop()
if ($ffmpegExit -ne 0) {
  throw "FFmpeg decoder failed with exit code $ffmpegExit. See $ffmpegLog."
}
$probeJson = & $ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,profile,width,height,pix_fmt,level -of json $bitstreamPath
[IO.File]::WriteAllText((Join-Path $outputPath 'ffprobe.json'), ($probeJson -join [Environment]::NewLine), [Text.UTF8Encoding]::new($false))

$encoderReconHash = Get-Sha256 $encoderReconPath
$jmDecodedHash = Get-Sha256 $jmDecodedPath
$ffmpegDecodedHash = Get-Sha256 $ffmpegDecodedPath
$result = [ordered]@{
  schema = 'oracle-h264-state-exporter/prediction-replay-v1'
  created_at = (Get-Date).ToString('o')
  inputs = [ordered]@{
    prediction = $predictionPath
    prediction_sha256 = Get-Sha256 $predictionPath
    oracle_template_state = $templatePath
    oracle_template_state_sha256 = Get-Sha256 $templatePath
  }
  configuration = $adapterResult.configuration
  adapter = $adapterResult
  outputs = [ordered]@{
    predicted_state = $predictedState
    predicted_state_sha256 = Get-Sha256 $predictedState
    predicted_bitstream = $bitstreamPath
    predicted_bitstream_sha256 = Get-Sha256 $bitstreamPath
    predicted_bitstream_bytes = (Get-Item -LiteralPath $bitstreamPath).Length
  }
  validation = [ordered]@{
    encoder_recon_sha256 = $encoderReconHash
    jm_decoded_sha256 = $jmDecodedHash
    ffmpeg_decoded_sha256 = $ffmpegDecodedHash
    encoder_recon_matches_standard_decode = ($jmDecodedHash -eq $encoderReconHash)
    jm_matches_ffmpeg = ($jmDecodedHash -eq $ffmpegDecodedHash)
    standard_decode_valid = ($jmDecodedHash -eq $ffmpegDecodedHash)
    note = 'JM replay encoder ReconFile is diagnostic only; standard JM and FFmpeg decoder agreement is the acceptance gate.'
  }
  timing_ms = [ordered]@{
    jm_replay = $encoderTimer.ElapsedMilliseconds
    jm_decode = $jmDecodeTimer.ElapsedMilliseconds
    ffmpeg_decode = $ffmpegTimer.ElapsedMilliseconds
  }
}
$resultPath = Join-Path $outputPath 'prediction_replay_result.json'
[IO.File]::WriteAllText($resultPath, ($result | ConvertTo-Json -Depth 10), [Text.UTF8Encoding]::new($false))

if (-not $result.validation.standard_decode_valid) {
  throw "Prediction bitstream decoded inconsistently. Inspect $resultPath."
}

Write-Host ''
Write-Host 'Prediction-to-H.264 completed:'
Write-Host "  bitstream: $bitstreamPath"
Write-Host "  state:     $predictedState"
Write-Host "  result:    $resultPath"
