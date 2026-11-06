[CmdletBinding()]
param(
  [Parameter(Mandatory=$true)][string]$JmPackageRoot
)

$ErrorActionPreference = 'Stop'
$repoRoot = $PSScriptRoot
$jmRoot = (Resolve-Path -LiteralPath $JmPackageRoot).Path

$required = @(
  'scripts\run_b1_closure.ps1',
  'bin\lencod_state_b1.exe',
  'bin\ldecod_ref.exe',
  'cfg\encoder_baseline.cfg',
  'cfg\state_ip_p16.cfg',
  'cfg\state_ip_variable_multiref.cfg',
  'cfg\decoder.cfg'
)

foreach ($relativePath in $required) {
  $candidate = Join-Path $jmRoot $relativePath
  if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
    throw "Missing required JM artifact: $candidate"
  }
}

$config = [ordered]@{
  schema = 'oracle-h264-state-exporter/local-config-v1'
  jm_package_root = $jmRoot
  configured_at = (Get-Date).ToString('o')
}

$configPath = Join-Path $repoRoot '.oracle-state.local.json'
[IO.File]::WriteAllText(
  $configPath,
  ($config | ConvertTo-Json -Depth 4),
  [Text.UTF8Encoding]::new($false)
)

Write-Host "Configured JM package: $jmRoot"
Write-Host "Local configuration: $configPath"
