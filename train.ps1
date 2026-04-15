param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PassThruArgs
)

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$propy = Join-Path $env:ProgramFiles 'ArcGIS\Pro\bin\Python\Scripts\propy.bat'

if (-not (Test-Path $propy)) {
    throw "ArcGIS Pro Python launcher not found at $propy"
}

& $propy (Join-Path $repoRoot 'train.py') @PassThruArgs
exit $LASTEXITCODE
