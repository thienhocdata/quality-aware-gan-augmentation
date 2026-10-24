$ErrorActionPreference = 'Stop'
$releaseRoot = $PSScriptRoot
$manifestPath = Join-Path $releaseRoot 'FULL_ARCHIVE_MANIFEST.json'
$releaseManifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ([System.IO.Path]::GetFileName($releaseManifest.archive) -ne $releaseManifest.archive) {
    throw 'Invalid archive filename in manifest.'
}
$targetPath = Join-Path $releaseRoot $releaseManifest.archive
if (Test-Path -LiteralPath $targetPath) {
    if ((Get-FileHash -LiteralPath $targetPath -Algorithm SHA256).Hash.ToLowerInvariant() -eq $releaseManifest.sha256) {
        Write-Host "Archive already verified: $targetPath"
        exit 0
    }
    throw 'An archive with the same name already exists but its hash differs. Move it before retrying.'
}
foreach ($releasePart in $releaseManifest.parts) {
    if ([System.IO.Path]::GetFileName($releasePart.file) -ne $releasePart.file) {
        throw 'Invalid part filename in manifest.'
    }
    $partPath = Join-Path $releaseRoot $releasePart.file
    if (!(Test-Path -LiteralPath $partPath -PathType Leaf)) { throw "Missing part: $($releasePart.file)" }
    if ((Get-Item -LiteralPath $partPath).Length -ne $releasePart.bytes) { throw "Size mismatch: $($releasePart.file)" }
    if ((Get-FileHash -LiteralPath $partPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $releasePart.sha256) {
        throw "SHA256 mismatch: $($releasePart.file)"
    }
}
$temporaryPath = "$targetPath.partial"
if (Test-Path -LiteralPath $temporaryPath) { throw 'A .partial file exists. Move it before retrying.' }
$destinationStream = [System.IO.File]::Open($temporaryPath, [System.IO.FileMode]::CreateNew)
try {
    foreach ($releasePart in $releaseManifest.parts) {
        $sourceStream = [System.IO.File]::OpenRead((Join-Path $releaseRoot $releasePart.file))
        try { $sourceStream.CopyTo($destinationStream) } finally { $sourceStream.Dispose() }
    }
} finally { $destinationStream.Dispose() }
if ((Get-Item -LiteralPath $temporaryPath).Length -ne $releaseManifest.bytes) { throw 'Joined size mismatch.' }
if ((Get-FileHash -LiteralPath $temporaryPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $releaseManifest.sha256) {
    throw 'Joined SHA256 mismatch. The .partial file is preserved for inspection.'
}
Move-Item -LiteralPath $temporaryPath -Destination $targetPath
Write-Host "SHA256 verified: $targetPath"
Write-Host 'Extract this ZIP with 7-Zip, then open KLTN/notebooks/00_MASTER_CURRENT.ipynb.'
