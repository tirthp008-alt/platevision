$workspace = Split-Path $PSScriptRoot -Parent
foreach ($relative in @('static-deploy', 'platevision/frontend/public/console', 'plate-sight-live/public/console')) {
    $destination = Join-Path $workspace $relative
    New-Item -ItemType Directory -Force -Path $destination | Out-Null
    foreach ($file in @('index.html', 'styles.css', 'app.js', 'video-analysis.js', 'video-export.js', 'opencv.js')) {
        Copy-Item -LiteralPath (Join-Path $workspace $file) -Destination (Join-Path $destination $file) -Force
    }
}
