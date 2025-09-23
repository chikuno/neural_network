# CI-friendly smoke test: runs inference non-interactively and checks output
$python = "c:\Users\USER\Miniconda3\python.exe"
$script = "c:\Users\USER\OneDrive\Desktop\neural_network\main.py"

$process = Start-Process -FilePath $python -ArgumentList "`"$script`" --mode infer --skip-chat" -NoNewWindow -RedirectStandardOutput stdout.txt -RedirectStandardError stderr.txt -PassThru -Wait
$stdout = Get-Content stdout.txt -Raw
$stderr = Get-Content stderr.txt -Raw

Write-Host "ExitCode: $($process.ExitCode)"
if ($process.ExitCode -ne 0) {
    Write-Host "Smoke test failed with non-zero exit code. STDERR:" -ForegroundColor Red
    Write-Host $stderr
    exit 1
}

# Basic check: look for 'Generated' or non-empty ensemble line
if ($stdout -match "Ensemble Generated Text:" -or $stdout -match "Generated Text") {
    Write-Host "Smoke test passed. Output snippet:" -ForegroundColor Green
    $snippet = ($stdout -split "\n") | Select-Object -Last 10
    $snippet -join "\n" | Write-Host
    exit 0
} else {
    Write-Host "Smoke test did not find expected output. STDOUT:" -ForegroundColor Yellow
    Write-Host $stdout
    exit 2
}
