param(
    [switch]$SkipRfid,
    [switch]$SkipWorkbook,
    [string]$RfidMapping = "",
    [string]$Recap = "data\LMT RECAP ALL EXPERIMENTS.xlsx",
    [string]$DateCorrections = "data\manual_date_corrections.csv",
    [string]$RfidOutputRoot = "D:\lmt_rfid_reconstruction\runs",
    [string]$AnalysisOutputRoot = "D:\lmt_thesis_analysis\runs",
    [int]$RfidBootstrapSamples = 200,
    [int]$RfidNullSamples = 1000,
    [int]$AnalysisBootstrapIterations = 1000,
    [int]$AnalysisPermutationIterations = 10000,
    [int]$Seed = 20240612
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = (Get-Command python).Source
$Node = "C:\Users\andre\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe"
$NodeModules = "C:\Users\andre\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\node_modules"
$LocalNodeModules = Join-Path $PSScriptRoot "node_modules"

Push-Location $ProjectRoot
try {
    $RfidRun = $null
    if (-not $SkipRfid) {
        & $Python -m src.reconstruction `
            --output-root $RfidOutputRoot `
            --bootstrap-samples $RfidBootstrapSamples `
            --null-samples $RfidNullSamples
        if ($LASTEXITCODE -ne 0) {
            throw "RFID reconstruction failed with exit code $LASTEXITCODE"
        }
        $RfidRun = Get-ChildItem $RfidOutputRoot -Directory |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1
        $RfidMapping = Join-Path $RfidRun.FullName "rfid_id_mapping.csv"
    }

    $AnalysisArgs = @(
        "-m", "src.analysis.thesis_analysis",
        "--output-root", $AnalysisOutputRoot,
        "--bootstrap-iterations", $AnalysisBootstrapIterations,
        "--permutation-iterations", $AnalysisPermutationIterations,
        "--seed", $Seed
    )
    if ($RfidMapping -and (Test-Path -LiteralPath $RfidMapping)) {
        $AnalysisArgs += @("--rfid-map", $RfidMapping)
    }
    if ($Recap) {
        if (-not (Test-Path -LiteralPath $Recap)) {
            throw "Recap workbook not found: $Recap"
        }
        $AnalysisArgs += @("--recap", $Recap)
    }
    if ($DateCorrections) {
        $ResolvedDateCorrections = $DateCorrections
        if (-not [System.IO.Path]::IsPathRooted($DateCorrections)) {
            $ResolvedDateCorrections = Join-Path $ProjectRoot $DateCorrections
        }
        if (Test-Path -LiteralPath $ResolvedDateCorrections) {
            $AnalysisArgs += @("--date-corrections", $ResolvedDateCorrections)
        }
    }
    $env:LMT_DEBUG = "False"
    & $Python @AnalysisArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Thesis analysis failed with exit code $LASTEXITCODE"
    }

    $AnalysisRun = Get-ChildItem $AnalysisOutputRoot -Directory |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -eq $AnalysisRun) {
        throw "No analysis run directory was created"
    }

    if (-not $SkipWorkbook) {
        if (-not (Test-Path $LocalNodeModules)) {
            New-Item -ItemType Junction -Path $LocalNodeModules -Target $NodeModules | Out-Null
        }
        $WorkbookPath = Join-Path $AnalysisRun.FullName "LMT_thesis_analysis.xlsx"
        & $Node (Join-Path $PSScriptRoot "build_thesis_workbook.mjs") $AnalysisRun.FullName $WorkbookPath
        if ($LASTEXITCODE -ne 0 -and -not (Test-Path $WorkbookPath)) {
            throw "Workbook generation failed with exit code $LASTEXITCODE"
        }
        & $Python -c "import sys,zipfile; z=zipfile.ZipFile(sys.argv[1]); assert z.testzip() is None" $WorkbookPath
        if ($LASTEXITCODE -ne 0) {
            throw "Workbook ZIP validation failed"
        }
    }

    Write-Output "Analysis run: $($AnalysisRun.FullName)"
}
finally {
    Pop-Location
}
