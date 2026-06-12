# Script pour identifier et nettoyer les labels orphelins dans un dataset YOLO Roboflow
# Structure attendue: dataset/train/images + dataset/train/labels, idem pour valid, test

param(
    [string]$DatasetPath = ".",
    [switch]$Delete = $false
)

$ImageExtensions = @('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
$Splits = @('train', 'valid', 'test')

Write-Host "=== Nettoyage des labels orphelins ===" -ForegroundColor Cyan
Write-Host "Dataset: $DatasetPath"
Write-Host ""

$TotalOrphans = 0

foreach ($split in $Splits) {
    $ImagesPath = Join-Path (Join-Path $DatasetPath $split) "images"
    $LabelsPath = Join-Path (Join-Path $DatasetPath $split) "labels"

    # Vérifier que les dossiers existent
    if (-not (Test-Path $ImagesPath) -or -not (Test-Path $LabelsPath)) {
        continue
    }

    Write-Host "--- Split: $split ---" -ForegroundColor Cyan

    # Récupérer tous les fichiers images (sans extension)
    $ImageFiles = @()
    foreach ($ext in $ImageExtensions) {
        $ImageFiles += Get-ChildItem -Path $ImagesPath -Filter "*$ext" | Select-Object -ExpandProperty BaseName
    }
    $ImageFiles = $ImageFiles | Sort-Object -Unique

    Write-Host "Images: $($ImageFiles.Count)"

    # Récupérer tous les fichiers labels
    $LabelFiles = Get-ChildItem -Path $LabelsPath -Filter "*.txt" | Select-Object -ExpandProperty BaseName
    $LabelFiles = $LabelFiles | Sort-Object -Unique

    Write-Host "Labels: $($LabelFiles.Count)"

    # Identifier les labels orphelins
    $OrphanLabels = @()
    foreach ($label in $LabelFiles) {
        if ($label -notin $ImageFiles) {
            $OrphanLabels += $label
        }
    }

    if ($OrphanLabels.Count -eq 0) {
        Write-Host "OK - Aucun orphelin"
    } else {
        Write-Host "ORPHELINS: $($OrphanLabels.Count)" -ForegroundColor Yellow
        $OrphanLabels | ForEach-Object {
            Write-Host "  - $_"
        }
        $TotalOrphans += $OrphanLabels.Count

        if ($Delete) {
            Write-Host "Suppression..." -ForegroundColor Red
            foreach ($label in $OrphanLabels) {
                $LabelFile = Join-Path $LabelsPath "$label.txt"
                Remove-Item -Path $LabelFile -Force
            }
        }
    }
    Write-Host ""
}

Write-Host "=== Résumé ===" -ForegroundColor Cyan
if ($TotalOrphans -eq 0) {
    Write-Host "OK - Dataset coherent, aucun orphelin!" -ForegroundColor Green
} elseif ($Delete) {
    Write-Host "Nettoyage determine: $TotalOrphans labels supprimes" -ForegroundColor Green
} else {
    Write-Host "$TotalOrphans labels orphelins detects" -ForegroundColor Yellow
    Write-Host "Pour les supprimer: .\cleanup_orphan_labels.ps1 -DatasetPath 'ton/chemin' -Delete" -ForegroundColor Cyan
}
