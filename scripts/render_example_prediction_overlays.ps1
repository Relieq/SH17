Add-Type -AssemblyName System.Drawing

$ExampleRoot = "D:\DS-AI\SH17\example_images"

$ModelOutputFiles = [ordered]@{
    "YOLOv9s Baseline"  = "yolov9s_baseline_overlay.jpg"
    "YOLOv9s Variant 3" = "yolov9s_variant3_overlay.jpg"
    "YOLOv9e"           = "yolov9e_overlay.jpg"
}

$SectionHeaders = @{
    "YOLOv9s Baseline projected output" = "YOLOv9s Baseline"
    "YOLOv9s Baseline projected labels" = "YOLOv9s Baseline"
    "YOLOv9s Variant 3 projected output" = "YOLOv9s Variant 3"
    "YOLOv9s Variant 3 projected labels" = "YOLOv9s Variant 3"
    "YOLOv9e projected output" = "YOLOv9e"
    "YOLOv9e projected labels" = "YOLOv9e"
}

$ClassColors = @{
    "person"       = [System.Drawing.Color]::FromArgb(52, 101, 164)
    "head"         = [System.Drawing.Color]::FromArgb(64, 160, 171)
    "face"         = [System.Drawing.Color]::FromArgb(80, 180, 215)
    "ear"          = [System.Drawing.Color]::FromArgb(118, 196, 210)
    "ear-mufs"     = [System.Drawing.Color]::FromArgb(120, 92, 196)
    "face-guard"   = [System.Drawing.Color]::FromArgb(123, 104, 238)
    "face-mask"    = [System.Drawing.Color]::FromArgb(196, 92, 174)
    "glasses"      = [System.Drawing.Color]::FromArgb(90, 127, 214)
    "hands"        = [System.Drawing.Color]::FromArgb(232, 146, 60)
    "gloves"       = [System.Drawing.Color]::FromArgb(241, 180, 76)
    "helmet"       = [System.Drawing.Color]::FromArgb(236, 201, 75)
    "safety-vest"  = [System.Drawing.Color]::FromArgb(133, 186, 68)
    "medical-suit" = [System.Drawing.Color]::FromArgb(60, 142, 108)
    "safety-suit"  = [System.Drawing.Color]::FromArgb(75, 160, 135)
    "tool"         = [System.Drawing.Color]::FromArgb(138, 109, 82)
    "foot"         = [System.Drawing.Color]::FromArgb(210, 92, 78)
    "shoes"        = [System.Drawing.Color]::FromArgb(182, 64, 62)
}

function Parse-PredictionFile {
    param(
        [string]$Path
    )

    $predictionPattern = '^(?<id>\d+)\s+(?<name>[A-Za-z0-9\-]+)\s+(?<conf>\d+(?:\.\d+)?)\s+(?<xc>\d+(?:\.\d+)?)\s+(?<yc>\d+(?:\.\d+)?)\s+(?<w>\d+(?:\.\d+)?)\s+(?<h>\d+(?:\.\d+)?)$'
    $falsePositivePattern = '^FP\s+(?<id>\d+)\s+(?<name>[A-Za-z0-9\-]+)\s+(?<conf>\d+(?:\.\d+)?)\s+(?<xc>\d+(?:\.\d+)?)\s+(?<yc>\d+(?:\.\d+)?)\s+(?<w>\d+(?:\.\d+)?)\s+(?<h>\d+(?:\.\d+)?)$'

    $sections = @{}
    $currentModel = $null

    foreach ($rawLine in Get-Content -LiteralPath $Path) {
        $line = $rawLine.Trim()
        if (-not $line) {
            continue
        }

        if ($SectionHeaders.ContainsKey($line)) {
            $currentModel = $SectionHeaders[$line]
            if (-not $sections.ContainsKey($currentModel)) {
                $sections[$currentModel] = New-Object System.Collections.ArrayList
            }
            continue
        }

        if (-not $currentModel) {
            continue
        }

        if ($line -match $predictionPattern) {
            $null = $sections[$currentModel].Add([pscustomobject]@{
                    ClassId         = [int]$Matches.id
                    ClassName       = $Matches.name
                    Confidence      = [double]$Matches.conf
                    XCenter         = [double]$Matches.xc
                    YCenter         = [double]$Matches.yc
                    Width           = [double]$Matches.w
                    Height          = [double]$Matches.h
                    IsFalsePositive = $false
                })
            continue
        }

        if ($line -match $falsePositivePattern) {
            $null = $sections[$currentModel].Add([pscustomobject]@{
                    ClassId         = [int]$Matches.id
                    ClassName       = $Matches.name
                    Confidence      = [double]$Matches.conf
                    XCenter         = [double]$Matches.xc
                    YCenter         = [double]$Matches.yc
                    Width           = [double]$Matches.w
                    Height          = [double]$Matches.h
                    IsFalsePositive = $true
                })
        }
    }

    return $sections
}

function Convert-YoloToPixels {
    param(
        [pscustomobject]$Prediction,
        [int]$ImageWidth,
        [int]$ImageHeight
    )

    $boxWidth = $Prediction.Width * $ImageWidth
    $boxHeight = $Prediction.Height * $ImageHeight
    $centerX = $Prediction.XCenter * $ImageWidth
    $centerY = $Prediction.YCenter * $ImageHeight

    return @{
        X1 = [int][Math]::Round($centerX - ($boxWidth / 2.0))
        Y1 = [int][Math]::Round($centerY - ($boxHeight / 2.0))
        X2 = [int][Math]::Round($centerX + ($boxWidth / 2.0))
        Y2 = [int][Math]::Round($centerY + ($boxHeight / 2.0))
    }
}

function Draw-Label {
    param(
        [System.Drawing.Graphics]$Graphics,
        [System.Drawing.Rectangle]$Box,
        [string]$Text,
        [System.Drawing.Color]$Color,
        [System.Drawing.Font]$Font,
        [int]$ImageWidth,
        [int]$ImageHeight
    )

    $paddingX = 4
    $paddingY = 2
    $measured = $Graphics.MeasureString($Text, $Font)
    $labelWidth = [int][Math]::Ceiling($measured.Width) + ($paddingX * 2)
    $labelHeight = [int][Math]::Ceiling($measured.Height) + ($paddingY * 2)

    $labelX = [Math]::Max(0, [Math]::Min($Box.X, $ImageWidth - $labelWidth))
    $labelY = $Box.Y - $labelHeight - 2
    if ($labelY -lt 0) {
        $labelY = [Math]::Min($ImageHeight - $labelHeight, $Box.Bottom + 2)
    }

    $background = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(220, $Color))
    $foreground = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::White)
    try {
        $Graphics.FillRectangle($background, $labelX, $labelY, $labelWidth, $labelHeight)
        $Graphics.DrawString($Text, $Font, $foreground, $labelX + $paddingX, $labelY + $paddingY - 1)
    }
    finally {
        $background.Dispose()
        $foreground.Dispose()
    }
}

function Render-Overlay {
    param(
        [string]$ImagePath,
        [System.Collections.ArrayList]$Predictions,
        [string]$OutputPath
    )

    $bitmap = [System.Drawing.Bitmap]::new($ImagePath)
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    $fontSize = [Math]::Max(12, [int][Math]::Round(([Math]::Min($bitmap.Width, $bitmap.Height)) / 55.0))
    $font = New-Object System.Drawing.Font("Segoe UI", $fontSize, [System.Drawing.FontStyle]::Bold, [System.Drawing.GraphicsUnit]::Pixel)

    try {
        $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::HighQuality
        $graphics.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
        $graphics.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality
        $graphics.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::ClearTypeGridFit

        $strokeWidth = [Math]::Max(2, [int][Math]::Round(([Math]::Min($bitmap.Width, $bitmap.Height)) / 500.0))
        $includeConfidence = $Predictions.Count -le 18
        $labelAreaFactor = if ($Predictions.Count -le 18) {
            0.0025
        }
        elseif ($Predictions.Count -le 30) {
            0.0045
        }
        else {
            0.0075
        }
        $minLabelArea = $bitmap.Width * $bitmap.Height * $labelAreaFactor

        foreach ($prediction in $Predictions) {
            $color = if ($ClassColors.ContainsKey($prediction.ClassName)) { $ClassColors[$prediction.ClassName] } else { [System.Drawing.Color]::White }
            $coords = Convert-YoloToPixels -Prediction $prediction -ImageWidth $bitmap.Width -ImageHeight $bitmap.Height
            $rectWidth = [Math]::Max(1, $coords.X2 - $coords.X1)
            $rectHeight = [Math]::Max(1, $coords.Y2 - $coords.Y1)
            $box = New-Object System.Drawing.Rectangle($coords.X1, $coords.Y1, $rectWidth, $rectHeight)

            $penWidth = if ($prediction.IsFalsePositive) { [Math]::Max(1, $strokeWidth - 1) } else { $strokeWidth }
            $pen = New-Object System.Drawing.Pen($color, $penWidth)
            try {
                if ($prediction.IsFalsePositive) {
                    $pen.DashStyle = [System.Drawing.Drawing2D.DashStyle]::Dash
                }
                $graphics.DrawRectangle($pen, $box)
            }
            finally {
                $pen.Dispose()
            }

            $boxArea = $rectWidth * $rectHeight
            if ($boxArea -lt $minLabelArea) {
                continue
            }

            if ($prediction.IsFalsePositive) {
                $label = "FP $($prediction.ClassName)"
            }
            elseif ($includeConfidence) {
                $label = "{0} {1:N2}" -f $prediction.ClassName, $prediction.Confidence
            }
            else {
                $label = $prediction.ClassName
            }

            Draw-Label -Graphics $graphics -Box $box -Text $label -Color $color -Font $font -ImageWidth $bitmap.Width -ImageHeight $bitmap.Height
        }

        $bitmap.Save($OutputPath, [System.Drawing.Imaging.ImageFormat]::Jpeg)
    }
    finally {
        $font.Dispose()
        $graphics.Dispose()
        $bitmap.Dispose()
    }
}

function Get-ImageFile {
    param(
        [string]$ExampleDir
    )

    $file = Get-ChildItem -LiteralPath $ExampleDir -Filter *.jpeg | Sort-Object Name | Select-Object -First 1
    if (-not $file) {
        throw "No .jpeg image found in $ExampleDir"
    }
    return $file.FullName
}

$createdFiles = New-Object System.Collections.ArrayList

Get-ChildItem -LiteralPath $ExampleRoot -Directory | Where-Object { $_.Name -like "image_*" } | Sort-Object Name | ForEach-Object {
    $exampleDir = $_.FullName
    $predictedFile = Get-ChildItem -LiteralPath $exampleDir -Filter *_predicted.md | Sort-Object Name | Select-Object -First 1
    if (-not $predictedFile) {
        return
    }

    $imagePath = Get-ImageFile -ExampleDir $exampleDir
    $rendersDir = Join-Path $exampleDir "renders"
    New-Item -ItemType Directory -Path $rendersDir -Force | Out-Null

    $sections = Parse-PredictionFile -Path $predictedFile.FullName
    foreach ($modelName in $ModelOutputFiles.Keys) {
        if (-not $sections.ContainsKey($modelName)) {
            continue
        }
        $outputPath = Join-Path $rendersDir $ModelOutputFiles[$modelName]
        Render-Overlay -ImagePath $imagePath -Predictions $sections[$modelName] -OutputPath $outputPath
        $null = $createdFiles.Add($outputPath)
    }
}

Write-Output ("Created {0} overlay images." -f $createdFiles.Count)
$createdFiles | ForEach-Object { Write-Output $_ }
