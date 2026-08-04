$ErrorActionPreference = "Stop"
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0

$files = Get-ChildItem -Path "data_fetch" -Filter "*.doc"
foreach ($file in $files) {
    if ($file.Name.StartsWith("~$")) { continue }
    $docxPath = Join-Path -Path "data_fetch" -ChildPath ($file.BaseName + ".docx")
    if (Test-Path $docxPath) { continue }
    Write-Host "Converting: $($file.Name)"
    $doc = $word.Documents.Open($file.FullName, $false, $true)
    $doc.SaveAs([ref]$docxPath, [ref]16)
    $doc.Close()
}
$word.Quit()
