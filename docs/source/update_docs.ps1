# Rebuild the Word guide from its sources (Windows, PowerShell 7, Python 3 with Pillow, Microsoft Word).
#
#   pwsh docs/source/update_docs.ps1
#
# Sources in this folder:
#   manual.html        document content (edit this when the system changes)
#   make_diagrams.py   diagrams (docs/images/en)
#   build_docx.py      converts manual.html + diagrams into the .docx (Office Open XML, no extra libraries)
# Word is then used only to fill in the table-of-contents page numbers; if Word is not available the
# document is still complete and Word offers to update the table of contents when it is opened.
$ErrorActionPreference = "Stop"
$src  = $PSScriptRoot
$docx = Join-Path (Split-Path $src -Parent) "Cancer_Image_Diagnosis_System_Manual.docx"
$env:PYTHONIOENCODING = "utf-8"

python (Join-Path $src "make_diagrams.py")
python (Join-Path $src "build_docx.py") $docx

$before = @(Get-Process WINWORD -ErrorAction SilentlyContinue | ForEach-Object Id)
$job = Start-Job -ScriptBlock {
    param($docx)
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false; $word.DisplayAlerts = 0
    try {
        $doc = $word.Documents.Open($docx, $false, $false, $false)
        foreach ($t in $doc.TablesOfContents) { $t.Update() }
        $doc.Save()
        "table of contents updated, pages: " + $doc.ComputeStatistics(2)
        $doc.Close(0)
    } finally { $word.Quit() }
} -ArgumentList $docx
if (Wait-Job $job -Timeout 120) { Receive-Job $job }
else {
    Write-Warning "Word did not respond; the table of contents will be updated when the document is opened."
    Stop-Job $job
    Get-Process WINWORD -ErrorAction SilentlyContinue | Where-Object { $before -notcontains $_.Id } |
        ForEach-Object { Stop-Process -Id $_.Id -Force -Confirm:$false }
}
Remove-Job $job -Force
"done: $docx"
