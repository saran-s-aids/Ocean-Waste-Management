# Enable Windows Long Path Support
# This script must be run as Administrator

Write-Host "Enabling Windows Long Path Support..." -ForegroundColor Green

try {
    New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
                     -Name "LongPathsEnabled" `
                     -Value 1 `
                     -PropertyType DWORD `
                     -Force | Out-Null
    
    Write-Host "✓ Long Path Support has been enabled successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Important: You may need to restart your computer for changes to take effect." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "After restart, run:" -ForegroundColor Cyan
    Write-Host "  pip install tensorflow --no-cache-dir" -ForegroundColor White
    
} catch {
    Write-Host "✗ Error: Failed to enable Long Path Support" -ForegroundColor Red
    Write-Host "  Make sure you run PowerShell as Administrator" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To run as Administrator:" -ForegroundColor Cyan
    Write-Host "  1. Right-click PowerShell" -ForegroundColor White
    Write-Host "  2. Select 'Run as Administrator'" -ForegroundColor White
    Write-Host "  3. Navigate to: D:\DS_project" -ForegroundColor White
    Write-Host "  4. Run: .\enable_long_paths.ps1" -ForegroundColor White
}

Read-Host "`nPress Enter to exit"
