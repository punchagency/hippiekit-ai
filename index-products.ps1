# Index Products Script
# Run this after the AI service is running

param(
    [int]$MaxProducts = 10
)

Write-Host "╔═══════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║   Indexing Products to Pinecone                   ║" -ForegroundColor Cyan
Write-Host "╚═══════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

Write-Host "📊 Indexing $MaxProducts products..." -ForegroundColor Yellow
Write-Host ""

try {
    $response = Invoke-WebRequest `
        -Uri "http://localhost:8001/index/products?max_products=$MaxProducts" `
        -Method POST `
        -ContentType "application/json"
    
    $result = $response.Content | ConvertFrom-Json
    
    Write-Host "✅ Success!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Results:" -ForegroundColor Cyan
    Write-Host "  • Indexed: $($result.indexed_count) products" -ForegroundColor White
    Write-Host "  • Skipped: $($result.skipped_count) products" -ForegroundColor White
    Write-Host "  • Total vectors: $($result.index_stats.total_vectors)" -ForegroundColor White
    Write-Host ""
    Write-Host $result.message -ForegroundColor Green
    
} catch {
    Write-Host "❌ Error indexing products!" -ForegroundColor Red
    Write-Host ""
    
    if ($_.Exception.Response.StatusCode -eq 404) {
        Write-Host "The AI service is not running or not accessible." -ForegroundColor Yellow
        Write-Host "Make sure you started the service with: python main.py" -ForegroundColor Yellow
    } else {
        Write-Host "Error details: $($_.Exception.Message)" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "To index more products, run:" -ForegroundColor Cyan
Write-Host "  .\index-products.ps1 -MaxProducts 50" -ForegroundColor White
Write-Host ""
Write-Host "To index all products, run:" -ForegroundColor Cyan
Write-Host "  Invoke-WebRequest -Uri 'http://localhost:8001/index/products' -Method POST" -ForegroundColor White
