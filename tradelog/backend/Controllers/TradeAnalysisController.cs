using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using tradelog.Data;
using tradelog.Dtos;
using tradelog.Models;

namespace tradelog.Controllers;

[ApiController]
[Route("api/trades/{tradeId}/analysis")]
[Produces("application/json")]
public class TradeAnalysisController : ControllerBase
{
    private readonly DataContext _context;

    public TradeAnalysisController(DataContext context)
    {
        _context = context;
    }

    [HttpGet]
    public async Task<ActionResult<IEnumerable<TradeAnalysisDto>>> GetAll(int tradeId)
    {
        var analyses = await _context.TradeAnalyses
            .Where(a => a.TradeId == tradeId)
            .OrderByDescending(a => a.AnalysisDate)
            .ToListAsync();

        return analyses.Select(ToDto).ToList();
    }

    [HttpPost]
    public async Task<ActionResult<TradeAnalysisDto>> Create(int tradeId, [FromBody] TradeAnalysisCreateDto dto)
    {
        var trade = await _context.Trades.FindAsync(tradeId);
        if (trade == null) return NotFound($"Trade {tradeId} not found");

        var entity = new TradeAnalysis
        {
            TradeId = tradeId,
            AnalysisDate = dto.AnalysisDate.Date,
            Score = dto.Score,
            Analysis = dto.Analysis,
            Model = dto.Model,
        };

        _context.TradeAnalyses.Add(entity);
        await _context.SaveChangesAsync();

        return CreatedAtAction(nameof(GetAll), new { tradeId }, ToDto(entity));
    }

    [HttpPut("{analysisId}")]
    public async Task<ActionResult<TradeAnalysisDto>> Update(int tradeId, int analysisId, [FromBody] TradeAnalysisUpdateDto dto)
    {
        var entity = await _context.TradeAnalyses
            .FirstOrDefaultAsync(a => a.Id == analysisId && a.TradeId == tradeId);

        if (entity == null) return NotFound();

        entity.Score = dto.Score;
        entity.Analysis = dto.Analysis;
        await _context.SaveChangesAsync();

        return ToDto(entity);
    }

    private static TradeAnalysisDto ToDto(TradeAnalysis e) => new()
    {
        Id = e.Id,
        TradeId = e.TradeId,
        AnalysisDate = e.AnalysisDate,
        Score = e.Score,
        Analysis = e.Analysis,
        Model = e.Model,
        CreatedAt = e.CreatedAt,
    };
}
