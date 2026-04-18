using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using tradelog.Data;
using tradelog.Dtos;
using tradelog.Models;

namespace tradelog.Controllers;

[ApiController]
[Route("api/daily-prep")]
[Produces("application/json")]
public class DailyPrepController : ControllerBase
{
    private readonly DataContext _context;

    public DailyPrepController(DataContext context)
    {
        _context = context;
    }

    [HttpGet("latest")]
    public async Task<ActionResult<DailyPrepDto>> GetLatest()
    {
        var entity = await _context.DailyPreps
            .OrderByDescending(e => e.Date)
            .FirstOrDefaultAsync();

        if (entity == null) return NotFound();
        return ToDto(entity);
    }

    [HttpGet]
    public async Task<ActionResult<DailyPrepDto>> GetByDate([FromQuery] DateTime date)
    {
        var dateOnly = date.Date;
        var entity = await _context.DailyPreps
            .FirstOrDefaultAsync(e => e.Date == dateOnly);

        if (entity == null) return NotFound();
        return ToDto(entity);
    }

    [HttpPost]
    public async Task<ActionResult<DailyPrepDto>> Upsert([FromBody] DailyPrepUpsertDto dto)
    {
        var dateOnly = dto.Date.Date;
        var existing = await _context.DailyPreps
            .FirstOrDefaultAsync(e => e.Date == dateOnly);

        if (existing != null)
        {
            existing.MarketSummary = dto.MarketSummary;
            existing.Watchlist = dto.Watchlist;
            existing.EmailCount = dto.EmailCount;
            existing.CandidateCount = dto.CandidateCount;
            existing.UpdatedAt = DateTime.UtcNow;
        }
        else
        {
            existing = new DailyPrep
            {
                Date = dateOnly,
                MarketSummary = dto.MarketSummary,
                Watchlist = dto.Watchlist,
                EmailCount = dto.EmailCount,
                CandidateCount = dto.CandidateCount,
            };
            _context.DailyPreps.Add(existing);
        }

        await _context.SaveChangesAsync();
        return ToDto(existing);
    }

    private static DailyPrepDto ToDto(DailyPrep e) => new()
    {
        Id = e.Id,
        Date = e.Date,
        MarketSummary = e.MarketSummary,
        Watchlist = e.Watchlist,
        EmailCount = e.EmailCount,
        CandidateCount = e.CandidateCount,
        CreatedAt = e.CreatedAt,
        UpdatedAt = e.UpdatedAt,
    };
}
