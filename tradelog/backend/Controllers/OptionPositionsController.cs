using tradelog.Data;
using tradelog.Dtos;
using tradelog.Models;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

namespace tradelog.Controllers;

[ApiController]
[Route("api/option-positions")]
[Produces("application/json")]
public class OptionPositionsController : ControllerBase
{
    private readonly DataContext _context;

    public OptionPositionsController(DataContext context)
    {
        _context = context;
    }

    [HttpGet]
    public async Task<ActionResult<IEnumerable<OptionPositionDto>>> GetAll(
        [FromQuery] string? symbol,
        [FromQuery] string? status)
    {
        var query = _context.OptionPositions.AsQueryable();

        if (!string.IsNullOrWhiteSpace(symbol))
            query = query.Where(p => p.Symbol == symbol);

        if (string.Equals(status, "open", StringComparison.OrdinalIgnoreCase))
            query = query.Where(p => p.Closed == null);
        else if (string.Equals(status, "closed", StringComparison.OrdinalIgnoreCase))
            query = query.Where(p => p.Closed != null);

        var positions = await query.OrderBy(p => p.Symbol).ThenBy(p => p.Expiry).ToListAsync();
        var contractIds = positions.Select(p => p.ContractId).Distinct().ToList();

        // Get latest log entry per contractId in one query
        var latestLogs = await _context.OptionPositionsLogs
            .Where(l => contractIds.Contains(l.ContractId))
            .GroupBy(l => l.ContractId)
            .Select(g => g.OrderByDescending(l => l.DateTime).First())
            .ToDictionaryAsync(l => l.ContractId);

        return positions.Select(p => ToDto(p, latestLogs.GetValueOrDefault(p.ContractId))).ToList();
    }

    [HttpGet("{id:int}")]
    public async Task<ActionResult<OptionPositionDto>> GetById(int id)
    {
        var position = await _context.OptionPositions.FindAsync(id);
        if (position == null) return NotFound();

        var latestLog = await _context.OptionPositionsLogs
            .Where(l => l.ContractId == position.ContractId)
            .OrderByDescending(l => l.DateTime)
            .FirstOrDefaultAsync();

        return ToDto(position, latestLog);
    }

    [HttpPost]
    public async Task<ActionResult<OptionPositionDto>> Create(OptionPosition position)
    {
        _context.OptionPositions.Add(position);
        await _context.SaveChangesAsync();
        return CreatedAtAction(nameof(GetById), new { id = position.Id }, ToDto(position, null));
    }

    [HttpPut("{id:int}")]
    public async Task<IActionResult> Update(int id, OptionPosition position)
    {
        if (id != position.Id) return BadRequest();

        var existing = await _context.OptionPositions.FindAsync(id);
        if (existing == null) return NotFound();

        _context.Entry(existing).CurrentValues.SetValues(position);
        await _context.SaveChangesAsync();
        return NoContent();
    }

    [HttpDelete("{id:int}")]
    public async Task<IActionResult> Delete(int id)
    {
        var position = await _context.OptionPositions.FindAsync(id);
        if (position == null) return NotFound();

        _context.OptionPositions.Remove(position);
        await _context.SaveChangesAsync();
        return NoContent();
    }

    private static OptionPositionDto ToDto(OptionPosition p, OptionPositionsLog? log)
    {
        var isOpen = p.Closed == null;
        var today = DateTime.UtcNow.Date;

        // Greeks from latest log (open positions only)
        decimal? lastPrice = isOpen ? log?.Price : null;
        decimal? margin = log?.Margin;

        // P/L calculations per SPEC 3.2
        decimal? unrealizedPnl = null;
        decimal? unrealizedPnlPct = null;
        decimal? realizedPnl = null;
        decimal? realizedPnlPct = null;

        if (isOpen && lastPrice.HasValue)
        {
            unrealizedPnl = p.Pos * (lastPrice.Value - p.Cost);
            unrealizedPnlPct = p.Cost != 0
                ? Math.Round(unrealizedPnl.Value / p.Cost, 2) * 100
                : null;
        }

        if (!isOpen && p.ClosePrice.HasValue)
        {
            realizedPnl = (p.ClosePrice.Value - p.Cost) * p.Multiplier * p.Pos - p.Commission;
            realizedPnlPct = (p.Cost * p.Multiplier) != 0
                ? Math.Round(realizedPnl.Value / (p.Cost * p.Multiplier), 1) * 100
                : null;
        }

        // Duration % = (closedOrToday - opened) / (expiry - opened) * 100
        var endDate = p.Closed ?? today;
        var totalSpan = (p.Expiry - p.Opened).TotalDays;
        var elapsedSpan = (endDate - p.Opened).TotalDays;
        decimal? durationPct = totalSpan > 0 ? (decimal)(elapsedSpan / totalSpan * 100) : null;

        // ROIC = realizedPnl * 100 / margin
        decimal? roic = (realizedPnl.HasValue && margin.HasValue && margin.Value != 0)
            ? realizedPnl.Value * 100 / margin.Value
            : null;

        return new OptionPositionDto
        {
            Id = p.Id,
            Symbol = p.Symbol,
            ContractId = p.ContractId,
            Opened = p.Opened,
            Expiry = p.Expiry,
            Closed = p.Closed,
            Pos = p.Pos,
            Right = p.Right,
            Strike = p.Strike,
            Cost = p.Cost,
            ClosePrice = p.ClosePrice,
            Commission = p.Commission,
            Multiplier = p.Multiplier,
            CloseReasons = p.CloseReasons,
            LastPrice = lastPrice,
            LastValue = lastPrice.HasValue ? lastPrice.Value * p.Multiplier : null,
            TimeValue = isOpen ? log?.TimeValue : null,
            Delta = isOpen ? (log?.Delta * p.Pos) : null,
            Theta = isOpen ? (log?.Theta * p.Pos) : null,
            Gamma = isOpen ? log?.Gamma : null,
            Vega = isOpen ? log?.Vega : null,
            Iv = isOpen ? (log?.Iv * 100) : null,
            Margin = margin,
            UnrealizedPnl = unrealizedPnl,
            UnrealizedPnlPct = unrealizedPnlPct,
            RealizedPnl = realizedPnl,
            RealizedPnlPct = realizedPnlPct,
            DurationPct = durationPct,
            Roic = roic,
        };
    }
}
