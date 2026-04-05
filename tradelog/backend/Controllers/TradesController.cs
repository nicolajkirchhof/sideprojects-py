using tradelog.Data;
using tradelog.Dtos;
using tradelog.Models;
using tradelog.Services;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

namespace tradelog.Controllers;

[ApiController]
[Route("api/trades")]
[Produces("application/json")]
public class TradesController : ControllerBase
{
    private readonly DataContext _context;

    public TradesController(DataContext context)
    {
        _context = context;
    }

    [HttpGet]
    public async Task<ActionResult<IEnumerable<Trade>>> GetAll(
        [FromQuery] string? symbol,
        [FromQuery] Budget? budget,
        [FromQuery] Strategy? strategy)
    {
        var query = _context.Trades.AsQueryable();

        if (!string.IsNullOrWhiteSpace(symbol))
            query = query.Where(e => e.Symbol == symbol);
        if (budget.HasValue)
            query = query.Where(e => e.Budget == budget.Value);
        if (strategy.HasValue)
            query = query.Where(e => e.Strategy == strategy.Value);

        return await query.OrderByDescending(e => e.Date).ToListAsync();
    }

    [HttpGet("{id:int}")]
    public async Task<ActionResult<TradeDetailDto>> GetById(int id)
    {
        var trade = await _context.Trades.FindAsync(id);
        if (trade == null) return NotFound();

        var optionPositions = await _context.OptionPositions
            .Where(p => p.TradeId == id)
            .OrderBy(p => p.Symbol).ThenBy(p => p.Expiry)
            .ToListAsync();

        var contractIds = optionPositions.Select(p => p.ContractId).Distinct().ToList();
        var latestLogs = await _context.OptionPositionsLogs
            .Where(l => contractIds.Contains(l.ContractId))
            .GroupBy(l => l.ContractId)
            .Select(g => g.OrderByDescending(l => l.DateTime).First())
            .ToDictionaryAsync(l => l.ContractId);

        var stockPositions = await _context.StockPositions
            .Where(p => p.TradeId == id)
            .OrderBy(p => p.Symbol).ThenBy(p => p.Date).ThenBy(p => p.Id)
            .ToListAsync();

        return new TradeDetailDto
        {
            Id = trade.Id,
            Symbol = trade.Symbol,
            Date = trade.Date,
            TypeOfTrade = trade.TypeOfTrade,
            Notes = trade.Notes,
            Directional = trade.Directional,
            Timeframe = trade.Timeframe,
            Budget = trade.Budget,
            Strategy = trade.Strategy,
            NewsCatalyst = trade.NewsCatalyst,
            RecentEarnings = trade.RecentEarnings,
            SectorSupport = trade.SectorSupport,
            Ath = trade.Ath,
            Rvol = trade.Rvol,
            InstitutionalSupport = trade.InstitutionalSupport,
            GapPct = trade.GapPct,
            XAtrMove = trade.XAtrMove,
            TaFaNotes = trade.TaFaNotes,
            IntendedManagement = trade.IntendedManagement,
            ActualManagement = trade.ActualManagement,
            ManagementRating = trade.ManagementRating,
            Learnings = trade.Learnings,
            ParentTradeId = trade.ParentTradeId,
            OptionPositions = optionPositions
                .Select(p => OptionPositionDtoMapper.ToDto(p, latestLogs.GetValueOrDefault(p.ContractId)))
                .ToList(),
            StockPositions = StockPositionComputations.ComputeRunningFields(stockPositions),
        };
    }

    [HttpPost]
    public async Task<ActionResult<Trade>> Create(Trade trade)
    {
        _context.Trades.Add(trade);
        await _context.SaveChangesAsync();
        return CreatedAtAction(nameof(GetById), new { id = trade.Id }, trade);
    }

    [HttpPut("{id:int}")]
    public async Task<IActionResult> Update(int id, Trade trade)
    {
        if (id != trade.Id) return BadRequest();

        var existing = await _context.Trades.FindAsync(id);
        if (existing == null) return NotFound();

        _context.Entry(existing).CurrentValues.SetValues(trade);
        await _context.SaveChangesAsync();
        return NoContent();
    }

    [HttpDelete("{id:int}")]
    public async Task<IActionResult> Delete(int id)
    {
        var trade = await _context.Trades.FindAsync(id);
        if (trade == null) return NotFound();

        _context.Trades.Remove(trade);
        await _context.SaveChangesAsync();
        return NoContent();
    }
}
