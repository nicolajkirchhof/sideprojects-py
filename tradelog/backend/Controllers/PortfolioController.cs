using tradelog.Data;
using tradelog.Dtos;
using tradelog.Models;
using tradelog.Services;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

namespace tradelog.Controllers;

[ApiController]
[Route("api/portfolio")]
[Produces("application/json")]
public class PortfolioController : ControllerBase
{
    private readonly DataContext _context;

    public PortfolioController(DataContext context)
    {
        _context = context;
    }

    [HttpGet]
    public async Task<ActionResult<IEnumerable<PortfolioDto>>> GetAll()
    {
        var portfolios = await _context.Portfolios.ToListAsync();

        // Get latest netLiquidity for allocation %
        var latestCapital = await _context.Capitals
            .OrderByDescending(c => c.Date)
            .FirstOrDefaultAsync();
        var netLiquidity = latestCapital?.NetLiquidity ?? 0;

        // Compute current margin per budget from open option positions
        var openPositions = await _context.OptionPositions
            .Where(p => p.Closed == null)
            .ToListAsync();

        var contractIds = openPositions.Select(p => p.ContractId).Distinct().ToList();
        var latestLogs = await _context.OptionPositionsLogs
            .Where(l => contractIds.Contains(l.ContractId))
            .GroupBy(l => l.ContractId)
            .Select(g => g.OrderByDescending(l => l.DateTime).First())
            .ToDictionaryAsync(l => l.ContractId);

        // Get budget per symbol from TradeEntry
        var symbols = openPositions.Select(p => p.Symbol).Distinct().ToList();
        var latestEntries = await _context.TradeEntries
            .Where(e => symbols.Contains(e.Symbol))
            .GroupBy(e => e.Symbol)
            .Select(g => g.OrderByDescending(e => e.Date).First())
            .ToDictionaryAsync(e => e.Symbol);

        // Sum margin and P/L per budget
        var budgetMargin = new Dictionary<string, decimal>();
        var budgetPnl = new Dictionary<string, decimal>();

        foreach (var p in openPositions)
        {
            var entry = latestEntries.GetValueOrDefault(p.Symbol);
            var budget = entry?.Budget.ToString() ?? "Unknown";
            var log = latestLogs.GetValueOrDefault(p.ContractId);

            if (!budgetMargin.ContainsKey(budget))
            {
                budgetMargin[budget] = 0;
                budgetPnl[budget] = 0;
            }

            if (log != null)
            {
                budgetMargin[budget] += log.Margin;
                budgetPnl[budget] += p.Pos * (log.Price - p.Cost) * p.Multiplier;
            }
        }

        var result = portfolios.Select(pf =>
        {
            var budgetKey = pf.Budget.ToString();
            var margin = budgetMargin.GetValueOrDefault(budgetKey, 0);
            var currentAllocation = netLiquidity > 0
                ? Math.Round(margin * 100 / netLiquidity, 1)
                : 0;

            return new PortfolioDto
            {
                Id = pf.Id,
                Budget = budgetKey,
                Strategy = pf.Strategy.ToString(),
                MinAllocation = pf.MinAllocation,
                MaxAllocation = pf.MaxAllocation,
                CurrentAllocation = currentAllocation,
                Pnl = Math.Round(budgetPnl.GetValueOrDefault(budgetKey, 0), 2),
            };
        }).ToList();

        return result;
    }

    [HttpGet("{id:int}")]
    public async Task<ActionResult<Portfolio>> GetById(int id)
    {
        var entry = await _context.Portfolios.FindAsync(id);
        if (entry == null) return NotFound();
        return entry;
    }

    [HttpPut("{id:int}")]
    public async Task<IActionResult> Update(int id, Portfolio portfolio)
    {
        if (id != portfolio.Id) return BadRequest();

        var existing = await _context.Portfolios.FindAsync(id);
        if (existing == null) return NotFound();

        existing.MinAllocation = portfolio.MinAllocation;
        existing.MaxAllocation = portfolio.MaxAllocation;
        await _context.SaveChangesAsync();
        return NoContent();
    }

    [HttpPost]
    public async Task<ActionResult<Portfolio>> Create(Portfolio portfolio)
    {
        _context.Portfolios.Add(portfolio);
        await _context.SaveChangesAsync();
        return CreatedAtAction(nameof(GetById), new { id = portfolio.Id }, portfolio);
    }

    [HttpDelete("{id:int}")]
    public async Task<IActionResult> Delete(int id)
    {
        var entry = await _context.Portfolios.FindAsync(id);
        if (entry == null) return NotFound();

        _context.Portfolios.Remove(entry);
        await _context.SaveChangesAsync();
        return NoContent();
    }
}
