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
        [FromQuery] int? budget,
        [FromQuery] int? strategy)
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

    /// <summary>Resolves lookup value names for a set of IDs. Returns empty string for unknown IDs.</summary>
    private async Task<Dictionary<int, string>> ResolveLookupNames(params int?[] ids)
    {
        var nonNull = ids.Where(i => i.HasValue).Select(i => i!.Value).Distinct().ToList();
        if (nonNull.Count == 0) return new();
        return await _context.LookupValues
            .IgnoreQueryFilters()
            .Where(lv => nonNull.Contains(lv.Id))
            .ToDictionaryAsync(lv => lv.Id, lv => lv.Name);
    }

    [HttpGet("{id:int}")]
    public async Task<ActionResult<TradeDetailDto>> GetById(int id)
    {
        var trade = await _context.Trades.FindAsync(id);
        if (trade == null) return NotFound();

        var names = await ResolveLookupNames(
            trade.TypeOfTrade, trade.Directional, trade.Timeframe,
            trade.Budget, trade.Strategy, trade.ManagementRating);

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

        var events = await _context.TradeEvents
            .Where(e => e.TradeId == id)
            .OrderBy(e => e.Date)
            .Select(e => new TradeEventDto
            {
                Id = e.Id,
                TradeId = e.TradeId,
                Type = e.Type.ToString(),
                Date = e.Date,
                Notes = e.Notes,
                PnlImpact = e.PnlImpact,
            })
            .ToListAsync();

        return new TradeDetailDto
        {
            Id = trade.Id,
            Symbol = trade.Symbol,
            Date = trade.Date,
            TypeOfTrade = trade.TypeOfTrade,
            TypeOfTradeName = names.GetValueOrDefault(trade.TypeOfTrade, ""),
            Notes = trade.Notes,
            Directional = trade.Directional,
            DirectionalName = trade.Directional.HasValue ? names.GetValueOrDefault(trade.Directional.Value) : null,
            Timeframe = trade.Timeframe,
            TimeframeName = trade.Timeframe.HasValue ? names.GetValueOrDefault(trade.Timeframe.Value) : null,
            Budget = trade.Budget,
            BudgetName = names.GetValueOrDefault(trade.Budget, ""),
            Strategy = trade.Strategy,
            StrategyName = names.GetValueOrDefault(trade.Strategy, ""),
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
            ManagementRatingName = trade.ManagementRating.HasValue ? names.GetValueOrDefault(trade.ManagementRating.Value) : null,
            Learnings = trade.Learnings,
            ParentTradeId = trade.ParentTradeId,
            ChildTradeIds = await _context.Trades
                .Where(t => t.ParentTradeId == id)
                .Select(t => t.Id)
                .ToListAsync(),
            Events = events,
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

    /// <summary>
    /// Returns the full chain for a trade: walks up to the root, then returns
    /// all descendants in date order (root first).
    /// </summary>
    [HttpGet("{id:int}/chain")]
    public async Task<ActionResult<IEnumerable<Trade>>> GetChain(int id)
    {
        var trade = await _context.Trades.FindAsync(id);
        if (trade == null) return NotFound();

        // Walk up to root
        var root = trade;
        while (root.ParentTradeId.HasValue)
        {
            root = await _context.Trades.FindAsync(root.ParentTradeId.Value);
            if (root == null) return NotFound();
        }

        // Collect all descendants via BFS
        var chain = new List<Trade> { root };
        var queue = new Queue<int>();
        queue.Enqueue(root!.Id);

        while (queue.Count > 0)
        {
            var parentId = queue.Dequeue();
            var children = await _context.Trades
                .Where(t => t.ParentTradeId == parentId)
                .OrderBy(t => t.Date)
                .ToListAsync();
            foreach (var child in children)
            {
                chain.Add(child);
                queue.Enqueue(child.Id);
            }
        }

        return chain;
    }

    [HttpDelete("{id:int}")]
    public async Task<IActionResult> Delete(int id)
    {
        var trade = await _context.Trades.FindAsync(id);
        if (trade == null) return NotFound();

        if (await _context.Trades.AnyAsync(t => t.ParentTradeId == id))
            return BadRequest("Cannot delete a trade that has follow-up trades. Delete the children first.");

        _context.Trades.Remove(trade);
        await _context.SaveChangesAsync();
        return NoContent();
    }
}
