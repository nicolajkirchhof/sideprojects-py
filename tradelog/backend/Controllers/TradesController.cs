using tradelog.Data;
using tradelog.Models;
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
    public async Task<ActionResult<Trade>> GetById(int id)
    {
        var trade = await _context.Trades.FindAsync(id);
        if (trade == null) return NotFound();
        return trade;
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
