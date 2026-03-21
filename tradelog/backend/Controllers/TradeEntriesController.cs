using tradelog.Data;
using tradelog.Models;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

namespace tradelog.Controllers;

[ApiController]
[Route("api/trade-entries")]
[Produces("application/json")]
public class TradeEntriesController : ControllerBase
{
    private readonly DataContext _context;

    public TradeEntriesController(DataContext context)
    {
        _context = context;
    }

    [HttpGet]
    public async Task<ActionResult<IEnumerable<TradeEntry>>> GetAll(
        [FromQuery] string? symbol,
        [FromQuery] Budget? budget,
        [FromQuery] Strategy? strategy)
    {
        var query = _context.TradeEntries.AsQueryable();

        if (!string.IsNullOrWhiteSpace(symbol))
            query = query.Where(e => e.Symbol == symbol);
        if (budget.HasValue)
            query = query.Where(e => e.Budget == budget.Value);
        if (strategy.HasValue)
            query = query.Where(e => e.Strategy == strategy.Value);

        return await query.OrderByDescending(e => e.Date).ToListAsync();
    }

    [HttpGet("{id:int}")]
    public async Task<ActionResult<TradeEntry>> GetById(int id)
    {
        var entry = await _context.TradeEntries.FindAsync(id);
        if (entry == null) return NotFound();
        return entry;
    }

    [HttpPost]
    public async Task<ActionResult<TradeEntry>> Create(TradeEntry entry)
    {
        _context.TradeEntries.Add(entry);
        await _context.SaveChangesAsync();
        return CreatedAtAction(nameof(GetById), new { id = entry.Id }, entry);
    }

    [HttpPut("{id:int}")]
    public async Task<IActionResult> Update(int id, TradeEntry entry)
    {
        if (id != entry.Id) return BadRequest();

        var existing = await _context.TradeEntries.FindAsync(id);
        if (existing == null) return NotFound();

        _context.Entry(existing).CurrentValues.SetValues(entry);
        await _context.SaveChangesAsync();
        return NoContent();
    }

    [HttpDelete("{id:int}")]
    public async Task<IActionResult> Delete(int id)
    {
        var entry = await _context.TradeEntries.FindAsync(id);
        if (entry == null) return NotFound();

        _context.TradeEntries.Remove(entry);
        await _context.SaveChangesAsync();
        return NoContent();
    }
}
