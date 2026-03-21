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
    public async Task<ActionResult<IEnumerable<TradeDto>>> GetAll([FromQuery] string? symbol)
    {
        var query = _context.Trades.AsQueryable();

        if (!string.IsNullOrWhiteSpace(symbol))
            query = query.Where(t => t.Symbol == symbol);

        var trades = await query.OrderBy(t => t.Symbol).ThenBy(t => t.Date).ThenBy(t => t.Id).ToListAsync();
        return TradeComputations.ComputeRunningFields(trades);
    }

    [HttpGet("{id:int}")]
    public async Task<ActionResult<TradeDto>> GetById(int id)
    {
        var trade = await _context.Trades.FindAsync(id);
        if (trade == null) return NotFound();

        var allForSymbol = await _context.Trades
            .Where(t => t.Symbol == trade.Symbol)
            .OrderBy(t => t.Date).ThenBy(t => t.Id)
            .ToListAsync();

        var dtos = TradeComputations.ComputeRunningFields(allForSymbol);
        var dto = dtos.FirstOrDefault(d => d.Id == id);
        if (dto == null) return NotFound();
        return dto;
    }

    [HttpPost]
    public async Task<ActionResult<TradeDto>> Create(Trade trade)
    {
        _context.Trades.Add(trade);
        await _context.SaveChangesAsync();

        var allForSymbol = await _context.Trades
            .Where(t => t.Symbol == trade.Symbol)
            .OrderBy(t => t.Date).ThenBy(t => t.Id)
            .ToListAsync();

        var dtos = TradeComputations.ComputeRunningFields(allForSymbol);
        var dto = dtos.First(d => d.Id == trade.Id);
        return CreatedAtAction(nameof(GetById), new { id = trade.Id }, dto);
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
