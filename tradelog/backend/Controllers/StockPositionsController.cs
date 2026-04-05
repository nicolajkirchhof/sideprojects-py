using tradelog.Data;
using tradelog.Dtos;
using tradelog.Models;
using tradelog.Services;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

namespace tradelog.Controllers;

[ApiController]
[Route("api/stock-positions")]
[Produces("application/json")]
public class StockPositionsController : ControllerBase
{
    private readonly DataContext _context;

    public StockPositionsController(DataContext context)
    {
        _context = context;
    }

    [HttpGet]
    public async Task<ActionResult<IEnumerable<StockPositionDto>>> GetAll(
        [FromQuery] string? symbol,
        [FromQuery] bool? unassigned)
    {
        var query = _context.StockPositions.AsQueryable();

        if (!string.IsNullOrWhiteSpace(symbol))
            query = query.Where(t => t.Symbol == symbol);

        if (unassigned == true)
            query = query.Where(t => t.TradeId == null);

        var positions = await query.OrderBy(t => t.Symbol).ThenBy(t => t.Date).ThenBy(t => t.Id).ToListAsync();
        return StockPositionComputations.ComputeRunningFields(positions);
    }

    [HttpGet("{id:int}")]
    public async Task<ActionResult<StockPositionDto>> GetById(int id)
    {
        var position = await _context.StockPositions.FindAsync(id);
        if (position == null) return NotFound();

        var allForSymbol = await _context.StockPositions
            .Where(t => t.Symbol == position.Symbol)
            .OrderBy(t => t.Date).ThenBy(t => t.Id)
            .ToListAsync();

        var dtos = StockPositionComputations.ComputeRunningFields(allForSymbol);
        var dto = dtos.FirstOrDefault(d => d.Id == id);
        if (dto == null) return NotFound();
        return dto;
    }

    [HttpPost]
    public async Task<ActionResult<StockPositionDto>> Create(StockPosition position)
    {
        _context.StockPositions.Add(position);
        await _context.SaveChangesAsync();

        var allForSymbol = await _context.StockPositions
            .Where(t => t.Symbol == position.Symbol)
            .OrderBy(t => t.Date).ThenBy(t => t.Id)
            .ToListAsync();

        var dtos = StockPositionComputations.ComputeRunningFields(allForSymbol);
        var dto = dtos.First(d => d.Id == position.Id);
        return CreatedAtAction(nameof(GetById), new { id = position.Id }, dto);
    }

    [HttpPut("{id:int}")]
    public async Task<IActionResult> Update(int id, StockPosition position)
    {
        if (id != position.Id) return BadRequest();

        var existing = await _context.StockPositions.FindAsync(id);
        if (existing == null) return NotFound();

        _context.Entry(existing).CurrentValues.SetValues(position);
        await _context.SaveChangesAsync();
        return NoContent();
    }

    [HttpPatch("{id:int}/assign")]
    public async Task<IActionResult> Assign(int id, [FromBody] AssignTradeDto dto)
    {
        var position = await _context.StockPositions.FindAsync(id);
        if (position == null) return NotFound();

        if (dto.TradeId.HasValue && !await _context.Trades.AnyAsync(t => t.Id == dto.TradeId.Value))
            return BadRequest("Trade not found");

        position.TradeId = dto.TradeId;
        await _context.SaveChangesAsync();
        return NoContent();
    }

    [HttpDelete("{id:int}")]
    public async Task<IActionResult> Delete(int id)
    {
        var position = await _context.StockPositions.FindAsync(id);
        if (position == null) return NotFound();

        _context.StockPositions.Remove(position);
        await _context.SaveChangesAsync();
        return NoContent();
    }
}
