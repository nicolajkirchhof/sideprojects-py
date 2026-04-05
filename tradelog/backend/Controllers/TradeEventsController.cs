using tradelog.Data;
using tradelog.Dtos;
using tradelog.Models;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

namespace tradelog.Controllers;

[ApiController]
[Produces("application/json")]
public class TradeEventsController : ControllerBase
{
    private readonly DataContext _context;

    public TradeEventsController(DataContext context)
    {
        _context = context;
    }

    [HttpGet("api/trades/{tradeId:int}/events")]
    public async Task<ActionResult<IEnumerable<TradeEventDto>>> GetEvents(int tradeId)
    {
        var events = await _context.TradeEvents
            .Where(e => e.TradeId == tradeId)
            .OrderBy(e => e.Date)
            .ToListAsync();

        return events.Select(ToDto).ToList();
    }

    [HttpPost("api/trades/{tradeId:int}/events")]
    public async Task<ActionResult<TradeEventDto>> CreateEvent(int tradeId, [FromBody] TradeEventDto dto)
    {
        if (!await _context.Trades.AnyAsync(t => t.Id == tradeId))
            return NotFound();

        if (!Enum.TryParse<TradeEventType>(dto.Type, out var eventType))
            return BadRequest($"Invalid event type: {dto.Type}");

        var entity = new TradeEvent
        {
            TradeId = tradeId,
            Type = eventType,
            Date = dto.Date,
            Notes = dto.Notes,
            PnlImpact = dto.PnlImpact,
        };

        _context.TradeEvents.Add(entity);
        await _context.SaveChangesAsync();

        return CreatedAtAction(nameof(GetEvents), new { tradeId }, ToDto(entity));
    }

    [HttpDelete("api/trade-events/{id:int}")]
    public async Task<IActionResult> DeleteEvent(int id)
    {
        var entity = await _context.TradeEvents.FindAsync(id);
        if (entity == null) return NotFound();

        _context.TradeEvents.Remove(entity);
        await _context.SaveChangesAsync();
        return NoContent();
    }

    private static TradeEventDto ToDto(TradeEvent e) => new()
    {
        Id = e.Id,
        TradeId = e.TradeId,
        Type = e.Type.ToString(),
        Date = e.Date,
        Notes = e.Notes,
        PnlImpact = e.PnlImpact,
    };
}
