using tradelog.Data;
using tradelog.Dtos;
using tradelog.Models;
using tradelog.Services;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

namespace tradelog.Controllers;

[ApiController]
[Route("api/option-positions")]
[Produces("application/json")]
public class OptionPositionsController : ControllerBase
{
    private readonly DataContext _context;
    private readonly OptionPositionLogCountService _logCountService;
    private readonly TradeStatusService _statusService;

    public OptionPositionsController(
        DataContext context,
        OptionPositionLogCountService logCountService,
        TradeStatusService statusService)
    {
        _context = context;
        _logCountService = logCountService;
        _statusService = statusService;
    }

    [HttpGet]
    public async Task<ActionResult<IEnumerable<OptionPositionDto>>> GetAll(
        [FromQuery] string? symbol,
        [FromQuery] string? status,
        [FromQuery] bool? unassigned)
    {
        var query = _context.OptionPositions.AsQueryable();

        if (!string.IsNullOrWhiteSpace(symbol))
            query = query.Where(p => p.Symbol == symbol);

        if (string.Equals(status, "open", StringComparison.OrdinalIgnoreCase))
            query = query.Where(p => p.Closed == null);
        else if (string.Equals(status, "closed", StringComparison.OrdinalIgnoreCase))
            query = query.Where(p => p.Closed != null);

        if (unassigned == true)
            query = query.Where(p => p.TradeId == null);

        var positions = await query.OrderBy(p => p.Symbol).ThenBy(p => p.Expiry).ToListAsync();
        var contractIds = positions.Select(p => p.ContractId).Distinct().ToList();

        var latestLogs = await _context.OptionPositionsLogs
            .Where(l => contractIds.Contains(l.ContractId))
            .GroupBy(l => l.ContractId)
            .Select(g => g.OrderByDescending(l => l.DateTime).First())
            .ToDictionaryAsync(l => l.ContractId);

        return positions
            .Select(p => OptionPositionDtoMapper.ToDto(p, latestLogs.GetValueOrDefault(p.ContractId)))
            .ToList();
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

        return OptionPositionDtoMapper.ToDto(position, latestLog);
    }

    [HttpPost]
    public async Task<ActionResult<OptionPositionDto>> Create(OptionPosition position)
    {
        _context.OptionPositions.Add(position);
        await _context.SaveChangesAsync();
        return CreatedAtAction(nameof(GetById), new { id = position.Id }, OptionPositionDtoMapper.ToDto(position, null));
    }

    [HttpPut("{id:int}")]
    public async Task<IActionResult> Update(int id, OptionPosition position)
    {
        if (id != position.Id) return BadRequest();

        var existing = await _context.OptionPositions.FindAsync(id);
        if (existing == null) return NotFound();

        var openedChanged = existing.Opened != position.Opened;
        var cachedLogCount = existing.LogCount;

        _context.Entry(existing).CurrentValues.SetValues(position);

        // LogCount is a server-maintained cache field — never trust the payload.
        // Preserve the pre-update value across SetValues, then recompute only if
        // the Opened date actually changed on a still-open position.
        existing.LogCount = cachedLogCount;

        if (openedChanged && existing.Closed == null)
        {
            await _logCountService.RecomputeForAsync(new[] { existing });
        }

        await _context.SaveChangesAsync();
        return NoContent();
    }

    [HttpPatch("{id:int}/assign")]
    public async Task<IActionResult> Assign(int id, [FromBody] AssignTradeDto dto)
    {
        var position = await _context.OptionPositions.FindAsync(id);
        if (position == null) return NotFound();

        if (dto.TradeId.HasValue && !await _context.Trades.AnyAsync(t => t.Id == dto.TradeId.Value))
            return BadRequest("Trade not found");

        var oldTradeId = position.TradeId;
        position.TradeId = dto.TradeId;
        await _context.SaveChangesAsync();

        // Recompute status for both old and new trade
        var affectedIds = new[] { oldTradeId, dto.TradeId }.Where(id => id.HasValue).Select(id => id!.Value);
        await _statusService.RecomputeForAsync(affectedIds);
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
}
