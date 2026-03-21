using tradelog.Data;
using tradelog.Models;
using tradelog.Services;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

namespace tradelog.Controllers;

[ApiController]
[Route("api/capital")]
[Produces("application/json")]
public class CapitalController : ControllerBase
{
    private readonly DataContext _context;

    public CapitalController(DataContext context)
    {
        _context = context;
    }

    [HttpGet]
    public async Task<ActionResult<IEnumerable<Capital>>> GetAll()
    {
        return await _context.Capitals.OrderByDescending(c => c.Date).ToListAsync();
    }

    [HttpGet("{id:int}")]
    public async Task<ActionResult<Capital>> GetById(int id)
    {
        var capital = await _context.Capitals.FindAsync(id);
        if (capital == null) return NotFound();
        return capital;
    }

    [HttpPost]
    public async Task<ActionResult<Capital>> Create(Capital capital)
    {
        // Compute maintenancePct
        capital.MaintenancePct = capital.NetLiquidity != 0
            ? Math.Round(capital.Maintenance * 100 / capital.NetLiquidity, 2)
            : 0;

        // Snapshot portfolio aggregations from current positions
        var agg = await PortfolioAggregationService.ComputeAsync(_context);
        capital.TotalPnl = agg.TotalPnl;
        capital.UnrealizedPnl = agg.UnrealizedPnl;
        capital.RealizedPnl = agg.RealizedPnl;
        capital.NetDelta = agg.NetDelta;
        capital.NetTheta = agg.NetTheta;
        capital.NetVega = agg.NetVega;
        capital.NetGamma = agg.NetGamma;
        capital.AvgIv = agg.AvgIv;
        capital.TotalMargin = agg.TotalMargin;
        capital.TotalCommissions = agg.TotalCommissions;

        _context.Capitals.Add(capital);
        await _context.SaveChangesAsync();
        return CreatedAtAction(nameof(GetById), new { id = capital.Id }, capital);
    }

    [HttpPut("{id:int}")]
    public async Task<IActionResult> Update(int id, Capital capital)
    {
        if (id != capital.Id) return BadRequest();

        var existing = await _context.Capitals.FindAsync(id);
        if (existing == null) return NotFound();

        _context.Entry(existing).CurrentValues.SetValues(capital);

        // Recompute maintenancePct on update
        existing.MaintenancePct = existing.NetLiquidity != 0
            ? Math.Round(existing.Maintenance * 100 / existing.NetLiquidity, 2)
            : 0;

        await _context.SaveChangesAsync();
        return NoContent();
    }

    [HttpDelete("{id:int}")]
    public async Task<IActionResult> Delete(int id)
    {
        var capital = await _context.Capitals.FindAsync(id);
        if (capital == null) return NotFound();

        _context.Capitals.Remove(capital);
        await _context.SaveChangesAsync();
        return NoContent();
    }

}
