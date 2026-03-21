using tradelog.Data;
using tradelog.Models;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

namespace tradelog.Controllers;

[ApiController]
[Route("api/weekly-prep")]
[Produces("application/json")]
public class WeeklyPrepController : ControllerBase
{
    private readonly DataContext _context;

    public WeeklyPrepController(DataContext context)
    {
        _context = context;
    }

    [HttpGet]
    public async Task<ActionResult<IEnumerable<WeeklyPrep>>> GetAll(
        [FromQuery] int? year)
    {
        var query = _context.WeeklyPreps.AsQueryable();

        if (year.HasValue)
            query = query.Where(e => e.Date.Year == year.Value);

        return await query.OrderByDescending(e => e.Date).ToListAsync();
    }

    [HttpGet("{id:int}")]
    public async Task<ActionResult<WeeklyPrep>> GetById(int id)
    {
        var entry = await _context.WeeklyPreps.FindAsync(id);
        if (entry == null) return NotFound();
        return entry;
    }

    [HttpPost]
    public async Task<ActionResult<WeeklyPrep>> Create(WeeklyPrep entry)
    {
        _context.WeeklyPreps.Add(entry);
        await _context.SaveChangesAsync();
        return CreatedAtAction(nameof(GetById), new { id = entry.Id }, entry);
    }

    [HttpPut("{id:int}")]
    public async Task<IActionResult> Update(int id, WeeklyPrep entry)
    {
        if (id != entry.Id) return BadRequest();

        var existing = await _context.WeeklyPreps.FindAsync(id);
        if (existing == null) return NotFound();

        _context.Entry(existing).CurrentValues.SetValues(entry);
        await _context.SaveChangesAsync();
        return NoContent();
    }

    [HttpDelete("{id:int}")]
    public async Task<IActionResult> Delete(int id)
    {
        var entry = await _context.WeeklyPreps.FindAsync(id);
        if (entry == null) return NotFound();

        _context.WeeklyPreps.Remove(entry);
        await _context.SaveChangesAsync();
        return NoContent();
    }
}
