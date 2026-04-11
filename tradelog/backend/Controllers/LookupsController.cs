using tradelog.Data;
using tradelog.Models;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

namespace tradelog.Controllers;

[ApiController]
[Route("api/lookups")]
[Produces("application/json")]
public class LookupsController : ControllerBase
{
    private readonly DataContext _context;

    public LookupsController(DataContext context)
    {
        _context = context;
    }

    [HttpGet("{category}")]
    public async Task<ActionResult<IEnumerable<LookupValue>>> GetByCategory(string category)
    {
        if (!LookupCategory.All.Contains(category))
            return BadRequest($"Unknown category: {category}");

        return await _context.LookupValues
            .Where(lv => lv.Category == category)
            .OrderBy(lv => lv.SortOrder)
            .ToListAsync();
    }

    [HttpGet]
    public async Task<ActionResult<IEnumerable<LookupValue>>> GetAll()
    {
        return await _context.LookupValues
            .OrderBy(lv => lv.Category)
            .ThenBy(lv => lv.SortOrder)
            .ToListAsync();
    }

    [HttpPost("{category}")]
    public async Task<ActionResult<LookupValue>> Create(string category, [FromBody] LookupValueCreateDto dto)
    {
        if (!LookupCategory.All.Contains(category))
            return BadRequest($"Unknown category: {category}");

        var maxSort = await _context.LookupValues
            .Where(lv => lv.Category == category)
            .MaxAsync(lv => (int?)lv.SortOrder) ?? -1;

        var entry = new LookupValue
        {
            Category = category,
            Name = dto.Name.Trim(),
            SortOrder = maxSort + 1,
            IsActive = true,
        };

        _context.LookupValues.Add(entry);
        await _context.SaveChangesAsync();
        return CreatedAtAction(nameof(GetByCategory), new { category }, entry);
    }

    [HttpPut("{id:int}")]
    public async Task<IActionResult> Rename(int id, [FromBody] LookupValueRenameDto dto)
    {
        var entry = await _context.LookupValues.FindAsync(id);
        if (entry == null) return NotFound();

        entry.Name = dto.Name.Trim();
        await _context.SaveChangesAsync();
        return NoContent();
    }

    [HttpPatch("{id:int}/deactivate")]
    public async Task<IActionResult> Deactivate(int id)
    {
        var entry = await _context.LookupValues.FindAsync(id);
        if (entry == null) return NotFound();

        entry.IsActive = false;
        await _context.SaveChangesAsync();
        return NoContent();
    }

    [HttpPatch("{id:int}/reactivate")]
    public async Task<IActionResult> Reactivate(int id)
    {
        var entry = await _context.LookupValues.FindAsync(id);
        if (entry == null) return NotFound();

        entry.IsActive = true;
        await _context.SaveChangesAsync();
        return NoContent();
    }

    [HttpPatch("{id:int}/reorder")]
    public async Task<IActionResult> Reorder(int id, [FromBody] LookupValueReorderDto dto)
    {
        var entry = await _context.LookupValues.FindAsync(id);
        if (entry == null) return NotFound();

        entry.SortOrder = dto.SortOrder;
        await _context.SaveChangesAsync();
        return NoContent();
    }
}

public record LookupValueCreateDto(string Name);
public record LookupValueRenameDto(string Name);
public record LookupValueReorderDto(int SortOrder);
