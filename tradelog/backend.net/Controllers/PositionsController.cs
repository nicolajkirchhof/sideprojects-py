using backend.net.Data;
using backend.net.Models;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

namespace backend.net.Controllers;

[ApiController]
[Route("api/[controller]")]
[Produces("application/json")]
public class PositionsController : ControllerBase
{
    private readonly DataContext _context;

    public PositionsController(DataContext context)
    {
        _context = context;
    }

    // GET: /Positions
    [HttpGet]
    public async Task<ActionResult<IEnumerable<Position>>> GetPositions()
    {
        return await _context.Positions.ToListAsync();
    }

    // GET: /Positions/{id}
    [HttpGet("{id:int}")]
    public async Task<ActionResult<Position>> GetPosition(int id)
    {
        var position = await _context.Positions.FindAsync(id);
        if (position == null)
            return NotFound();
        return position;
    }

    // POST: /Positions
    [HttpPost]
    [Consumes("application/json")]
    public async Task<ActionResult<Position>> CreatePosition([FromBody] Position position)
    {
        if (!ModelState.IsValid)
            return ValidationProblem(ModelState);

        _context.Positions.Add(position);
        await _context.SaveChangesAsync();
        return CreatedAtAction(nameof(GetPosition), new { id = position.Id }, position);
    }

    // PUT: /Positions/{id}
    [HttpPut("{id:int}")]
    [Consumes("application/json")]
    public async Task<IActionResult> UpdatePosition(int id, [FromBody] Position position)
    {
        if (id != position.Id)
            return BadRequest("Id in route does not match request body");

        _context.Entry(position).State = EntityState.Modified;

        try
        {
            await _context.SaveChangesAsync();
        }
        catch (DbUpdateConcurrencyException)
        {
            var exists = await _context.Positions.AnyAsync(e => e.Id == id);
            if (!exists)
                return NotFound();
            throw;
        }

        return NoContent();
    }

    // DELETE: /Positions/{id}
    [HttpDelete("{id:int}")]
    public async Task<IActionResult> DeletePosition(int id)
    {
        var position = await _context.Positions.FindAsync(id);
        if (position == null)
            return NotFound();

        _context.Positions.Remove(position);
        await _context.SaveChangesAsync();
        return NoContent();
    }
}
