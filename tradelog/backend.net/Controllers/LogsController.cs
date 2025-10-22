using backend.net.Data;
using backend.net.Models;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

namespace backend.net.Controllers;

[ApiController]
[Route("api/[controller]")]
[Produces("application/json")]
public class LogsController : ControllerBase
{
    private readonly DataContext _context;

    public LogsController(DataContext context)
    {
        _context = context;
    }

    // GET: /Logs
    [HttpGet]
    public async Task<ActionResult<IEnumerable<Log>>> GetLogs()
    {
        return await _context.Logs.ToListAsync();
    }

    // GET: /Logs/{id}
    [HttpGet("{id:int}")]
    public async Task<ActionResult<Log>> GetLog(int id)
    {
        var log = await _context.Logs.FindAsync(id);
        if (log == null)
            return NotFound();
        return log;
    }

    // POST: /Logs
    [HttpPost]
    [Consumes("application/json")]
    public async Task<ActionResult<Log>> CreateLog([FromBody] Log log)
    {
        if (!ModelState.IsValid)
            return ValidationProblem(ModelState);

        _context.Logs.Add(log);
        await _context.SaveChangesAsync();
        return CreatedAtAction(nameof(GetLog), new { id = log.Id }, log);
    }

    // PUT: /Logs/{id}
    [HttpPut("{id:int}")]
    [Consumes("application/json")]
    public async Task<IActionResult> UpdateLog(int id, [FromBody] Log log)
    {
        if (id != log.Id)
            return BadRequest("Id in route does not match request body");

        _context.Entry(log).State = EntityState.Modified;

        try
        {
            await _context.SaveChangesAsync();
        }
        catch (DbUpdateConcurrencyException)
        {
            var exists = await _context.Logs.AnyAsync(e => e.Id == id);
            if (!exists)
                return NotFound();
            throw;
        }

        return NoContent();
    }

    // DELETE: /Logs/{id}
    [HttpDelete("{id:int}")]
    public async Task<IActionResult> DeleteLog(int id)
    {
        var log = await _context.Logs.FindAsync(id);
        if (log == null)
            return NotFound();

        _context.Logs.Remove(log);
        await _context.SaveChangesAsync();
        return NoContent();
    }
}
