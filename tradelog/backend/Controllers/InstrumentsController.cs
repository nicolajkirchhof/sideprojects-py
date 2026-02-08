using backend.net.Data;
using backend.net.Models;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

namespace backend.net.Controllers;

[ApiController]
[Route("api/[controller]")]
[Produces("application/json")]
public class InstrumentsController : ControllerBase
{
    private readonly DataContext _context;

    public InstrumentsController(DataContext context)
    {
        _context = context;
    }

    // GET: /Instruments
    [HttpGet]
    public async Task<ActionResult<IEnumerable<Instrument>>> GetInstruments()
    {
        return await _context.Instruments.ToListAsync();
    }

    // GET: /Instruments/{id}
    [HttpGet("{id:int}")]
    public async Task<ActionResult<Instrument>> GetInstrument(int id)
    {
        var instrument = await _context.Instruments.FindAsync(id);
        if (instrument == null)
            return NotFound();
        return instrument;
    }

    // POST: /Instruments
    [HttpPost]
    [Consumes("application/json")]
    public async Task<ActionResult<Instrument>> CreateInstrument([FromBody] Instrument instrument)
    {
        if (!ModelState.IsValid)
            return ValidationProblem(ModelState);

        _context.Instruments.Add(instrument);
        await _context.SaveChangesAsync();
        return CreatedAtAction(nameof(GetInstrument), new { id = instrument.Id }, instrument);
    }

    // PUT: /Instruments/{id}
    [HttpPut("{id:int}")]
    [Consumes("application/json")]
    public async Task<IActionResult> UpdateInstrument(int id, [FromBody] Instrument instrument)
    {
        if (id != instrument.Id)
            return BadRequest("Id in route does not match request body");

        // Track entity as modified
        _context.Entry(instrument).State = EntityState.Modified;

        try
        {
            await _context.SaveChangesAsync();
        }
        catch (DbUpdateConcurrencyException)
        {
            var exists = await _context.Instruments.AnyAsync(e => e.Id == id);
            if (!exists)
                return NotFound();
            throw;
        }

        return NoContent();
    }

    // DELETE: /Instruments/{id}
    [HttpDelete("{id:int}")]
    public async Task<IActionResult> DeleteInstrument(int id)
    {
        var instrument = await _context.Instruments.FindAsync(id);
        if (instrument == null)
            return NotFound();

        _context.Instruments.Remove(instrument);
        await _context.SaveChangesAsync();
        return NoContent();
    }
}
