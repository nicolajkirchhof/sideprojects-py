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

    [HttpGet]
    public async Task<ActionResult<IEnumerable<Log>>> GetTradeIdeas()
    {
        return await _context.Logs.ToListAsync();
    }
}
