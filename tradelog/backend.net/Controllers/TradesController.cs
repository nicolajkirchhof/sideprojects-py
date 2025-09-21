using backend.net.Data;
using backend.net.Models;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

namespace backend.net.Controllers;

[ApiController]
[Route("[controller]")]
public class TradesController : ControllerBase
{
    private readonly DataContext _context;

    public TradesController(DataContext context)
    {
        _context = context;
    }

    [HttpGet]
    public async Task<ActionResult<IEnumerable<Trade>>> GetTrades()
    {
        return await _context.Trades.ToListAsync();
    }
}
