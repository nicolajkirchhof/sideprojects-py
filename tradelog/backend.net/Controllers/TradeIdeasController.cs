using backend.net.Data;
using backend.net.Models;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

namespace backend.net.Controllers;

[ApiController]
[Route("[controller]")]
public class TradeIdeasController : ControllerBase
{
    private readonly DataContext _context;

    public TradeIdeasController(DataContext context)
    {
        _context = context;
    }

    [HttpGet]
    public async Task<ActionResult<IEnumerable<TradeIdea>>> GetTradeIdeas()
    {
        return await _context.TradeIdeas.ToListAsync();
    }
}
