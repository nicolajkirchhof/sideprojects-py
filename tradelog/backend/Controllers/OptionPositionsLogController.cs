using tradelog.Data;
using tradelog.Models;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

namespace tradelog.Controllers;

[ApiController]
[Route("api/option-positions-log")]
[Produces("application/json")]
public class OptionPositionsLogController : ControllerBase
{
    private readonly DataContext _context;

    public OptionPositionsLogController(DataContext context)
    {
        _context = context;
    }

    [HttpGet]
    public async Task<ActionResult<IEnumerable<OptionPositionsLog>>> GetByContract(
        [FromQuery] string contractId)
    {
        if (string.IsNullOrWhiteSpace(contractId))
            return BadRequest("contractId query parameter is required");

        return await _context.OptionPositionsLogs
            .Where(l => l.ContractId == contractId)
            .OrderBy(l => l.DateTime)
            .ToListAsync();
    }

    [HttpGet("latest")]
    public async Task<ActionResult<IEnumerable<OptionPositionsLog>>> GetLatest()
    {
        // Get the most recent snapshot per contractId (for open positions dashboard)
        var latest = await _context.OptionPositionsLogs
            .GroupBy(l => l.ContractId)
            .Select(g => g.OrderByDescending(l => l.DateTime).First())
            .ToListAsync();

        return latest;
    }

    [HttpPost("bulk")]
    public async Task<ActionResult<object>> BulkInsert(List<OptionPositionsLog> entries)
    {
        return await InsertWithDedup(entries);
    }

    [HttpPost("sync")]
    public async Task<ActionResult<object>> Sync(List<OptionPositionsLog> entries)
    {
        return await InsertWithDedup(entries);
    }

    [HttpGet("last-sync")]
    public async Task<ActionResult<object>> GetLastSync()
    {
        var lastEntry = await _context.OptionPositionsLogs
            .OrderByDescending(l => l.DateTime)
            .FirstOrDefaultAsync();

        if (lastEntry == null)
            return Ok(new { lastSync = (DateTime?)null });

        return Ok(new { lastSync = lastEntry.DateTime });
    }

    private async Task<ActionResult<object>> InsertWithDedup(List<OptionPositionsLog> entries)
    {
        if (entries == null || entries.Count == 0)
            return BadRequest("No entries provided");

        // Get existing (contractId, dateTime) pairs to skip duplicates
        // IgnoreQueryFilters: the Python sync path doesn't send the X-Account-Id header
        var incoming = entries.Select(e => new { e.ContractId, e.DateTime }).ToList();
        var existingKeys = await _context.OptionPositionsLogs
            .IgnoreQueryFilters()
            .Where(l => incoming.Select(i => i.ContractId).Contains(l.ContractId))
            .Select(l => new { l.ContractId, l.DateTime })
            .ToListAsync();

        var existingSet = existingKeys
            .Select(k => $"{k.ContractId}|{k.DateTime:O}")
            .ToHashSet();

        var toInsert = entries
            .Where(e => !existingSet.Contains($"{e.ContractId}|{e.DateTime:O}"))
            .ToList();

        if (toInsert.Count > 0)
        {
            _context.OptionPositionsLogs.AddRange(toInsert);
            await _context.SaveChangesAsync();
        }

        return Ok(new { inserted = toInsert.Count, skipped = entries.Count - toInsert.Count });
    }
}
