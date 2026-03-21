using tradelog.Data;
using tradelog.Models;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

namespace tradelog.Controllers;

[ApiController]
[Route("api/ibkr")]
[Produces("application/json")]
public class IbkrController : ControllerBase
{
    private readonly DataContext _context;
    private static readonly TimeSpan SyncCooldown = TimeSpan.FromHours(1);

    public IbkrController(DataContext context)
    {
        _context = context;
    }

    [HttpGet("config")]
    public async Task<ActionResult<IbkrConfig>> GetConfig()
    {
        var config = await _context.IbkrConfigs.FirstOrDefaultAsync();
        if (config == null)
        {
            // Return defaults without persisting
            return new IbkrConfig { Host = "127.0.0.1", Port = 7497, ClientId = 1 };
        }
        return config;
    }

    [HttpPut("config")]
    public async Task<ActionResult<IbkrConfig>> UpdateConfig(IbkrConfig input)
    {
        var existing = await _context.IbkrConfigs.FirstOrDefaultAsync();
        if (existing == null)
        {
            existing = new IbkrConfig();
            _context.IbkrConfigs.Add(existing);
        }

        existing.Host = input.Host;
        existing.Port = input.Port;
        existing.ClientId = input.ClientId;

        await _context.SaveChangesAsync();
        return existing;
    }

    [HttpGet("sync/status")]
    public async Task<ActionResult<object>> GetSyncStatus()
    {
        var config = await _context.IbkrConfigs.FirstOrDefaultAsync();

        var lastSyncAt = config?.LastSyncAt;
        var lastSyncResult = config?.LastSyncResult;
        var now = DateTime.UtcNow;

        bool canSync;
        int? cooldownRemainingSeconds = null;

        if (lastSyncAt == null)
        {
            canSync = true;
        }
        else
        {
            var elapsed = now - lastSyncAt.Value;
            canSync = elapsed >= SyncCooldown;
            if (!canSync)
                cooldownRemainingSeconds = (int)(SyncCooldown - elapsed).TotalSeconds;
        }

        return Ok(new
        {
            lastSyncAt,
            lastSyncResult,
            canSync,
            cooldownRemainingSeconds,
        });
    }

    [HttpPost("sync")]
    public async Task<ActionResult<object>> TriggerSync()
    {
        var config = await _context.IbkrConfigs.FirstOrDefaultAsync();
        if (config == null)
            return BadRequest("IBKR connection not configured. Set host, port, and client ID first.");

        // Enforce cooldown
        if (config.LastSyncAt != null)
        {
            var elapsed = DateTime.UtcNow - config.LastSyncAt.Value;
            if (elapsed < SyncCooldown)
            {
                var remaining = (int)(SyncCooldown - elapsed).TotalSeconds;
                return Conflict(new { message = $"Next sync available in {remaining / 60} minutes.", cooldownRemainingSeconds = remaining });
            }
        }

        // Phase 2 will replace this with actual TWS sync
        return StatusCode(501, new { message = "IBKR sync not yet implemented. TWS API integration coming in Phase 2." });
    }
}
