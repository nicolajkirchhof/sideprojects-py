using tradelog.Data;
using tradelog.Dtos;
using tradelog.Models;
using tradelog.Services;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

namespace tradelog.Controllers;

[ApiController]
[Route("api/ibkr")]
[Produces("application/json")]
public class IbkrController : ControllerBase
{
    private readonly DataContext _context;
    private readonly IbkrSyncService _syncService;
    private readonly IAccountContext _accountContext;
    private static readonly TimeSpan SyncCooldown = TimeSpan.FromHours(1);

    public IbkrController(DataContext context, IbkrSyncService syncService, IAccountContext accountContext)
    {
        _context = context;
        _syncService = syncService;
        _accountContext = accountContext;
    }

    [HttpGet("sync/status")]
    public async Task<ActionResult<object>> GetSyncStatus()
    {
        var account = await GetCurrentAccount();
        if (account == null)
            return Ok(new { lastSyncAt = (DateTime?)null, lastSyncResult = (string?)null, canSync = false, cooldownRemainingSeconds = (int?)null });

        var now = DateTime.UtcNow;
        bool canSync;
        int? cooldownRemainingSeconds = null;

        if (account.LastSyncAt == null)
        {
            canSync = true;
        }
        else
        {
            var elapsed = now - account.LastSyncAt.Value;
            canSync = elapsed >= SyncCooldown;
            if (!canSync)
                cooldownRemainingSeconds = (int)(SyncCooldown - elapsed).TotalSeconds;
        }

        return Ok(new
        {
            lastSyncAt = account.LastSyncAt,
            lastSyncResult = account.LastSyncResult,
            canSync,
            cooldownRemainingSeconds,
        });
    }

    [HttpPost("sync")]
    public async Task<ActionResult<object>> TriggerSync()
    {
        var account = await GetCurrentAccount();
        if (account == null)
            return BadRequest("No account selected. Select an account first.");

        // Enforce cooldown
        if (account.LastSyncAt != null)
        {
            var elapsed = DateTime.UtcNow - account.LastSyncAt.Value;
            if (elapsed < SyncCooldown)
            {
                var remaining = (int)(SyncCooldown - elapsed).TotalSeconds;
                return Conflict(new { message = $"Next sync available in {remaining / 60} minutes.", cooldownRemainingSeconds = remaining });
            }
        }

        // Run sync
        var result = await _syncService.SyncAll(account);

        // Update sync metadata (only set LastSyncAt on success)
        if (result.Success)
            account.LastSyncAt = DateTime.UtcNow;
        account.LastSyncResult = result.Message;
        await _context.SaveChangesAsync();

        if (result.Success)
            return Ok(result);
        else
            return StatusCode(500, result);
    }

    private async Task<Account?> GetCurrentAccount()
    {
        var accountId = _accountContext.CurrentAccountId;
        if (accountId == 0) return null;
        return await _context.Accounts.IgnoreQueryFilters().FirstOrDefaultAsync(a => a.Id == accountId);
    }
}
