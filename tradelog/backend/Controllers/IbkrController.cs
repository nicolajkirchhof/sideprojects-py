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
    private readonly FlexQueryClient _flexClient;
    private readonly FlexReportParser _flexParser;
    private readonly FlexSyncService _flexSync;
    private readonly TwsLiveSyncService _liveSync;
    private readonly IAccountContext _accountContext;
    private readonly ILogger<IbkrController> _logger;
    private static readonly TimeSpan LiveSyncCooldown = TimeSpan.FromHours(1);

    public IbkrController(
        DataContext context,
        FlexQueryClient flexClient,
        FlexReportParser flexParser,
        FlexSyncService flexSync,
        TwsLiveSyncService liveSync,
        IAccountContext accountContext,
        ILogger<IbkrController> logger)
    {
        _context = context;
        _flexClient = flexClient;
        _flexParser = flexParser;
        _flexSync = flexSync;
        _liveSync = liveSync;
        _accountContext = accountContext;
        _logger = logger;
    }

    // ────────────────────────────────────────────
    // Status
    // ────────────────────────────────────────────

    [HttpGet("sync/status")]
    public async Task<ActionResult<object>> GetSyncStatus()
    {
        var account = await GetCurrentAccount();
        if (account == null)
            return Ok(new
            {
                flexConfigured = false,
                lastFlexSyncAt = (DateTime?)null,
                lastFlexSyncResult = (string?)null,
                lastLiveSyncAt = (DateTime?)null,
                lastLiveSyncResult = (string?)null,
                canLiveSync = false,
                liveSyncCooldownSeconds = (int?)null,
            });

        var now = DateTime.UtcNow;
        bool canLiveSync;
        int? cooldownSeconds = null;

        if (account.LastSyncAt == null)
        {
            canLiveSync = true;
        }
        else
        {
            var elapsed = now - account.LastSyncAt.Value;
            canLiveSync = elapsed >= LiveSyncCooldown;
            if (!canLiveSync)
                cooldownSeconds = (int)(LiveSyncCooldown - elapsed).TotalSeconds;
        }

        return Ok(new
        {
            flexConfigured = !string.IsNullOrEmpty(account.FlexToken) && !string.IsNullOrEmpty(account.FlexQueryId),
            lastFlexSyncAt = account.LastFlexSyncAt,
            lastFlexSyncResult = account.LastFlexSyncResult,
            lastLiveSyncAt = account.LastSyncAt,
            lastLiveSyncResult = account.LastSyncResult,
            canLiveSync,
            liveSyncCooldownSeconds = cooldownSeconds,
        });
    }

    // ────────────────────────────────────────────
    // Flex Sync
    // ────────────────────────────────────────────

    [HttpPost("flex-sync")]
    public async Task<ActionResult<FlexSyncResultDto>> TriggerFlexSync(CancellationToken ct)
    {
        var account = await GetCurrentAccount();
        if (account == null)
            return BadRequest("No account selected.");

        if (string.IsNullOrEmpty(account.FlexToken) || string.IsNullOrEmpty(account.FlexQueryId))
            return BadRequest("Flex credentials not configured. Set FlexToken and FlexQueryId in account settings.");

        var result = new FlexSyncResultDto();

        try
        {
            _logger.LogInformation("Starting Flex sync for account {AccountId}", account.IbkrAccountId);

            var xml = await _flexClient.FetchReportAsync(account.FlexToken, account.FlexQueryId, ct);
            _flexParser.ValidateSections(xml);

            var trades = _flexParser.ParseTrades(xml);
            var positions = _flexParser.ParseOpenPositions(xml);
            var equity = _flexParser.ParseEquitySummary(xml);
            var optionEvents = _flexParser.ParseOptionEvents(xml);

            var (tradesCreated, tradesUpdated, optCreated, optClosed) = await _flexSync.SyncTradesAsync(trades);
            result.TradesCreated = tradesCreated;
            result.TradesUpdated = tradesUpdated;
            result.OptionPositionsCreated = optCreated;
            result.OptionPositionsClosed = optClosed;

            result.OptionEventsProcessed = await _flexSync.SyncOptionEventsAsync(optionEvents);
            result.CapitalDaysCreated = await _flexSync.SyncCapitalAsync(equity);
            result.Warnings = await _flexSync.ReconcilePositionsAsync(positions);

            result.Success = true;
            result.Message = result.ToSummary();

            account.LastFlexSyncAt = DateTime.UtcNow;
            account.LastFlexSyncResult = result.Message;
            await _context.SaveChangesAsync();

            _logger.LogInformation("Flex sync complete: {Summary}", result.Message);
            return Ok(result);
        }
        catch (Exception ex)
        {
            result.Success = false;
            result.Message = $"Flex sync failed: {ex.Message}";

            account.LastFlexSyncResult = result.Message;
            await _context.SaveChangesAsync();

            _logger.LogError(ex, "Flex sync failed");
            return StatusCode(500, result);
        }
    }

    // ────────────────────────────────────────────
    // Live Sync (TWS — Greeks + Stock Prices)
    // ────────────────────────────────────────────

    [HttpPost("live-sync")]
    public async Task<ActionResult<LiveSyncResultDto>> TriggerLiveSync()
    {
        var account = await GetCurrentAccount();
        if (account == null)
            return BadRequest("No account selected.");

        // Greeks log upserts within 1h, so cooldown is no longer enforced here.
        var result = await _liveSync.SyncAll(account);

        if (result.Success)
            account.LastSyncAt = DateTime.UtcNow;
        account.LastSyncResult = result.Message;
        await _context.SaveChangesAsync();

        return result.Success ? Ok(result) : StatusCode(500, result);
    }

    // ────────────────────────────────────────────

    private async Task<Account?> GetCurrentAccount()
    {
        var accountId = _accountContext.CurrentAccountId;
        if (accountId == 0) return null;
        return await _context.Accounts.IgnoreQueryFilters().FirstOrDefaultAsync(a => a.Id == accountId);
    }
}
