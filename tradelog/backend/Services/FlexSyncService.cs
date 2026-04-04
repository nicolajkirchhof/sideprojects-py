using System.Globalization;
using tradelog.Data;
using tradelog.Dtos;
using tradelog.Models;
using Microsoft.EntityFrameworkCore;

namespace tradelog.Services;

/// <summary>
/// Imports parsed Flex Query data into the database.
/// Handles trades (STK/FUT → Trades, OPT/FOP → OptionPositions) and capital history.
/// </summary>
public class FlexSyncService
{
    private readonly DataContext _context;
    private readonly ILogger<FlexSyncService> _logger;

    public FlexSyncService(DataContext context, ILogger<FlexSyncService> logger)
    {
        _context = context;
        _logger = logger;
    }

    /// <summary>
    /// Imports STK/FUT trades and manages OPT/FOP position lifecycle from Flex trade data.
    /// Returns (tradesCreated, tradesUpdated, optionsCreated, optionsClosed).
    /// </summary>
    public async Task<(int tradesCreated, int tradesUpdated, int optionsCreated, int optionsClosed)> SyncTradesAsync(
        List<FlexTradeDto> flexTrades)
    {
        var stockFutTrades = flexTrades.Where(t => t.AssetCategory is "STK" or "FUT").ToList();
        var optionTrades = flexTrades.Where(t => t.AssetCategory is "OPT" or "FOP").ToList();

        _logger.LogInformation("FlexSync: {StockCount} STK/FUT trades, {OptCount} OPT/FOP trades",
            stockFutTrades.Count, optionTrades.Count);

        var (created, updated) = await SyncStockFutureTradesAsync(stockFutTrades);
        var (optCreated, optClosed) = await SyncOptionTradesAsync(optionTrades);

        return (created, updated, optCreated, optClosed);
    }

    /// <summary>Upserts STK/FUT executions into the Trades table, deduplicating by TradeId → ExecutionId.</summary>
    private async Task<(int created, int updated)> SyncStockFutureTradesAsync(List<FlexTradeDto> trades)
    {
        if (trades.Count == 0) return (0, 0);

        var tradeIds = trades.Select(t => t.TradeId).ToList();
        var existingByExecId = await _context.Trades
            .Where(t => t.ExecutionId != null && tradeIds.Contains(t.ExecutionId))
            .ToDictionaryAsync(t => t.ExecutionId!);

        int created = 0, updated = 0;

        foreach (var flex in trades)
        {
            var posChange = flex.BuySell == "BUY"
                ? (int)flex.Quantity
                : -(int)Math.Abs(flex.Quantity);

            if (existingByExecId.TryGetValue(flex.TradeId, out var existing))
            {
                // Backfill fields that may be missing from old TWS sync
                bool changed = false;
                if (existing.Commission == 0 && flex.Commission != 0)
                    { existing.Commission = Math.Abs(flex.Commission); changed = true; }
                if (existing.FxRateToBase == 0 && flex.FxRateToBase != 0)
                    { existing.FxRateToBase = flex.FxRateToBase; changed = true; }
                if (existing.FifoPnlRealized == 0 && flex.FifoPnlRealized != 0)
                    { existing.FifoPnlRealized = flex.FifoPnlRealized; changed = true; }
                if (existing.Taxes == 0 && flex.Taxes != 0)
                    { existing.Taxes = flex.Taxes; changed = true; }
                if (existing.Exchange == null && flex.Exchange != null)
                    { existing.Exchange = flex.Exchange; changed = true; }
                if (changed) updated++;
                continue;
            }

            _context.Trades.Add(new Trade
            {
                Symbol = flex.Symbol,
                Date = flex.DateTime,
                PosChange = posChange,
                Price = flex.TradePrice,
                Commission = Math.Abs(flex.Commission),
                Multiplier = flex.Multiplier,
                ConId = flex.ConId,
                ExecutionId = flex.TradeId,
                Exchange = flex.Exchange,
                FxRateToBase = flex.FxRateToBase,
                FifoPnlRealized = flex.FifoPnlRealized,
                Taxes = flex.Taxes,
            });
            created++;
        }

        if (created > 0 || updated > 0)
            await _context.SaveChangesAsync();

        _logger.LogInformation("FlexSync STK/FUT: {Created} created, {Updated} updated", created, updated);
        return (created, updated);
    }

    /// <summary>
    /// Manages option position lifecycle from Flex trade executions.
    /// Groups trades by ConId, creates/updates OptionPosition records,
    /// and closes positions when net quantity reaches zero.
    /// </summary>
    private async Task<(int created, int closed)> SyncOptionTradesAsync(List<FlexTradeDto> trades)
    {
        if (trades.Count == 0) return (0, 0);

        // Group by ConId and sort chronologically within each group
        var groups = trades
            .GroupBy(t => t.ConId)
            .ToDictionary(g => g.Key, g => g.OrderBy(t => t.DateTime).ToList());

        // Load existing positions for these ConIds
        var conIds = groups.Keys.ToList();
        var existingPositions = await _context.OptionPositions
            .Where(p => p.ConId != null && conIds.Contains(p.ConId.Value))
            .ToDictionaryAsync(p => p.ConId!.Value);

        int created = 0, closed = 0;

        foreach (var (conId, conIdTrades) in groups)
        {
            if (existingPositions.TryGetValue(conId, out var pos))
            {
                // Position exists — recompute state from all trades (idempotent)
                bool wasClosed = pos.Closed != null;
                RecomputePositionFromTrades(pos, conIdTrades);
                bool nowClosed = pos.Closed != null;
                if (nowClosed && !wasClosed) closed++;
                else if (!nowClosed && wasClosed) closed--;
            }
            else
            {
                // New position — create from first trade, then apply remaining
                pos = CreatePositionFromTrade(conIdTrades[0]);
                _context.OptionPositions.Add(pos);
                created++;

                if (conIdTrades.Count > 1)
                    UpdatePositionFromTrades(pos, conIdTrades.Skip(1).ToList(), ref closed);
            }
        }

        await _context.SaveChangesAsync();
        _logger.LogInformation("FlexSync OPT/FOP: {Created} created, {Closed} closed", created, closed);
        return (created, closed);
    }

    private static readonly string[] ExpiryFormats = ["yyyy-MM-dd", "yyyyMMdd"];

    private static OptionPosition CreatePositionFromTrade(FlexTradeDto trade)
    {
        DateTime.TryParseExact(trade.Expiry, ExpiryFormats,
            CultureInfo.InvariantCulture, DateTimeStyles.None, out var expiry);

        return new OptionPosition
        {
            Symbol = trade.UnderlyingSymbol ?? trade.Symbol,
            ContractId = trade.ConId.ToString(),
            ConId = trade.ConId,
            SecType = trade.AssetCategory,
            Opened = trade.DateTime.Date,
            Expiry = expiry,
            Pos = (int)trade.Quantity,
            Right = trade.PutCall == "P" ? PositionRight.Put : PositionRight.Call,
            Strike = trade.Strike,
            Cost = trade.TradePrice,
            Commission = Math.Abs(trade.Commission),
            Multiplier = trade.Multiplier > 0 ? trade.Multiplier : 100,
            UnderlyingSymbol = trade.UnderlyingSymbol,
            UnderlyingConid = trade.UnderlyingConid,
        };
    }

    /// <summary>
    /// Applies subsequent trades to an existing position, updating quantity and
    /// detecting close (net quantity reaches zero).
    /// </summary>
    private void UpdatePositionFromTrades(OptionPosition pos, IList<FlexTradeDto> trades, ref int closedCount)
    {
        foreach (var trade in trades)
        {
            pos.Pos += (int)trade.Quantity;
            pos.Commission += Math.Abs(trade.Commission);

            if (pos.Pos == 0 && pos.Closed == null)
            {
                pos.Closed = trade.DateTime.Date;
                pos.ClosePrice = trade.TradePrice;
                closedCount++;
            }
            else if (pos.Pos != 0 && pos.Closed != null)
            {
                // Position was reopened by a subsequent trade — clear close state
                pos.Closed = null;
                pos.ClosePrice = null;
                closedCount--;
            }
        }
    }

    /// <summary>
    /// Recomputes position state from all Flex trades for a ConId.
    /// Idempotent: sets absolute values rather than incrementing.
    /// </summary>
    private static void RecomputePositionFromTrades(OptionPosition pos, IList<FlexTradeDto> allTrades)
    {
        int netPos = 0;
        decimal totalCommission = 0;
        DateTime? closedDate = null;
        decimal? closePrice = null;

        foreach (var trade in allTrades)
        {
            netPos += (int)trade.Quantity;
            totalCommission += Math.Abs(trade.Commission);

            if (netPos == 0 && closedDate == null)
            {
                closedDate = trade.DateTime.Date;
                closePrice = trade.TradePrice;
            }
            else if (netPos != 0)
            {
                // Position reopened — clear close state
                closedDate = null;
                closePrice = null;
            }
        }

        pos.Pos = netPos;
        pos.Commission = totalCommission;

        // Backfill underlying symbol if missing (fixes positions created before this change)
        var firstTrade = allTrades[0];
        if (firstTrade.UnderlyingSymbol != null)
        {
            pos.Symbol = firstTrade.UnderlyingSymbol;
            pos.UnderlyingSymbol ??= firstTrade.UnderlyingSymbol;
            pos.UnderlyingConid ??= firstTrade.UnderlyingConid;
        }

        // Only update close state from trades if trades actually close the position.
        // If net qty != 0, the position may have been closed by an event (assignment/expiration)
        // — preserve that state; SyncOptionEventsAsync handles it separately.
        if (netPos == 0)
        {
            pos.Closed = closedDate;
            pos.ClosePrice = closePrice;
        }
    }

    /// <summary>
    /// Imports daily equity snapshots from Flex data.
    /// Only inserts new dates — does not overwrite existing Capital records.
    /// </summary>
    public async Task<int> SyncCapitalAsync(List<FlexEquitySummaryDto> summaries)
    {
        if (summaries.Count == 0) return 0;

        var dates = summaries.Select(s => s.ReportDate).ToList();
        var existingDates = await _context.Capitals
            .Where(c => dates.Contains(c.Date))
            .Select(c => c.Date)
            .ToHashSetAsync();

        int created = 0;

        foreach (var summary in summaries)
        {
            if (existingDates.Contains(summary.ReportDate)) continue;
            if (summary.Total == 0) continue;

            _context.Capitals.Add(new Capital
            {
                Date = summary.ReportDate,
                NetLiquidity = summary.Total,
            });
            created++;
        }

        if (created > 0)
            await _context.SaveChangesAsync();

        _logger.LogInformation("FlexSync Capital: {Created} days created, {Skipped} already existed",
            created, summaries.Count - created);
        return created;
    }

    /// <summary>
    /// Processes option assignment, exercise, and expiration events from Flex OptionEAE.
    /// Closes matching OptionPosition records. Skips Buy/Sell entries (stock legs from assignments).
    /// </summary>
    public async Task<int> SyncOptionEventsAsync(List<FlexOptionEventDto> events)
    {
        // Only process option-level events, skip stock delivery legs
        var optionEvents = events
            .Where(e => e.AssetCategory is "OPT" or "FOP")
            .Where(e => e.TransactionType is "Assignment" or "Exercise" or "Expiration")
            .ToList();

        if (optionEvents.Count == 0) return 0;

        var conIds = optionEvents.Select(e => e.ConId).Distinct().ToList();
        var positions = await _context.OptionPositions
            .Where(p => p.ConId != null && conIds.Contains(p.ConId.Value) && p.Closed == null)
            .ToDictionaryAsync(p => p.ConId!.Value);

        int processed = 0;

        foreach (var evt in optionEvents)
        {
            if (!positions.TryGetValue(evt.ConId, out var pos)) continue;

            pos.Closed = evt.Date;
            pos.ClosePrice = evt.TradePrice;
            pos.Commission += Math.Abs(evt.Commission);

            var reason = evt.TransactionType switch
            {
                "Expiration" => CloseReasons.TimeLimit,
                "Assignment" or "Exercise" => CloseReasons.Other,
                _ => CloseReasons.Other,
            };
            pos.CloseReasons = (pos.CloseReasons ?? CloseReasons.None) | reason;

            processed++;
            _logger.LogInformation("OptionEAE: {Type} {Symbol} conId={ConId} on {Date}",
                evt.TransactionType, evt.Symbol, evt.ConId, evt.Date);
        }

        if (processed > 0)
            await _context.SaveChangesAsync();

        _logger.LogInformation("FlexSync OptionEAE: {Processed} events processed", processed);
        return processed;
    }

    /// <summary>
    /// Compares DB open positions against Flex open positions.
    /// Returns warnings for mismatches — does not auto-correct.
    /// </summary>
    public async Task<List<string>> ReconcilePositionsAsync(List<FlexPositionDto> flexPositions)
    {
        var warnings = new List<string>();

        var dbOpen = await _context.OptionPositions
            .Where(p => p.Closed == null)
            .ToListAsync();

        var flexConIds = flexPositions
            .Where(p => p.AssetCategory is "OPT" or "FOP")
            .GroupBy(p => p.ConId)
            .ToDictionary(g => g.Key, g => g.First());

        // Check DB positions against Flex
        foreach (var db in dbOpen)
        {
            if (db.ConId == null)
            {
                warnings.Add($"DB position {db.Symbol} (id={db.Id}) has no ConId — cannot reconcile");
                continue;
            }

            if (!flexConIds.TryGetValue(db.ConId.Value, out var flexPos))
            {
                warnings.Add($"DB position {db.Symbol} strike={db.Strike} conId={db.ConId} is OPEN in DB but NOT in Flex");
                continue;
            }

            if ((int)flexPos.Quantity != db.Pos)
            {
                warnings.Add($"Quantity mismatch for {db.Symbol} conId={db.ConId}: DB={db.Pos}, Flex={flexPos.Quantity}");
            }

            flexConIds.Remove(db.ConId.Value);
        }

        // Remaining Flex positions not in DB
        foreach (var (conId, flexPos) in flexConIds)
        {
            warnings.Add($"Flex position {flexPos.Symbol} strike={flexPos.Strike} conId={conId} is in Flex but NOT in DB");
        }

        foreach (var w in warnings)
            _logger.LogWarning("Reconciliation: {Warning}", w);

        return warnings;
    }
}
