using tradelog.Data;
using tradelog.Dtos;
using tradelog.Models;
using tradelog.Services;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

namespace tradelog.Controllers;

[ApiController]
[Route("api/instrument-summaries")]
[Produces("application/json")]
public class InstrumentSummariesController : ControllerBase
{
    private readonly DataContext _context;

    public InstrumentSummariesController(DataContext context)
    {
        _context = context;
    }

    private async Task<Dictionary<int, string>> GetLookupNames()
    {
        return await _context.LookupValues
            .IgnoreQueryFilters()
            .ToDictionaryAsync(lv => lv.Id, lv => lv.Name);
    }

    [HttpGet("options")]
    public async Task<ActionResult<IEnumerable<OptionInstrumentSummaryDto>>> GetOptionSummaries(
        [FromQuery] string? status)
    {
        var names = await GetLookupNames();
        var positions = await _context.OptionPositions
            .OrderBy(p => p.Symbol).ThenBy(p => p.Expiry)
            .ToListAsync();

        // Batch-fetch latest log per contractId
        var contractIds = positions.Select(p => p.ContractId).Distinct().ToList();
        var latestLogs = await _context.OptionPositionsLogs
            .Where(l => contractIds.Contains(l.ContractId))
            .GroupBy(l => l.ContractId)
            .Select(g => g.OrderByDescending(l => l.DateTime).First())
            .ToDictionaryAsync(l => l.ContractId);

        // Fetch latest Trade per symbol for metadata
        var symbols = positions.Select(p => p.Symbol).Distinct().ToList();
        var latestEntries = await GetLatestTrades(symbols);

        var today = DateTime.UtcNow.Date;
        var grouped = positions.GroupBy(p => p.Symbol);
        var result = new List<OptionInstrumentSummaryDto>();

        foreach (var group in grouped)
        {
            var symbol = group.Key;
            var all = group.ToList();
            var openPositions = all.Where(p => p.Closed == null).ToList();
            var isOpen = openPositions.Any();

            if (string.Equals(status, "open", StringComparison.OrdinalIgnoreCase) && !isOpen) continue;
            if (string.Equals(status, "closed", StringComparison.OrdinalIgnoreCase) && isOpen) continue;

            // Opened: earliest open date among open positions
            var opened = isOpen
                ? openPositions.Min(p => p.Opened)
                : all.Min(p => p.Opened);

            // Closed: latest close date only if ALL positions are closed
            DateTime? closed = isOpen ? null : all.Max(p => p.Closed);

            var dit = ((closed ?? today) - opened).Days;

            // DTE: min expiry - today among open positions
            int? dte = isOpen
                ? openPositions.Min(p => (p.Expiry - today).Days)
                : null;

            // Strikes: comma-joined from open positions
            var strikes = isOpen
                ? string.Join(", ", openPositions.Select(p => p.Strike).OrderBy(s => s))
                : null;

            // Compute per-position P/L and Greeks, then aggregate
            decimal sumUnrealizedPnlValue = 0, sumRealizedPnl = 0;
            decimal sumTimeValue = 0, sumDelta = 0, sumTheta = 0, sumGamma = 0, sumVega = 0, sumMargin = 0;
            decimal sumCommission = 0;
            var unrealizedPcts = new List<decimal>();
            var realizedPcts = new List<decimal>();
            var ivs = new List<decimal>();
            var durationPcts = new List<decimal>();
            var pnlPerDurPcts = new List<decimal>();

            foreach (var p in all)
            {
                var pIsOpen = p.Closed == null;
                var log = latestLogs.GetValueOrDefault(p.ContractId);
                sumCommission += p.Commission;

                if (pIsOpen && log != null)
                {
                    var unrealPnl = p.Pos * (log.Price - p.Cost);
                    var unrealPnlValue = unrealPnl * p.Multiplier;
                    sumUnrealizedPnlValue += unrealPnlValue;
                    if (p.Cost != 0)
                        unrealizedPcts.Add(Math.Round(unrealPnl / p.Cost, 2) * 100);

                    sumTimeValue += log.TimeValue;
                    sumDelta += log.Delta * p.Pos;
                    sumTheta += log.Theta * p.Pos;
                    sumGamma += log.Gamma;
                    sumVega += log.Vega;
                    ivs.Add(log.Iv * 100);
                }

                if (log != null)
                    sumMargin += log.Margin;

                if (!pIsOpen && p.ClosePrice.HasValue)
                {
                    var realPnl = (p.ClosePrice.Value - p.Cost) * p.Multiplier * p.Pos - p.Commission;
                    sumRealizedPnl += realPnl;
                    if (p.Cost * p.Multiplier != 0)
                        realizedPcts.Add(Math.Round(realPnl / (p.Cost * p.Multiplier), 1) * 100);
                }

                // Duration %
                var endDate = p.Closed ?? today;
                var totalSpan = (p.Expiry - p.Opened).TotalDays;
                var elapsed = (endDate - p.Opened).TotalDays;
                if (totalSpan > 0)
                {
                    var durPct = (decimal)(elapsed / totalSpan * 100);
                    durationPcts.Add(durPct);

                    // PnL%/Duration%
                    if (pIsOpen && p.Cost != 0 && log != null)
                    {
                        var unrealPnlPct = Math.Round(p.Pos * (log.Price - p.Cost) / p.Cost, 2) * 100;
                        if (unrealPnlPct != 0)
                            pnlPerDurPcts.Add(durPct / unrealPnlPct);
                    }
                }
            }

            var entry = latestEntries.GetValueOrDefault(symbol);

            // ROIC = unrealizedPnl / totalMargin * 100
            decimal? roic = sumMargin != 0 ? sumUnrealizedPnlValue * 100 / sumMargin : null;

            result.Add(new OptionInstrumentSummaryDto
            {
                Symbol = symbol,
                Opened = opened,
                Closed = closed,
                Dit = dit,
                Dte = dte,
                Status = isOpen ? "Open" : "Closed",
                Budget = entry != null ? names.GetValueOrDefault(entry.Budget) : null,
                CurrentSetup = entry != null ? names.GetValueOrDefault(entry.TypeOfTrade) : null,
                Strikes = strikes,
                IntendedManagement = entry?.IntendedManagement,
                Pnl = sumUnrealizedPnlValue + sumRealizedPnl,
                UnrealizedPnl = sumUnrealizedPnlValue,
                UnrealizedPnlPct = unrealizedPcts.Any() ? Math.Round(unrealizedPcts.Average(), 1) : null,
                RealizedPnl = sumRealizedPnl,
                RealizedPnlPct = realizedPcts.Any() ? Math.Round(realizedPcts.Average(), 1) : null,
                TimeValue = sumTimeValue,
                Delta = Math.Round(sumDelta, 4),
                Theta = Math.Round(sumTheta, 4),
                Gamma = Math.Round(sumGamma, 4),
                Vega = Math.Round(sumVega, 4),
                AvgIv = ivs.Any() ? Math.Round(ivs.Average(), 1) : null,
                Margin = sumMargin,
                DurationPct = durationPcts.Any() ? Math.Round(durationPcts.Average(), 1) : null,
                PnlPerDurationPct = pnlPerDurPcts.Any() ? Math.Round(pnlPerDurPcts.Average(), 1) : null,
                Roic = roic.HasValue ? Math.Round(roic.Value, 1) : null,
                Commissions = sumCommission,
            });
        }

        return result;
    }

    [HttpGet("trades")]
    public async Task<ActionResult<IEnumerable<TradeInstrumentSummaryDto>>> GetTradeSummaries()
    {
        var names = await GetLookupNames();
        var positions = await _context.StockPositions
            .OrderBy(t => t.Symbol).ThenBy(t => t.Date).ThenBy(t => t.Id)
            .ToListAsync();

        var dtos = StockPositionComputations.ComputeRunningFields(positions);

        var symbols = positions.Select(t => t.Symbol).Distinct().ToList();
        var latestEntries = await GetLatestTrades(symbols);

        // Fetch cached stock prices for unrealized P/L
        var priceCache = await _context.StockPriceCaches
            .Where(p => symbols.Contains(p.Symbol))
            .ToDictionaryAsync(p => p.Symbol);

        // Group by symbol: get final state
        var result = new List<TradeInstrumentSummaryDto>();

        foreach (var symbolGroup in dtos.GroupBy(d => d.Symbol))
        {
            var symbol = symbolGroup.Key;
            var ordered = symbolGroup.OrderBy(d => d.Date).ThenBy(d => d.Id).ToList();
            var last = ordered.Last();
            var totalRealizedPnl = ordered.Sum(d => d.Pnl);
            var totalCommissions = ordered.Sum(d => d.Commission);

            var entry = latestEntries.GetValueOrDefault(symbol);
            var cachedPrice = priceCache.GetValueOrDefault(symbol);

            // Unrealized P/L from cached last price
            decimal unrealizedPnl = 0;
            decimal? unrealizedPnlPct = null;
            if (last.TotalPos != 0 && cachedPrice != null)
            {
                unrealizedPnl = last.TotalPos * (cachedPrice.LastPrice - last.AvgPrice) * last.Multiplier;
                if (last.AvgPrice != 0)
                    unrealizedPnlPct = Math.Round(unrealizedPnl * 100 / (Math.Abs(last.TotalPos) * last.AvgPrice * last.Multiplier), 1);
            }

            result.Add(new TradeInstrumentSummaryDto
            {
                Symbol = symbol,
                Status = last.TotalPos != 0 ? "Open" : "Closed",
                Budget = entry != null ? names.GetValueOrDefault(entry.Budget) : null,
                PositionType = entry != null ? names.GetValueOrDefault(entry.TypeOfTrade) : null,
                IntendedManagement = entry?.IntendedManagement,
                TotalPos = last.TotalPos,
                AvgPrice = last.AvgPrice,
                Multiplier = last.Multiplier,
                Pnl = totalRealizedPnl + unrealizedPnl,
                UnrealizedPnl = unrealizedPnl,
                UnrealizedPnlPct = unrealizedPnlPct,
                RealizedPnl = totalRealizedPnl,
                Commissions = totalCommissions,
            });
        }

        return result;
    }

    /// <summary>
    /// Returns one row per Trade with aggregated P/L and Greeks from linked positions.
    /// Trades with no legs are included (with zero metrics).
    /// </summary>
    [HttpGet("trade-overview")]
    public async Task<ActionResult<IEnumerable<TradeSummaryDto>>> GetTradeOverview(
        [FromQuery] string? status)
    {
        var names = await GetLookupNames();
        var trades = await _context.Trades
            .OrderByDescending(t => t.Date)
            .ToListAsync();

        var tradeIds = trades.Select(t => t.Id).ToList();

        // Load all linked positions in batch
        var optionPositions = await _context.OptionPositions
            .Where(p => p.TradeId != null && tradeIds.Contains(p.TradeId.Value))
            .ToListAsync();

        var stockPositions = await _context.StockPositions
            .Where(p => p.TradeId != null && tradeIds.Contains(p.TradeId.Value))
            .ToListAsync();

        // Batch-fetch latest log per contractId for Greeks
        var contractIds = optionPositions.Select(p => p.ContractId).Distinct().ToList();
        var latestLogs = contractIds.Count > 0
            ? await _context.OptionPositionsLogs
                .Where(l => contractIds.Contains(l.ContractId))
                .GroupBy(l => l.ContractId)
                .Select(g => g.OrderByDescending(l => l.DateTime).First())
                .ToDictionaryAsync(l => l.ContractId)
            : new Dictionary<string, OptionPositionsLog>();

        var optionsByTrade = optionPositions.GroupBy(p => p.TradeId!.Value).ToDictionary(g => g.Key, g => g.ToList());
        var stocksByTrade = stockPositions.GroupBy(p => p.TradeId!.Value).ToDictionary(g => g.Key, g => g.ToList());

        var result = new List<TradeSummaryDto>();

        foreach (var trade in trades)
        {
            var opts = optionsByTrade.GetValueOrDefault(trade.Id) ?? [];
            var stks = stocksByTrade.GetValueOrDefault(trade.Id) ?? [];

            // Compute stock running fields first (needed for status + P/L)
            var stkDtos = StockPositionComputations.ComputeRunningFields(
                stks.OrderBy(s => s.Date).ThenBy(s => s.Id).ToList());

            // Determine status: open if any option leg is open or stock net position != 0
            var hasOpenOptions = opts.Any(p => p.Closed == null);
            var hasOpenStocks = stkDtos.Count > 0 && stkDtos.Last().TotalPos != 0;
            var isOpen = hasOpenOptions || hasOpenStocks;

            if (string.Equals(status, "open", StringComparison.OrdinalIgnoreCase) && !isOpen) continue;
            if (string.Equals(status, "closed", StringComparison.OrdinalIgnoreCase) && isOpen) continue;

            // Aggregate option P/L and Greeks
            decimal sumUnrealizedPnl = 0, sumRealizedPnl = 0, sumCommissions = 0;
            decimal sumDelta = 0, sumTheta = 0, sumGamma = 0, sumVega = 0, sumMargin = 0;

            foreach (var p in opts)
            {
                sumCommissions += p.Commission;
                var log = latestLogs.GetValueOrDefault(p.ContractId);

                if (p.Closed == null && log != null)
                {
                    sumUnrealizedPnl += p.Pos * (log.Price - p.Cost) * p.Multiplier;
                    sumDelta += log.Delta * p.Pos;
                    sumTheta += log.Theta * p.Pos;
                    sumGamma += log.Gamma;
                    sumVega += log.Vega;
                }
                if (log != null) sumMargin += log.Margin;

                if (p.Closed != null && p.ClosePrice.HasValue)
                    sumRealizedPnl += (p.ClosePrice.Value - p.Cost) * p.Multiplier * p.Pos - p.Commission;
            }

            // Aggregate stock P/L
            sumRealizedPnl += stkDtos.Sum(d => d.Pnl);
            sumCommissions += stkDtos.Sum(d => d.Commission);

            result.Add(new TradeSummaryDto
            {
                TradeId = trade.Id,
                Symbol = trade.Symbol,
                TypeOfTrade = names.GetValueOrDefault(trade.TypeOfTrade, ""),
                Strategy = names.GetValueOrDefault(trade.Strategy, ""),
                Budget = names.GetValueOrDefault(trade.Budget, ""),
                Status = isOpen ? "Open" : "Closed",
                Date = trade.Date,
                OptionLegCount = opts.Count,
                StockLegCount = stks.Count,
                Pnl = sumUnrealizedPnl + sumRealizedPnl,
                UnrealizedPnl = sumUnrealizedPnl,
                RealizedPnl = sumRealizedPnl,
                Commissions = sumCommissions,
                Delta = Math.Round(sumDelta, 4),
                Theta = Math.Round(sumTheta, 4),
                Gamma = Math.Round(sumGamma, 4),
                Vega = Math.Round(sumVega, 4),
                Margin = sumMargin,
            });
        }

        return result;
    }

    /// <summary>
    /// Get the most recent Trade per symbol (by date desc) for metadata lookups.
    /// </summary>
    private async Task<Dictionary<string, Trade>> GetLatestTrades(List<string> symbols)
    {
        return await _context.Trades
            .Where(e => symbols.Contains(e.Symbol))
            .GroupBy(e => e.Symbol)
            .Select(g => g.OrderByDescending(e => e.Date).First())
            .ToDictionaryAsync(e => e.Symbol);
    }
}
