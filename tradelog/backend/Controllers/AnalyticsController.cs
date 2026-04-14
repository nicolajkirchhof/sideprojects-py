using tradelog.Data;
using tradelog.Dtos;
using tradelog.Models;
using tradelog.Services;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

namespace tradelog.Controllers;

[ApiController]
[Route("api/analytics")]
[Produces("application/json")]
public class AnalyticsController : ControllerBase
{
    private readonly DataContext _context;

    public AnalyticsController(DataContext context)
    {
        _context = context;
    }

    private async Task<Dictionary<int, string>> GetLookupNames()
    {
        return await _context.LookupValues
            .IgnoreQueryFilters()
            .ToDictionaryAsync(lv => lv.Id, lv => lv.Name);
    }

    [HttpGet("strategies")]
    public async Task<ActionResult<IEnumerable<StrategyPerformanceDto>>> GetStrategyPerformance()
    {
        var (pnlsBySymbol, trades) = await LoadClosedData();
        var names = await GetLookupNames();

        var strategyPnls = new Dictionary<string, List<SymbolPnl>>();
        var strategyCommissions = new Dictionary<string, decimal>();

        foreach (var g in pnlsBySymbol.GroupBy(p => p.Symbol))
        {
            var entry = trades.GetValueOrDefault(g.Key);
            var strategy = entry != null ? names.GetValueOrDefault(entry.Strategy, "Unknown") : "Unknown";
            if (!strategyPnls.ContainsKey(strategy))
            {
                strategyPnls[strategy] = new List<SymbolPnl>();
                strategyCommissions[strategy] = 0;
            }
            strategyPnls[strategy].AddRange(g);
            strategyCommissions[strategy] += g.Sum(x => x.Commission);
        }

        var result = new List<StrategyPerformanceDto>();
        foreach (var (strategy, pnls) in strategyPnls)
        {
            var (avgWin, avgLoss, winRate, expectancy, maxDd) = AnalyticsComputations.ComputeMetrics(pnls);

            result.Add(new StrategyPerformanceDto
            {
                Strategy = strategy,
                TradeCount = pnls.Count,
                TotalPnl = Math.Round(pnls.Sum(p => p.Pnl), 2),
                AvgWin = avgWin,
                AvgLoss = avgLoss,
                WinRate = winRate,
                Expectancy = expectancy,
                MaxDrawdown = maxDd,
                TotalCommissions = strategyCommissions[strategy],
            });
        }

        return result.OrderByDescending(r => r.TotalPnl).ToList();
    }

    [HttpGet("strategies/{strategy}/equity-curve")]
    public async Task<ActionResult<IEnumerable<EquityCurvePointDto>>> GetStrategyEquityCurve(string strategy)
    {
        var (pnlsBySymbol, trades) = await LoadClosedData();
        var names = await GetLookupNames();

        var filtered = pnlsBySymbol.Where(p =>
        {
            var entry = trades.GetValueOrDefault(p.Symbol);
            return entry != null && names.GetValueOrDefault(entry.Strategy) == strategy;
        }).ToList();

        return AnalyticsComputations.BuildEquityCurve(filtered);
    }

    [HttpGet("budgets")]
    public async Task<ActionResult<IEnumerable<BudgetPerformanceDto>>> GetBudgetPerformance()
    {
        var (pnlsBySymbol, trades) = await LoadClosedData();
        var names = await GetLookupNames();

        var budgetPnls = new Dictionary<string, List<SymbolPnl>>();
        var budgetCommissions = new Dictionary<string, decimal>();

        foreach (var g in pnlsBySymbol.GroupBy(p => p.Symbol))
        {
            var entry = trades.GetValueOrDefault(g.Key);
            var budget = entry != null ? names.GetValueOrDefault(entry.Budget, "Unknown") : "Unknown";
            if (!budgetPnls.ContainsKey(budget))
            {
                budgetPnls[budget] = new List<SymbolPnl>();
                budgetCommissions[budget] = 0;
            }
            budgetPnls[budget].AddRange(g);
            budgetCommissions[budget] += g.Sum(x => x.Commission);
        }

        var result = new List<BudgetPerformanceDto>();
        foreach (var (budget, pnls) in budgetPnls)
        {
            var (avgWin, avgLoss, winRate, expectancy, _) = AnalyticsComputations.ComputeMetrics(pnls);

            result.Add(new BudgetPerformanceDto
            {
                Budget = budget,
                TradeCount = pnls.Count,
                TotalPnl = Math.Round(pnls.Sum(p => p.Pnl), 2),
                AvgWin = avgWin,
                AvgLoss = avgLoss,
                WinRate = winRate,
                Expectancy = expectancy,
                TotalCommissions = budgetCommissions[budget],
            });
        }

        return result;
    }

    [HttpGet("budgets/{budget}/equity-curve")]
    public async Task<ActionResult<IEnumerable<EquityCurvePointDto>>> GetBudgetEquityCurve(string budget)
    {
        var (pnlsBySymbol, trades) = await LoadClosedData();
        var names = await GetLookupNames();

        var filtered = pnlsBySymbol.Where(p =>
        {
            var entry = trades.GetValueOrDefault(p.Symbol);
            return entry != null && names.GetValueOrDefault(entry.Budget) == budget;
        }).ToList();

        return AnalyticsComputations.BuildEquityCurve(filtered);
    }

    [HttpGet("overall")]
    public async Task<ActionResult<OverallPerformanceDto>> GetOverallPerformance()
    {
        var (pnlsBySymbol, _) = await LoadClosedData();

        var totalPnl = Math.Round(pnlsBySymbol.Sum(p => p.Pnl), 2);
        var totalCommissions = Math.Round(pnlsBySymbol.Sum(p => p.Commission), 2);
        var netPnl = totalPnl - totalCommissions;

        var tradingDays = pnlsBySymbol.Select(p => p.Date.Date).Distinct().Count();
        var dailyPnl = tradingDays > 0 ? Math.Round(totalPnl / tradingDays, 2) : 0;

        // accountSize from most recent Capital snapshot
        var latestCapital = await _context.Capitals
            .OrderByDescending(c => c.Date)
            .FirstOrDefaultAsync();
        var accountSize = latestCapital?.NetLiquidity ?? 0;

        var annualizedRoi = accountSize > 0
            ? Math.Round(365 * dailyPnl * 100 / accountSize, 2)
            : 0;

        var (avgWin, avgLoss, winRate, _, _) = AnalyticsComputations.ComputeMetrics(pnlsBySymbol);

        return new OverallPerformanceDto
        {
            TotalPnl = totalPnl,
            TotalCommissions = totalCommissions,
            NetPnl = netPnl,
            DailyPnl = dailyPnl,
            AnnualizedRoi = annualizedRoi,
            TradingDays = tradingDays,
            TradeCount = pnlsBySymbol.Count,
            WinRate = winRate,
            AvgWin = avgWin,
            AvgLoss = avgLoss,
        };
    }

    [HttpGet("overall/equity-curve")]
    public async Task<ActionResult<IEnumerable<EquityCurvePointDto>>> GetOverallEquityCurve()
    {
        var (pnlsBySymbol, _) = await LoadClosedData();
        return AnalyticsComputations.BuildEquityCurve(pnlsBySymbol);
    }

    /// <summary>
    /// Load closed option positions + all stock positions, compute per-symbol P/L, and fetch Trade metadata.
    /// </summary>
    private async Task<(List<SymbolPnl> pnls, Dictionary<string, Trade> trades)> LoadClosedData()
    {
        var closedOptions = await _context.OptionPositions
            .Where(p => p.Closed != null && p.ClosePrice != null)
            .ToListAsync();

        var positions = await _context.StockPositions
            .OrderBy(t => t.Symbol).ThenBy(t => t.Date).ThenBy(t => t.Id)
            .ToListAsync();

        var pnls = AnalyticsComputations.ComputeClosedPnls(closedOptions, positions);

        var symbols = pnls.Select(p => p.Symbol).Distinct().ToList();
        var trades = await _context.Trades
            .Where(e => symbols.Contains(e.Symbol))
            .GroupBy(e => e.Symbol)
            .Select(g => g.OrderByDescending(e => e.Date).First())
            .ToDictionaryAsync(e => e.Symbol);

        return (pnls, trades);
    }

    /// <summary>
    /// Returns one row per trade chain (length >= 2) with aggregated P/L.
    /// Walks from root trades down through all descendants via ParentTradeId.
    /// </summary>
    [HttpGet("chains")]
    public async Task<ActionResult<IEnumerable<ChainSummaryDto>>> GetChains()
    {
        var names = await GetLookupNames();
        var allTrades = await _context.Trades.OrderBy(t => t.Date).ToListAsync();
        var childrenByParent = allTrades
            .Where(t => t.ParentTradeId.HasValue)
            .GroupBy(t => t.ParentTradeId!.Value)
            .ToDictionary(g => g.Key, g => g.ToList());

        // Root trades = no parent
        var roots = allTrades.Where(t => t.ParentTradeId == null).ToList();

        // Build chains via BFS from each root
        var chains = new List<(Trade Root, List<Trade> All)>();
        foreach (var root in roots)
        {
            var chain = new List<Trade> { root };
            var queue = new Queue<int>();
            queue.Enqueue(root.Id);
            while (queue.Count > 0)
            {
                var parentId = queue.Dequeue();
                if (childrenByParent.TryGetValue(parentId, out var children))
                {
                    chain.AddRange(children);
                    foreach (var child in children) queue.Enqueue(child.Id);
                }
            }
            if (chain.Count >= 2) chains.Add((root, chain));
        }

        if (chains.Count == 0) return new List<ChainSummaryDto>();

        // Batch-load all linked positions and events for chain trades
        var chainTradeIds = chains.SelectMany(c => c.All.Select(t => t.Id)).ToHashSet();

        var optionPositions = await _context.OptionPositions
            .Where(p => p.TradeId != null && chainTradeIds.Contains(p.TradeId.Value))
            .ToListAsync();
        var optionsByTrade = optionPositions
            .GroupBy(p => p.TradeId!.Value)
            .ToDictionary(g => g.Key, g => g.ToList());

        var stockPositions = await _context.StockPositions
            .Where(p => p.TradeId != null && chainTradeIds.Contains(p.TradeId.Value))
            .ToListAsync();
        var stocksByTrade = stockPositions
            .GroupBy(p => p.TradeId!.Value)
            .ToDictionary(g => g.Key, g => g.ToList());

        var result = new List<ChainSummaryDto>();

        foreach (var (root, chainTrades) in chains)
        {
            decimal totalPnl = 0, premiumCollected = 0, premiumLost = 0;

            var hasOpen = false;

            foreach (var trade in chainTrades)
            {


                // Option P/L
                var opts = optionsByTrade.GetValueOrDefault(trade.Id) ?? [];
                foreach (var p in opts)
                {
                    if (p.Closed == null) hasOpen = true;
                    if (p.Closed != null && p.ClosePrice.HasValue)
                    {
                        var pnl = (p.ClosePrice.Value - p.Cost) * p.Multiplier * p.Pos - p.Commission;
                        totalPnl += pnl;
                        if (pnl >= 0) premiumCollected += pnl;
                        else premiumLost += Math.Abs(pnl);
                    }
                }

                // Stock P/L
                var stks = stocksByTrade.GetValueOrDefault(trade.Id) ?? [];
                if (stks.Count > 0)
                {
                    var stkDtos = StockPositionComputations.ComputeRunningFields(
                        stks.OrderBy(s => s.Date).ThenBy(s => s.Id).ToList());
                    var stkPnl = stkDtos.Sum(d => d.Pnl);
                    totalPnl += stkPnl;
                    if (stkPnl >= 0) premiumCollected += stkPnl;
                    else premiumLost += Math.Abs(stkPnl);

                    if (stkDtos.Last().TotalPos != 0) hasOpen = true;
                }
            }

            result.Add(new ChainSummaryDto
            {
                RootTradeId = root.Id,
                Symbol = root.Symbol,
                Strategy = names.GetValueOrDefault(root.Strategy, "Unknown"),
                Budget = names.GetValueOrDefault(root.Budget, "Unknown"),
                ChainLength = chainTrades.Count,
                Status = hasOpen ? "Open" : "Closed",
                StartDate = root.Date,
                TotalPnl = totalPnl,
                PremiumCollected = premiumCollected,
                PremiumLost = premiumLost,
                EventCount = 0,
            });
        }

        return result.OrderByDescending(c => c.StartDate).ToList();
    }
}
