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

    [HttpGet("strategies")]
    public async Task<ActionResult<IEnumerable<StrategyPerformanceDto>>> GetStrategyPerformance()
    {
        var (pnlsBySymbol, tradeEntries) = await LoadClosedData();

        // Group PNLs by strategy (via TradeEntry linkage by symbol)
        var strategyPnls = new Dictionary<string, List<SymbolPnl>>();
        var strategyCommissions = new Dictionary<string, decimal>();

        foreach (var g in pnlsBySymbol.GroupBy(p => p.Symbol))
        {
            var entry = tradeEntries.GetValueOrDefault(g.Key);
            var strategy = entry?.Strategy.ToString() ?? "Unknown";
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
        var (pnlsBySymbol, tradeEntries) = await LoadClosedData();

        var filtered = pnlsBySymbol.Where(p =>
        {
            var entry = tradeEntries.GetValueOrDefault(p.Symbol);
            return entry?.Strategy.ToString() == strategy;
        }).ToList();

        return AnalyticsComputations.BuildEquityCurve(filtered);
    }

    [HttpGet("budgets")]
    public async Task<ActionResult<IEnumerable<BudgetPerformanceDto>>> GetBudgetPerformance()
    {
        var (pnlsBySymbol, tradeEntries) = await LoadClosedData();

        var budgetPnls = new Dictionary<string, List<SymbolPnl>>();
        var budgetCommissions = new Dictionary<string, decimal>();

        foreach (var g in pnlsBySymbol.GroupBy(p => p.Symbol))
        {
            var entry = tradeEntries.GetValueOrDefault(g.Key);
            var budget = entry?.Budget.ToString() ?? "Unknown";
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
        var (pnlsBySymbol, tradeEntries) = await LoadClosedData();

        var filtered = pnlsBySymbol.Where(p =>
        {
            var entry = tradeEntries.GetValueOrDefault(p.Symbol);
            return entry?.Budget.ToString() == budget;
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
    /// Load closed option positions + all trades, compute per-symbol P/L, and fetch TradeEntry metadata.
    /// </summary>
    private async Task<(List<SymbolPnl> pnls, Dictionary<string, TradeEntry> entries)> LoadClosedData()
    {
        var closedOptions = await _context.OptionPositions
            .Where(p => p.Closed != null && p.ClosePrice != null)
            .ToListAsync();

        var trades = await _context.Trades
            .OrderBy(t => t.Symbol).ThenBy(t => t.Date).ThenBy(t => t.Id)
            .ToListAsync();

        var pnls = AnalyticsComputations.ComputeClosedPnls(closedOptions, trades);

        var symbols = pnls.Select(p => p.Symbol).Distinct().ToList();
        var entries = await _context.TradeEntries
            .Where(e => symbols.Contains(e.Symbol))
            .GroupBy(e => e.Symbol)
            .Select(g => g.OrderByDescending(e => e.Date).First())
            .ToDictionaryAsync(e => e.Symbol);

        return (pnls, entries);
    }
}
