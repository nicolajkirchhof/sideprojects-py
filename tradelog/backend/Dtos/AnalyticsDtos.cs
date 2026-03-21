namespace tradelog.Dtos;

public class StrategyPerformanceDto
{
    public string Strategy { get; set; } = string.Empty;
    public int TradeCount { get; set; }
    public decimal TotalPnl { get; set; }
    public decimal AvgWin { get; set; }
    public decimal AvgLoss { get; set; }
    public decimal WinRate { get; set; }
    public decimal Expectancy { get; set; }
    public decimal MaxDrawdown { get; set; }
    public decimal TotalCommissions { get; set; }
}

public class BudgetPerformanceDto
{
    public string Budget { get; set; } = string.Empty;
    public int TradeCount { get; set; }
    public decimal TotalPnl { get; set; }
    public decimal AvgWin { get; set; }
    public decimal AvgLoss { get; set; }
    public decimal WinRate { get; set; }
    public decimal Expectancy { get; set; }
    public decimal TotalCommissions { get; set; }
}

public class OverallPerformanceDto
{
    public decimal TotalPnl { get; set; }
    public decimal TotalCommissions { get; set; }
    public decimal NetPnl { get; set; }
    public decimal DailyPnl { get; set; }
    public decimal AnnualizedRoi { get; set; }
    public int TradingDays { get; set; }
    public int TradeCount { get; set; }
    public decimal WinRate { get; set; }
    public decimal AvgWin { get; set; }
    public decimal AvgLoss { get; set; }
}

public class EquityCurvePointDto
{
    public DateTime Date { get; set; }
    public decimal CumulativePnl { get; set; }
}
