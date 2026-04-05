namespace tradelog.Dtos;

public class OptionInstrumentSummaryDto
{
    public string Symbol { get; set; } = string.Empty;
    public DateTime? Opened { get; set; }
    public DateTime? Closed { get; set; }
    public int Dit { get; set; }
    public int? Dte { get; set; }
    public string Status { get; set; } = "Closed";

    // From latest Trade
    public string? Budget { get; set; }
    public string? CurrentSetup { get; set; }
    public string? Strikes { get; set; }
    public string? IntendedManagement { get; set; }

    // Aggregated P/L
    public decimal Pnl { get; set; }
    public decimal UnrealizedPnl { get; set; }
    public decimal? UnrealizedPnlPct { get; set; }
    public decimal RealizedPnl { get; set; }
    public decimal? RealizedPnlPct { get; set; }

    // Aggregated Greeks
    public decimal TimeValue { get; set; }
    public decimal Delta { get; set; }
    public decimal Theta { get; set; }
    public decimal Gamma { get; set; }
    public decimal Vega { get; set; }
    public decimal? AvgIv { get; set; }
    public decimal Margin { get; set; }

    // Aggregated metrics
    public decimal? DurationPct { get; set; }
    public decimal? PnlPerDurationPct { get; set; }
    public decimal? Roic { get; set; }
    public decimal Commissions { get; set; }
}
