namespace tradelog.Dtos;

/// <summary>Result of a TWS live sync (Greeks + stock prices only).</summary>
public class LiveSyncResultDto
{
    public bool Success { get; set; }
    public string Message { get; set; } = string.Empty;
    public int GreeksLogged { get; set; }
    public int StockPricesUpdated { get; set; }
    public int CapitalAggregationsUpdated { get; set; }

    public string ToSummary()
    {
        var parts = new List<string>();
        if (GreeksLogged > 0) parts.Add($"{GreeksLogged} Greeks snapshots");
        if (StockPricesUpdated > 0) parts.Add($"{StockPricesUpdated} stock prices updated");
        if (CapitalAggregationsUpdated > 0) parts.Add($"capital aggregations updated");
        return parts.Count > 0 ? string.Join(", ", parts) : "No changes";
    }
}
