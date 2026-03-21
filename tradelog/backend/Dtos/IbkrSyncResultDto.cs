namespace tradelog.Dtos;

public class IbkrSyncResultDto
{
    public bool Success { get; set; }
    public string Message { get; set; } = string.Empty;
    public int CapitalCreated { get; set; }
    public int OptionPositionsCreated { get; set; }
    public int OptionPositionsClosed { get; set; }
    public int GreeksLogged { get; set; }
    public int TradesCreated { get; set; }
    public int StockPricesUpdated { get; set; }

    public string ToSummary()
    {
        var parts = new List<string>();
        if (CapitalCreated > 0) parts.Add($"{CapitalCreated} capital snapshot");
        if (OptionPositionsCreated > 0) parts.Add($"{OptionPositionsCreated} new option positions");
        if (OptionPositionsClosed > 0) parts.Add($"{OptionPositionsClosed} positions closed");
        if (GreeksLogged > 0) parts.Add($"{GreeksLogged} Greeks snapshots");
        if (TradesCreated > 0) parts.Add($"{TradesCreated} new trades");
        if (StockPricesUpdated > 0) parts.Add($"{StockPricesUpdated} stock prices updated");
        return parts.Count > 0 ? string.Join(", ", parts) : "No changes";
    }
}
