namespace tradelog.Dtos;

public class FlexSyncResultDto
{
    public bool Success { get; set; }
    public string Message { get; set; } = string.Empty;
    public int TradesCreated { get; set; }
    public int TradesUpdated { get; set; }
    public int OptionPositionsCreated { get; set; }
    public int OptionPositionsClosed { get; set; }
    public int CapitalDaysCreated { get; set; }
    public int OptionEventsProcessed { get; set; }
    public List<string> Warnings { get; set; } = new();

    public string ToSummary()
    {
        var parts = new List<string>();
        if (TradesCreated > 0) parts.Add($"{TradesCreated} trades imported");
        if (TradesUpdated > 0) parts.Add($"{TradesUpdated} trades updated");
        if (OptionPositionsCreated > 0) parts.Add($"{OptionPositionsCreated} option positions created");
        if (OptionPositionsClosed > 0) parts.Add($"{OptionPositionsClosed} option positions closed");
        if (OptionEventsProcessed > 0) parts.Add($"{OptionEventsProcessed} option events (assign/expire)");
        if (CapitalDaysCreated > 0) parts.Add($"{CapitalDaysCreated} capital days imported");
        if (Warnings.Count > 0) parts.Add($"{Warnings.Count} warnings");
        return parts.Count > 0 ? string.Join(", ", parts) : "No changes";
    }
}
