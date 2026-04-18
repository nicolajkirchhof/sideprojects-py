namespace tradelog.Dtos;

/// <summary>Bulk trade export DTO for the Python analyst pipeline.</summary>
public class TradeExportDto
{
    public int Id { get; set; }
    public string Symbol { get; set; } = string.Empty;
    public DateTime Date { get; set; }
    public string? TypeOfTrade { get; set; }
    public string? Strategy { get; set; }
    public string? Directional { get; set; }
    public string? Budget { get; set; }
    public string? Status { get; set; }
    public decimal? Pnl { get; set; }
    public string? Notes { get; set; }
    public string? IntendedManagement { get; set; }
    public string? ActualManagement { get; set; }
    public int? ManagementRating { get; set; }
    public string? Learnings { get; set; }

    /// <summary>Dates when this trade already has an analysis (to skip re-analysis).</summary>
    public List<DateTime> ExistingAnalysisDates { get; set; } = [];
}
