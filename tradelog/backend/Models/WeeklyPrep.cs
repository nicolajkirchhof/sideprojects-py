using System.ComponentModel.DataAnnotations;

namespace tradelog.Models;

public class WeeklyPrep : tradelog.Data.IAccountScoped
{
    public int Id { get; set; }
    public int AccountId { get; set; }

    [Required]
    public DateTime Date { get; set; }

    [StringLength(20)]
    public string? IndexBias { get; set; }

    [StringLength(20)]
    public string? Breadth { get; set; }

    public string? NotableSectors { get; set; }
    public string? VolatilityNotes { get; set; }
    public string? OpenPositionsRequiringManagement { get; set; }

    [StringLength(20)]
    public string? CurrentPortfolioRisk { get; set; }

    public string? PortfolioNotes { get; set; }

    [StringLength(50)]
    public string? ScanningFor { get; set; }

    public string? IndexSectorPreference { get; set; }
    public string? Watchlist { get; set; }
    public string? Learnings { get; set; }
    public string? FocusForImprovement { get; set; }
    public string? ExternalComments { get; set; }
}
