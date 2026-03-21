using System.ComponentModel.DataAnnotations;

namespace tradelog.Models;

public class TradeEntry : tradelog.Data.IAccountScoped
{
    public int Id { get; set; }
    public int AccountId { get; set; }

    [Required, StringLength(20)]
    public string Symbol { get; set; } = string.Empty;

    [Required]
    public DateTime Date { get; set; }

    [Required]
    public TypeOfTrade TypeOfTrade { get; set; }

    public string? Notes { get; set; }
    public DirectionalBias? Directional { get; set; }
    public Timeframe? Timeframe { get; set; }

    [Required]
    public Budget Budget { get; set; }

    [Required]
    public Strategy Strategy { get; set; }

    public bool NewsCatalyst { get; set; }
    public bool RecentEarnings { get; set; }
    public bool SectorSupport { get; set; }
    public bool Ath { get; set; }
    public decimal? Rvol { get; set; }

    [StringLength(100)]
    public string? InstitutionalSupport { get; set; }

    public decimal? GapPct { get; set; }
    public decimal? XAtrMove { get; set; }
    public string? TaFaNotes { get; set; }
    public string? IntendedManagement { get; set; }
    public string? ActualManagement { get; set; }
    public ManagementRating? ManagementRating { get; set; }
    public string? Learnings { get; set; }
}
