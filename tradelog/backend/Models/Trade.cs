using System.ComponentModel.DataAnnotations;

namespace tradelog.Models;

public class Trade : tradelog.Data.IAccountScoped
{
    public int Id { get; set; }
    public int AccountId { get; set; }

    [Required, StringLength(20)]
    public string Symbol { get; set; } = string.Empty;

    [Required]
    public DateTime Date { get; set; }

    /// <summary>FK → LookupValues (Category = "TypeOfTrade")</summary>
    public int TypeOfTrade { get; set; }

    public string? Notes { get; set; }

    /// <summary>FK → LookupValues (Category = "Directional"), nullable</summary>
    public int? Directional { get; set; }

    /// <summary>FK → LookupValues (Category = "Timeframe"), nullable</summary>
    public int? Timeframe { get; set; }

    /// <summary>FK → LookupValues (Category = "Budget")</summary>
    public int Budget { get; set; }

    /// <summary>FK → LookupValues (Category = "Strategy")</summary>
    public int Strategy { get; set; }

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
    /// <summary>FK → LookupValues (Category = "ManagementRating"), nullable</summary>
    public int? ManagementRating { get; set; }
    public string? Learnings { get; set; }

    public int? ParentTradeId { get; set; }
    public ICollection<OptionPosition> OptionPositions { get; set; } = new List<OptionPosition>();
    public ICollection<StockPosition> StockPositions { get; set; } = new List<StockPosition>();
    public ICollection<TradeEvent> TradeEvents { get; set; } = new List<TradeEvent>();
}
