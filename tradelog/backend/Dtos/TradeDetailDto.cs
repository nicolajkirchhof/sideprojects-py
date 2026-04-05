using tradelog.Models;

namespace tradelog.Dtos;

public class TradeDetailDto
{
    public int Id { get; set; }
    public string Symbol { get; set; } = string.Empty;
    public DateTime Date { get; set; }
    public TypeOfTrade TypeOfTrade { get; set; }
    public string? Notes { get; set; }
    public DirectionalBias? Directional { get; set; }
    public Timeframe? Timeframe { get; set; }
    public Budget Budget { get; set; }
    public Strategy Strategy { get; set; }
    public bool NewsCatalyst { get; set; }
    public bool RecentEarnings { get; set; }
    public bool SectorSupport { get; set; }
    public bool Ath { get; set; }
    public decimal? Rvol { get; set; }
    public string? InstitutionalSupport { get; set; }
    public decimal? GapPct { get; set; }
    public decimal? XAtrMove { get; set; }
    public string? TaFaNotes { get; set; }
    public string? IntendedManagement { get; set; }
    public string? ActualManagement { get; set; }
    public ManagementRating? ManagementRating { get; set; }
    public string? Learnings { get; set; }
    public int? ParentTradeId { get; set; }

    // Linked positions
    public List<OptionPositionDto> OptionPositions { get; set; } = [];
    public List<StockPositionDto> StockPositions { get; set; } = [];
}
