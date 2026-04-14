namespace tradelog.Dtos;

public class TradeDetailDto
{
    public int Id { get; set; }
    public string Symbol { get; set; } = string.Empty;
    public DateTime Date { get; set; }

    // Lookup-backed fields: Id for form binding, Name for display
    public int TypeOfTrade { get; set; }
    public string TypeOfTradeName { get; set; } = string.Empty;
    public string? Notes { get; set; }
    public int? Directional { get; set; }
    public string? DirectionalName { get; set; }
    public int? Timeframe { get; set; }
    public string? TimeframeName { get; set; }
    public int Budget { get; set; }
    public string BudgetName { get; set; } = string.Empty;
    public int Strategy { get; set; }
    public string StrategyName { get; set; } = string.Empty;

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
    public int? ManagementRating { get; set; }
    public string? ManagementRatingName { get; set; }
    public string? Learnings { get; set; }
    public int? ParentTradeId { get; set; }

    // Follow-up chain
    public List<int> ChildTradeIds { get; set; } = [];

    // Linked positions
    public List<OptionPositionDto> OptionPositions { get; set; } = [];
    public List<StockPositionDto> StockPositions { get; set; } = [];
}
