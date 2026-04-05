namespace tradelog.Dtos;

public class TradeEventDto
{
    public int Id { get; set; }
    public int TradeId { get; set; }
    public string Type { get; set; } = string.Empty;
    public DateTime Date { get; set; }
    public string? Notes { get; set; }
    public decimal? PnlImpact { get; set; }
}
