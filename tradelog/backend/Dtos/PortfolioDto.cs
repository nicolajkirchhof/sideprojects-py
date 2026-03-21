namespace tradelog.Dtos;

public class PortfolioDto
{
    public int Id { get; set; }
    public string Budget { get; set; } = string.Empty;
    public string Strategy { get; set; } = string.Empty;
    public decimal MinAllocation { get; set; }
    public decimal MaxAllocation { get; set; }
    public decimal CurrentAllocation { get; set; }
    public decimal Pnl { get; set; }
}
