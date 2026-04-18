namespace tradelog.Dtos;

public class TradeAnalysisDto
{
    public int Id { get; set; }
    public int TradeId { get; set; }
    public DateTime AnalysisDate { get; set; }
    public int Score { get; set; }
    public string? Analysis { get; set; }
    public string? Model { get; set; }
    public DateTime CreatedAt { get; set; }
}

public class TradeAnalysisCreateDto
{
    public DateTime AnalysisDate { get; set; }
    public int Score { get; set; }
    public string? Analysis { get; set; }
    public string? Model { get; set; }
}

public class TradeAnalysisUpdateDto
{
    public int Score { get; set; }
    public string? Analysis { get; set; }
}
