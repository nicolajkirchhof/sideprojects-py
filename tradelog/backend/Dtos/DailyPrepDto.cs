namespace tradelog.Dtos;

public class DailyPrepDto
{
    public int Id { get; set; }
    public DateTime Date { get; set; }
    public string? MarketSummary { get; set; }
    public string? Watchlist { get; set; }
    public int EmailCount { get; set; }
    public int CandidateCount { get; set; }
    public DateTime CreatedAt { get; set; }
    public DateTime UpdatedAt { get; set; }
}

public class DailyPrepUpsertDto
{
    public DateTime Date { get; set; }
    public string? MarketSummary { get; set; }
    public string? Watchlist { get; set; }
    public int EmailCount { get; set; }
    public int CandidateCount { get; set; }
}
