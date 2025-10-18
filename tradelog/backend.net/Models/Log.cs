using System.ComponentModel.DataAnnotations.Schema;

namespace backend.net.Models;

[Flags]
public enum Sentiments
{
    None = 0,
    Bullish = 1,
    Neutral = 2,
    Bearish = 4,
}

public class Log
{
    public int Id { get; set; }
    public int InstrumentId { get; set; }
    public DateTime Date { get; set; }
    public string? Notes { get; set; }
    public string? Strategy { get; set; }

    public int? Sentiment { get; set; }
    public string? TA { get; set; }
    public string? ExpectedOutcome { get; set; }
    public string? Lernings { get; set; }
    public string? FA { get; set; }
}
