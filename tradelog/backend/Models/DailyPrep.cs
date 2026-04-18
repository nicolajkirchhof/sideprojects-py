using System.ComponentModel.DataAnnotations;

namespace tradelog.Models;

public class DailyPrep : tradelog.Data.IAccountScoped
{
    public int Id { get; set; }
    public int AccountId { get; set; }

    [Required]
    public DateTime Date { get; set; }

    /// <summary>JSON — market regime, themes, movers, risks, action items.</summary>
    public string? MarketSummary { get; set; }

    /// <summary>JSON — scored scanner candidates with 5-box results.</summary>
    public string? Watchlist { get; set; }

    public int EmailCount { get; set; }
    public int CandidateCount { get; set; }

    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;
}
