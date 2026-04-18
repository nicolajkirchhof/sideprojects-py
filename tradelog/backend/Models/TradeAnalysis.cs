using System.ComponentModel.DataAnnotations;

namespace tradelog.Models;

public class TradeAnalysis : tradelog.Data.IAccountScoped
{
    public int Id { get; set; }
    public int AccountId { get; set; }

    [Required]
    public int TradeId { get; set; }
    public Trade Trade { get; set; } = null!;

    [Required]
    public DateTime AnalysisDate { get; set; }

    /// <summary>Compliance score 1-5.</summary>
    public int Score { get; set; }

    /// <summary>Full markdown analysis from Claude (editable).</summary>
    public string? Analysis { get; set; }

    /// <summary>Claude model used, e.g. "claude-opus-4-6".</summary>
    [StringLength(50)]
    public string? Model { get; set; }

    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
}
