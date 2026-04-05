using System.ComponentModel.DataAnnotations;

namespace tradelog.Models;

public class TradeEvent : tradelog.Data.IAccountScoped
{
    public int Id { get; set; }
    public int AccountId { get; set; }

    [Required]
    public int TradeId { get; set; }

    [Required]
    public TradeEventType Type { get; set; }

    [Required]
    public DateTime Date { get; set; }

    public string? Notes { get; set; }

    public decimal? PnlImpact { get; set; }
}
