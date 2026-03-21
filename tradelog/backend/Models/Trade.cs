using System.ComponentModel.DataAnnotations;

namespace tradelog.Models;

public class Trade
{
    public int Id { get; set; }

    [Required, StringLength(20)]
    public string Symbol { get; set; } = string.Empty;

    [Required]
    public DateTime Date { get; set; }

    [Required]
    public int PosChange { get; set; }

    [Required]
    public decimal Price { get; set; }

    public decimal Commission { get; set; }
    public int Multiplier { get; set; } = 1;

    /// <summary>IBKR contract ID for TWS API matching.</summary>
    public int? ConId { get; set; }

    /// <summary>IBKR execution ID for dedup.</summary>
    [StringLength(50)]
    public string? ExecutionId { get; set; }
}
