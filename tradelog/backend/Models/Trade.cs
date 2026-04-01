using System.ComponentModel.DataAnnotations;

namespace tradelog.Models;

public class Trade : tradelog.Data.IAccountScoped
{
    public int Id { get; set; }
    public int AccountId { get; set; }

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

    /// <summary>IBKR execution ID / Flex TradeID for dedup.</summary>
    [StringLength(50)]
    public string? ExecutionId { get; set; }

    /// <summary>Best exit price for this execution (for trade review).</summary>
    public decimal? BestExitPrice { get; set; }

    /// <summary>Date of the best exit opportunity.</summary>
    public DateTime? BestExitDate { get; set; }

    // ── Flex Tier 1 fields ──────────────────────────

    /// <summary>Execution venue (e.g., DARK, NYSE, CBOE).</summary>
    [StringLength(20)]
    public string? Exchange { get; set; }

    /// <summary>FX rate to base currency (EUR) at trade time.</summary>
    public decimal FxRateToBase { get; set; }

    /// <summary>FIFO realized P&L from IBKR.</summary>
    public decimal FifoPnlRealized { get; set; }

    /// <summary>Transaction taxes (e.g., stamp duty).</summary>
    public decimal Taxes { get; set; }
}
