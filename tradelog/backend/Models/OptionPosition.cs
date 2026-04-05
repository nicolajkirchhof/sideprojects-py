using System.ComponentModel.DataAnnotations;

namespace tradelog.Models;

public class OptionPosition : tradelog.Data.IAccountScoped
{
    public int Id { get; set; }
    public int AccountId { get; set; }

    [Required, StringLength(60)]
    public string Symbol { get; set; } = string.Empty;

    [Required, StringLength(40)]
    public string ContractId { get; set; } = string.Empty;

    [Required]
    public DateTime Opened { get; set; }

    [Required]
    public DateTime Expiry { get; set; }

    public DateTime? Closed { get; set; }

    [Required]
    public int Pos { get; set; }

    [Required]
    public PositionRight Right { get; set; }

    [Required]
    public decimal Strike { get; set; }

    [Required]
    public decimal Cost { get; set; }

    public decimal? ClosePrice { get; set; }
    public decimal Commission { get; set; }
    public int Multiplier { get; set; } = 100;
    public CloseReasons? CloseReasons { get; set; }

    /// <summary>IBKR contract ID for TWS API matching.</summary>
    public int? ConId { get; set; }

    /// <summary>IBKR security type: "OPT" or "FOP".</summary>
    [StringLength(10)]
    public string? SecType { get; set; }

    /// <summary>Underlying ticker symbol (e.g., SPY for SPY options).</summary>
    [StringLength(20)]
    public string? UnderlyingSymbol { get; set; }

    /// <summary>Underlying IBKR contract ID — stable reference across symbol changes.</summary>
    public int? UnderlyingConid { get; set; }

    public int? TradeId { get; set; }

    /// <summary>Best exit price the position could have achieved (for trade review).</summary>
    public decimal? BestExitPrice { get; set; }

    /// <summary>Date of the best exit opportunity.</summary>
    public DateTime? BestExitDate { get; set; }
}
