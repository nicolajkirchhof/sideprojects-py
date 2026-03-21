using System.ComponentModel.DataAnnotations;

namespace tradelog.Models;

public class OptionPosition
{
    public int Id { get; set; }

    [Required, StringLength(20)]
    public string Symbol { get; set; } = string.Empty;

    [Required, StringLength(20)]
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
}
