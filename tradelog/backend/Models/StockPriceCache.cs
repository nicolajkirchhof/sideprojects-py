using System.ComponentModel.DataAnnotations;

namespace tradelog.Models;

public class StockPriceCache
{
    public int Id { get; set; }

    [Required, StringLength(20)]
    public string Symbol { get; set; } = string.Empty;

    public decimal LastPrice { get; set; }

    public DateTime UpdatedAt { get; set; }
}
