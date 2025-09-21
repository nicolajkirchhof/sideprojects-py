using System.ComponentModel.DataAnnotations;

namespace backend.net.Models;

public class Trade
{
    public int Id { get; set; }
    [StringLength(50)]
    public string Symbol { get; set; }
    [StringLength(50)]
    public string ProfitMechanism { get; set; }
}
