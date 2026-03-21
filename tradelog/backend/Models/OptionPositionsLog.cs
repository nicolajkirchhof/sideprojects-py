using System.ComponentModel.DataAnnotations;

namespace tradelog.Models;

public class OptionPositionsLog : tradelog.Data.IAccountScoped
{
    public int Id { get; set; }
    public int AccountId { get; set; }

    [Required]
    public DateTime DateTime { get; set; }

    [Required, StringLength(20)]
    public string ContractId { get; set; } = string.Empty;

    public decimal Underlying { get; set; }
    public decimal Iv { get; set; }
    public decimal Price { get; set; }
    public decimal TimeValue { get; set; }
    public decimal Delta { get; set; }
    public decimal Theta { get; set; }
    public decimal Gamma { get; set; }
    public decimal Vega { get; set; }
    public decimal Margin { get; set; }
}
