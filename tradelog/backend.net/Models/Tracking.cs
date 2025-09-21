using System.ComponentModel.DataAnnotations;

namespace backend.net.Models;

public class Tracking
{
    public int Id { get; set; }
    public int ContractId { get; set; }
    [StringLength(50)]
    public string Underlying { get; set; }
    public float IV { get; set; }
    public float Price { get; set; }
    public float TimeValue { get; set; }
    public float Delta { get; set; }
    public float Theta { get; set; }
    public float Gamma { get; set; }
    public float Vega { get; set; }
    public float Margin { get; set; }
}
