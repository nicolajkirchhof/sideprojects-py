using System.ComponentModel.DataAnnotations;

namespace backend.net.Models;

public class Instrument
{
    public int Id { get; set; }
    [StringLength(50)]
    public string SecType { get; set; }
    public int ContractId { get; set; }
    [StringLength(50)]
    public string Symbol { get; set; }
    public int Multiplier { get; set; }
    [StringLength(50)]
    public string Sector { get; set; }
    [StringLength(50)]
    public string Subsector { get; set; }
}
