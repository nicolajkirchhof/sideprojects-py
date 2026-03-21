using System.ComponentModel.DataAnnotations;

namespace tradelog.Models;

public class Portfolio
{
    public int Id { get; set; }

    [Required]
    public Budget Budget { get; set; }

    [Required]
    public Strategy Strategy { get; set; }

    public decimal MinAllocation { get; set; }
    public decimal MaxAllocation { get; set; }
}
