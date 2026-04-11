using System.ComponentModel.DataAnnotations;

namespace tradelog.Models;

public class Portfolio : tradelog.Data.IAccountScoped
{
    public int Id { get; set; }
    public int AccountId { get; set; }

    /// <summary>FK → LookupValues (Category = "Budget")</summary>
    public int Budget { get; set; }

    /// <summary>FK → LookupValues (Category = "Strategy")</summary>
    public int Strategy { get; set; }

    public decimal MinAllocation { get; set; }
    public decimal MaxAllocation { get; set; }
}
