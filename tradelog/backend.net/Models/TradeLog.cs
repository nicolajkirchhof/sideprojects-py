using System.ComponentModel.DataAnnotations.Schema;

namespace backend.net.Models;

public class TradeLog
{
    public int Id { get; set; }
    public int TradeId { get; set; }
    public DateTime Date { get; set; }
    public string? RichTextNotes { get; set; }

    [ForeignKey("TradeId")]
    public Trade Trade { get; set; }
}
