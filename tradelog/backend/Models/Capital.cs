using System.ComponentModel.DataAnnotations;

namespace tradelog.Models;

public class Capital : tradelog.Data.IAccountScoped
{
    public int Id { get; set; }
    public int AccountId { get; set; }

    [Required]
    public DateTime Date { get; set; }

    // Manual fields
    public decimal NetLiquidity { get; set; }
    public decimal Maintenance { get; set; }
    public decimal ExcessLiquidity { get; set; }
    public decimal Bpr { get; set; }

    // Snapshotted aggregations (computed at insert time, stored)
    public decimal MaintenancePct { get; set; }
    public decimal TotalPnl { get; set; }
    public decimal UnrealizedPnl { get; set; }
    public decimal RealizedPnl { get; set; }
    public decimal NetDelta { get; set; }
    public decimal NetTheta { get; set; }
    public decimal NetVega { get; set; }
    public decimal NetGamma { get; set; }
    public decimal AvgIv { get; set; }
    public decimal TotalMargin { get; set; }
    public decimal TotalCommissions { get; set; }
}
