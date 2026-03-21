using System.ComponentModel.DataAnnotations;

namespace tradelog.Models;

public class IbkrConfig
{
    public int Id { get; set; }

    [Required, StringLength(100)]
    public string Host { get; set; } = "127.0.0.1";

    [Required]
    public int Port { get; set; } = 7497;

    [Required]
    public int ClientId { get; set; } = 1;

    public DateTime? LastSyncAt { get; set; }

    [StringLength(500)]
    public string? LastSyncResult { get; set; }
}
