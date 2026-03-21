using System.ComponentModel.DataAnnotations;

namespace tradelog.Models;

public class Account
{
    public int Id { get; set; }

    [Required, StringLength(20)]
    public string IbkrAccountId { get; set; } = string.Empty;

    [Required, StringLength(50)]
    public string Name { get; set; } = string.Empty;

    [Required, StringLength(100)]
    public string Host { get; set; } = "127.0.0.1";

    [Required]
    public int Port { get; set; } = 7497;

    [Required]
    public int ClientId { get; set; } = 1;

    public bool IsDefault { get; set; }

    public DateTime? LastSyncAt { get; set; }

    [StringLength(500)]
    public string? LastSyncResult { get; set; }
}
