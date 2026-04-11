using System.ComponentModel.DataAnnotations;

namespace tradelog.Models;

/// <summary>
/// A user-configurable enum value stored in the database.
/// Replaces hardcoded C# enums for Strategy, TypeOfTrade, Budget,
/// Timeframe, DirectionalBias, and ManagementRating.
/// </summary>
public class LookupValue : tradelog.Data.IAccountScoped
{
    public int Id { get; set; }
    public int AccountId { get; set; }

    /// <summary>Discriminator: "Strategy", "TypeOfTrade", "Budget", etc.</summary>
    [Required, StringLength(30)]
    public string Category { get; set; } = string.Empty;

    [Required, StringLength(100)]
    public string Name { get; set; } = string.Empty;

    public int SortOrder { get; set; }
    public bool IsActive { get; set; } = true;
}

/// <summary>Well-known category names used as discriminators in LookupValues.</summary>
public static class LookupCategory
{
    public const string Strategy = "Strategy";
    public const string TypeOfTrade = "TypeOfTrade";
    public const string Budget = "Budget";
    public const string Timeframe = "Timeframe";
    public const string Directional = "Directional";
    public const string ManagementRating = "ManagementRating";

    public static readonly string[] All = [Strategy, TypeOfTrade, Budget, Timeframe, Directional, ManagementRating];
}
