using System.ComponentModel.DataAnnotations;

namespace tradelog.Models;

/// <summary>
/// A named markdown document — used for strategy playbooks, trading rules, and reference material.
/// Optionally linked to one or more strategies via the DocumentStrategyLink junction table.
/// </summary>
public class Document : tradelog.Data.IAccountScoped
{
    public int Id { get; set; }
    public int AccountId { get; set; }

    [Required, StringLength(200)]
    public string Title { get; set; } = string.Empty;

    /// <summary>Markdown content.</summary>
    public string? Content { get; set; }

    public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;

    public ICollection<DocumentStrategyLink> StrategyLinks { get; set; } = new List<DocumentStrategyLink>();
}

/// <summary>
/// Many-to-many junction between Documents and LookupValues (strategies).
/// </summary>
public class DocumentStrategyLink
{
    public int DocumentId { get; set; }
    public Document Document { get; set; } = null!;

    public int LookupValueId { get; set; }
    public LookupValue LookupValue { get; set; } = null!;
}
