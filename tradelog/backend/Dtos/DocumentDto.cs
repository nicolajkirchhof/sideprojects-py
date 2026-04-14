namespace tradelog.Dtos;

public class DocumentDto
{
    public int Id { get; set; }
    public string Title { get; set; } = string.Empty;
    public string? Content { get; set; }
    public DateTime UpdatedAt { get; set; }
    public List<int> StrategyIds { get; set; } = [];
}

public class DocumentUpsertDto
{
    public string Title { get; set; } = string.Empty;
    public string? Content { get; set; }
    public List<int>? StrategyIds { get; set; }
}
