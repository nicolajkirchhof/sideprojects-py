namespace tradelog.Services;

/// <summary>
/// Scoped service that resolves the current account ID from the X-Account-Id HTTP header.
/// Used by DataContext for global query filters and auto-setting AccountId on new entities.
/// </summary>
public interface IAccountContext
{
    int CurrentAccountId { get; }
}

public class AccountContext : IAccountContext
{
    public int CurrentAccountId { get; }

    public AccountContext(IHttpContextAccessor httpContextAccessor)
    {
        var header = httpContextAccessor.HttpContext?.Request.Headers["X-Account-Id"].FirstOrDefault();
        if (int.TryParse(header, out var id))
            CurrentAccountId = id;
    }
}
