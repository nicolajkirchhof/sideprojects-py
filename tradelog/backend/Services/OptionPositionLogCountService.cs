using Microsoft.EntityFrameworkCore;
using tradelog.Data;
using tradelog.Models;

namespace tradelog.Services;

/// <summary>
/// Maintains the cached <see cref="OptionPosition.LogCount"/> field.
/// Call <see cref="RecomputeForAsync"/> after writing greek log rows
/// and <see cref="RecomputeForPendingOpenAsync"/> at application startup
/// to backfill any open positions that have never been computed.
/// </summary>
public class OptionPositionLogCountService
{
    private readonly DataContext _context;

    public OptionPositionLogCountService(DataContext context)
    {
        _context = context;
    }

    /// <summary>
    /// Recomputes the log count for the given positions, mutating their
    /// <see cref="OptionPosition.LogCount"/> property. Caller is responsible
    /// for persisting via <see cref="DataContext.SaveChangesAsync"/>.
    /// Uses <c>IgnoreQueryFilters</c> on the log lookup and scopes manually by
    /// each position's <c>AccountId</c> so this works both in account-scoped
    /// request contexts and in the account-less startup initializer.
    /// </summary>
    public async Task RecomputeForAsync(IEnumerable<OptionPosition> positions, CancellationToken ct = default)
    {
        foreach (var p in positions)
        {
            p.LogCount = await _context.OptionPositionsLogs
                .IgnoreQueryFilters()
                .CountAsync(l =>
                    l.AccountId == p.AccountId &&
                    l.ContractId == p.ContractId &&
                    l.DateTime >= p.Opened,
                    ct);
        }
    }

    /// <summary>
    /// Finds all open positions across all accounts whose
    /// <see cref="OptionPosition.LogCount"/> has never been computed (NULL)
    /// and populates them. Intended to run once at application startup;
    /// becomes near-no-op on subsequent runs.
    /// </summary>
    public async Task<int> RecomputeForPendingOpenAsync(CancellationToken ct = default)
    {
        var pending = await _context.OptionPositions
            .IgnoreQueryFilters()
            .Where(p => p.LogCount == null && p.Closed == null)
            .ToListAsync(ct);

        if (pending.Count == 0) return 0;

        await RecomputeForAsync(pending, ct);
        await _context.SaveChangesAsync(ct);
        return pending.Count;
    }
}
