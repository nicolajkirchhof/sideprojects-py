using tradelog.Data;
using tradelog.Models;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

namespace tradelog.Controllers;

[ApiController]
[Route("api/accounts")]
[Produces("application/json")]
public class AccountsController : ControllerBase
{
    private readonly DataContext _context;

    public AccountsController(DataContext context)
    {
        _context = context;
    }

    [HttpGet]
    public async Task<ActionResult<IEnumerable<Account>>> GetAll()
    {
        // Bypass query filter — accounts are not account-scoped
        return await _context.Accounts.IgnoreQueryFilters().OrderBy(a => a.Name).ToListAsync();
    }

    [HttpGet("{id:int}")]
    public async Task<ActionResult<Account>> GetById(int id)
    {
        var account = await _context.Accounts.IgnoreQueryFilters().FirstOrDefaultAsync(a => a.Id == id);
        if (account == null) return NotFound();
        return account;
    }

    [HttpPost]
    public async Task<ActionResult<Account>> Create(Account account)
    {
        // If this is the first account, make it default
        var hasAccounts = await _context.Accounts.IgnoreQueryFilters().AnyAsync();
        if (!hasAccounts)
            account.IsDefault = true;

        _context.Accounts.Add(account);
        await _context.SaveChangesAsync();
        return CreatedAtAction(nameof(GetById), new { id = account.Id }, account);
    }

    [HttpPut("{id:int}")]
    public async Task<IActionResult> Update(int id, Account account)
    {
        if (id != account.Id) return BadRequest();

        var existing = await _context.Accounts.IgnoreQueryFilters().FirstOrDefaultAsync(a => a.Id == id);
        if (existing == null) return NotFound();

        existing.Name = account.Name;
        existing.IbkrAccountId = account.IbkrAccountId;
        existing.Host = account.Host;
        existing.Port = account.Port;
        existing.ClientId = account.ClientId;
        existing.IsDefault = account.IsDefault;
        existing.FlexToken = account.FlexToken;
        existing.FlexQueryId = account.FlexQueryId;

        await _context.SaveChangesAsync();
        return NoContent();
    }

    [HttpDelete("{id:int}")]
    public async Task<IActionResult> Delete(int id)
    {
        var account = await _context.Accounts.IgnoreQueryFilters().FirstOrDefaultAsync(a => a.Id == id);
        if (account == null) return NotFound();

        _context.Accounts.Remove(account);
        await _context.SaveChangesAsync();
        return NoContent();
    }
}
