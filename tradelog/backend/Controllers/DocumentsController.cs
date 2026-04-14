using tradelog.Data;
using tradelog.Dtos;
using tradelog.Models;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

namespace tradelog.Controllers;

[ApiController]
[Route("api/documents")]
[Produces("application/json")]
public class DocumentsController : ControllerBase
{
    private readonly DataContext _context;

    public DocumentsController(DataContext context)
    {
        _context = context;
    }

    [HttpGet]
    public async Task<ActionResult<IEnumerable<DocumentDto>>> GetAll()
    {
        var docs = await _context.Documents
            .Include(d => d.StrategyLinks)
            .OrderBy(d => d.Title)
            .ToListAsync();

        return docs.Select(ToDto).ToList();
    }

    [HttpGet("{id:int}")]
    public async Task<ActionResult<DocumentDto>> GetById(int id)
    {
        var doc = await _context.Documents
            .Include(d => d.StrategyLinks)
            .FirstOrDefaultAsync(d => d.Id == id);
        if (doc == null) return NotFound();
        return ToDto(doc);
    }

    [HttpPost]
    public async Task<ActionResult<DocumentDto>> Create(DocumentUpsertDto dto)
    {
        var doc = new Document
        {
            Title = dto.Title.Trim(),
            Content = dto.Content,
            UpdatedAt = DateTime.UtcNow,
        };

        _context.Documents.Add(doc);
        await _context.SaveChangesAsync();

        if (dto.StrategyIds is { Count: > 0 })
        {
            foreach (var sid in dto.StrategyIds)
                _context.DocumentStrategyLinks.Add(new DocumentStrategyLink { DocumentId = doc.Id, LookupValueId = sid });
            await _context.SaveChangesAsync();
        }

        return CreatedAtAction(nameof(GetById), new { id = doc.Id }, ToDto(doc));
    }

    [HttpPut("{id:int}")]
    public async Task<IActionResult> Update(int id, DocumentUpsertDto dto)
    {
        var doc = await _context.Documents
            .Include(d => d.StrategyLinks)
            .FirstOrDefaultAsync(d => d.Id == id);
        if (doc == null) return NotFound();

        doc.Title = dto.Title.Trim();
        doc.Content = dto.Content;
        doc.UpdatedAt = DateTime.UtcNow;

        // Replace strategy links
        if (dto.StrategyIds != null)
        {
            _context.DocumentStrategyLinks.RemoveRange(doc.StrategyLinks);
            foreach (var sid in dto.StrategyIds)
                _context.DocumentStrategyLinks.Add(new DocumentStrategyLink { DocumentId = doc.Id, LookupValueId = sid });
        }

        await _context.SaveChangesAsync();
        return NoContent();
    }

    [HttpDelete("{id:int}")]
    public async Task<IActionResult> Delete(int id)
    {
        var doc = await _context.Documents.FindAsync(id);
        if (doc == null) return NotFound();

        _context.Documents.Remove(doc);
        await _context.SaveChangesAsync();
        return NoContent();
    }

    private static DocumentDto ToDto(Document doc) => new()
    {
        Id = doc.Id,
        Title = doc.Title,
        Content = doc.Content,
        UpdatedAt = doc.UpdatedAt,
        StrategyIds = doc.StrategyLinks.Select(l => l.LookupValueId).ToList(),
    };
}
