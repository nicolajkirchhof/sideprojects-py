using tradelog.Data;
using tradelog.Models;
using tradelog.Services;
using Microsoft.EntityFrameworkCore;
using System.Text.Json.Serialization;

var builder = WebApplication.CreateBuilder(args);

// Load user-secrets in all environments (not just Development)
builder.Configuration.AddUserSecrets<Program>();

var connectionString = builder.Configuration.GetConnectionString("DefaultConnection")!;

builder.Services.AddDbContext<DataContext>(options => options.UseSqlite(connectionString));
builder.Services.AddControllers(options =>
{
    options.ReturnHttpNotAcceptable = true;
})
.AddJsonOptions(options =>
{
    options.JsonSerializerOptions.Converters.Add(new JsonStringEnumConverter());
});
builder.Services.AddHttpContextAccessor();
builder.Services.AddHttpClient();
builder.Services.AddScoped<IAccountContext, AccountContext>();
builder.Services.AddScoped<TwsLiveSyncService>();
builder.Services.AddScoped<FlexQueryClient>();
builder.Services.AddSingleton<FlexReportParser>();
builder.Services.AddScoped<FlexSyncService>(sp => new FlexSyncService(
    sp.GetRequiredService<DataContext>(),
    sp.GetRequiredService<ILogger<FlexSyncService>>(),
    cutoffDate: builder.Configuration.GetValue<DateTime?>("FlexSyncCutoffDate"),
    statusService: sp.GetRequiredService<TradeStatusService>()));
builder.Services.AddScoped<OptionPositionLogCountService>();
builder.Services.AddScoped<TradeStatusService>();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

// CORS only needed in Development (ng serve on port 4200)
builder.Services.AddCors(options =>
{
    options.AddPolicy("Cors", policy =>
    {
        policy.WithOrigins("http://localhost:4200")
            .AllowAnyHeader()
            .AllowAnyMethod();
    });
});

var app = builder.Build();

using (var scope = app.Services.CreateScope())
{
    var services = scope.ServiceProvider;
    var context = services.GetRequiredService<DataContext>();

    context.Database.EnsureCreated();

    // One-off backfill: populate LogCount for any open positions that have never
    // been computed. Runs synchronously after migrate so the column is coherent
    // before the first request is served.
    var logCountService = services.GetRequiredService<OptionPositionLogCountService>();
    var backfilled = await logCountService.RecomputeForPendingOpenAsync();
    if (backfilled > 0)
    {
        app.Logger.LogInformation("Backfilled LogCount for {Count} open position(s).", backfilled);
    }

    var statusService = services.GetRequiredService<TradeStatusService>();
    var statusBackfilled = await statusService.RecomputeForPendingAsync();
    if (statusBackfilled > 0)
    {
        app.Logger.LogInformation("Backfilled Status for {Count} trade(s).", statusBackfilled);
    }

    // Seed strategy documents from .md files (one-off — skips if documents already exist)
    var hasDocuments = await context.Documents.IgnoreQueryFilters().AnyAsync();
    if (!hasDocuments)
    {
        var mdDir = Path.Combine(app.Environment.ContentRootPath, "..", "investing_framework");
        if (Directory.Exists(mdDir))
        {
            var accounts = await context.Accounts.IgnoreQueryFilters().ToListAsync();
            foreach (var account in accounts)
            {
                foreach (var file in Directory.GetFiles(mdDir, "*.md"))
                {
                    var title = Path.GetFileNameWithoutExtension(file)
                        .Replace("-", " ").Replace("_", " ");
                    var content = await File.ReadAllTextAsync(file);
                    context.Documents.Add(new Document
                    {
                        AccountId = account.Id,
                        Title = title,
                        Content = content,
                        UpdatedAt = DateTime.UtcNow,
                    });
                }
            }
            await context.SaveChangesAsync();
            app.Logger.LogInformation("Seeded {Count} strategy documents from .md files.",
                Directory.GetFiles(mdDir, "*.md").Length);
        }
    }
}

if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI(options =>
    {
        options.SwaggerEndpoint("/swagger/v1/swagger.json", "v1");
        options.RoutePrefix = "swagger";
    });
    app.UseCors("Cors");
}

// Serve Angular frontend as static files (for Staging/Production standalone mode)
app.UseDefaultFiles();
app.UseStaticFiles();

app.MapControllers();

// SPA fallback: serve index.html for non-API routes (Angular routing)
app.MapFallbackToFile("index.html");

app.Run();
