using tradelog.Data;
using tradelog.Services;
using Microsoft.EntityFrameworkCore;
using System.Text.Json.Serialization;

var builder = WebApplication.CreateBuilder(args);

// Load user-secrets in all environments (not just Development)
builder.Configuration.AddUserSecrets<Program>();

// Inject the DB password from user-secrets into the connection string
var connectionString = builder.Configuration.GetConnectionString("DefaultConnection")!;
var passwordKey = $"DbPassword:{builder.Environment.EnvironmentName}";
var dbPassword = builder.Configuration[passwordKey]
    ?? throw new InvalidOperationException($"Missing user-secret '{passwordKey}'. Run: dotnet user-secrets set \"{passwordKey}\" \"<password>\"");
connectionString = connectionString.Replace("{password}", dbPassword);

builder.Services.AddDbContext<DataContext>(options =>
    options.UseSqlServer(connectionString));
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
    cutoffDate: builder.Configuration.GetValue<DateTime?>("FlexSyncCutoffDate")));
builder.Services.AddScoped<OptionPositionLogCountService>();
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
    context.Database.Migrate();

    // One-off backfill: populate LogCount for any open positions that have never
    // been computed. Runs synchronously after migrate so the column is coherent
    // before the first request is served.
    var logCountService = services.GetRequiredService<OptionPositionLogCountService>();
    var backfilled = await logCountService.RecomputeForPendingOpenAsync();
    if (backfilled > 0)
    {
        app.Logger.LogInformation("Backfilled LogCount for {Count} open position(s).", backfilled);
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
