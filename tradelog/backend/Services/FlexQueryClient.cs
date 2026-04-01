using System.Xml.Linq;

namespace tradelog.Services;

/// <summary>
/// HTTP client for the IBKR Flex Web Service API.
/// Two-step process: request a statement → poll until ready → return XML.
/// </summary>
public class FlexQueryClient
{
    private const string SendRequestUrl = "https://gdcdyn.interactivebrokers.com/Universal/servlet/FlexStatementService.SendRequest";
    private const string GetStatementUrl = "https://gdcdyn.interactivebrokers.com/Universal/servlet/FlexStatementService.GetStatement";
    private static readonly TimeSpan PollInterval = TimeSpan.FromSeconds(5);
    private static readonly TimeSpan PollTimeout = TimeSpan.FromMinutes(2);

    private readonly IHttpClientFactory _httpFactory;
    private readonly ILogger<FlexQueryClient> _logger;

    public FlexQueryClient(IHttpClientFactory httpFactory, ILogger<FlexQueryClient> logger)
    {
        _httpFactory = httpFactory;
        _logger = logger;
    }

    /// <summary>
    /// Fetches a Flex report: sends the request, polls until ready, returns the raw XML.
    /// </summary>
    public async Task<string> FetchReportAsync(string token, string queryId, CancellationToken ct = default)
    {
        var referenceCode = await SendRequestAsync(token, queryId, ct);
        _logger.LogInformation("Flex report requested, referenceCode={ReferenceCode}. Polling for result...", referenceCode);
        return await PollForStatementAsync(token, referenceCode, ct);
    }

    private async Task<string> SendRequestAsync(string token, string queryId, CancellationToken ct)
    {
        var client = _httpFactory.CreateClient();
        var url = $"{SendRequestUrl}?t={token}&q={queryId}&v=3";
        _logger.LogInformation("Sending Flex request for queryId={QueryId}", queryId);

        var response = await client.GetStringAsync(url, ct);
        var doc = XDocument.Parse(response);
        var root = doc.Root ?? throw new InvalidOperationException("Empty Flex response");

        var status = root.Element("Status")?.Value;
        if (status != "Success")
        {
            var errorCode = root.Element("ErrorCode")?.Value ?? "?";
            var errorMsg = root.Element("ErrorMessage")?.Value ?? response;
            throw new InvalidOperationException($"Flex request failed (code {errorCode}): {errorMsg}");
        }

        var refCode = root.Element("ReferenceCode")?.Value;
        if (string.IsNullOrEmpty(refCode))
            throw new InvalidOperationException("Flex response missing ReferenceCode");

        return refCode;
    }

    private async Task<string> PollForStatementAsync(string token, string referenceCode, CancellationToken ct)
    {
        var client = _httpFactory.CreateClient();
        var url = $"{GetStatementUrl}?q={referenceCode}&t={token}&v=3";
        var deadline = DateTime.UtcNow + PollTimeout;

        while (DateTime.UtcNow < deadline)
        {
            ct.ThrowIfCancellationRequested();
            var response = await client.GetStringAsync(url, ct);

            // If the response is the actual Flex report (starts with FlexQueryResponse or FlexStatements),
            // return it directly. Otherwise it's a status envelope.
            if (response.Contains("<FlexQueryResponse") || response.Contains("<FlexStatements"))
            {
                _logger.LogInformation("Flex report received ({Length} chars)", response.Length);
                return response;
            }

            // Parse status envelope
            var doc = XDocument.Parse(response);
            var root = doc.Root;
            var status = root?.Element("Status")?.Value;

            if (status == "Success")
            {
                // Some versions return the URL in the response
                var statementUrl = root?.Element("Url")?.Value;
                if (!string.IsNullOrEmpty(statementUrl))
                {
                    var report = await client.GetStringAsync(statementUrl, ct);
                    _logger.LogInformation("Flex report downloaded from URL ({Length} chars)", report.Length);
                    return report;
                }
            }

            if (status == "Warn")
            {
                _logger.LogDebug("Flex report not ready yet, retrying in {Interval}s...", PollInterval.TotalSeconds);
                await Task.Delay(PollInterval, ct);
                continue;
            }

            if (status == "Fail")
            {
                var errorCode = root?.Element("ErrorCode")?.Value ?? "?";
                var errorMsg = root?.Element("ErrorMessage")?.Value ?? response;
                throw new InvalidOperationException($"Flex statement failed (code {errorCode}): {errorMsg}");
            }

            // Unknown status — wait and retry
            _logger.LogWarning("Unexpected Flex poll status: {Status}, retrying...", status);
            await Task.Delay(PollInterval, ct);
        }

        throw new TimeoutException($"Flex report not ready after {PollTimeout.TotalSeconds}s");
    }
}
