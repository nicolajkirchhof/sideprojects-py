using System.Globalization;
using System.Xml.Linq;
using tradelog.Dtos;

namespace tradelog.Services;

/// <summary>
/// Parses IBKR Flex Query XML into typed DTOs. Stateless.
/// Supports both compact (yyyyMMdd) and ISO (yyyy-MM-dd) date formats.
/// </summary>
public class FlexReportParser
{
    private static readonly string[] DateTimeFormats =
    [
        "yyyy-MM-dd HH:mm:ss",
        "yyyyMMdd;HHmmss",
        "yyyy-MM-dd",
        "yyyyMMdd",
    ];

    private static readonly string[] DateFormats =
    [
        "yyyy-MM-dd",
        "yyyyMMdd",
    ];

    private readonly ILogger<FlexReportParser> _logger;

    public FlexReportParser(ILogger<FlexReportParser> logger)
    {
        _logger = logger;
    }

    public List<FlexTradeDto> ParseTrades(string xml)
    {
        var doc = XDocument.Parse(xml);
        var trades = doc.Descendants("Trade").ToList();
        _logger.LogInformation("Parsing {Count} trades from Flex XML", trades.Count);

        return trades.Select(el => new FlexTradeDto
        {
            TradeId = Attr(el, "tradeID"),
            ConId = AttrInt(el, "conid"),
            Symbol = Attr(el, "symbol"),
            AssetCategory = Attr(el, "assetCategory"),
            DateTime = ParseFlexDateTime(Attr(el, "dateTime")),
            Quantity = AttrDec(el, "quantity"),
            TradePrice = AttrDec(el, "tradePrice"),
            Commission = AttrDec(el, "ibCommission"),
            NetCash = AttrDec(el, "netCash"),
            BuySell = Attr(el, "buySell"),
            Strike = AttrDec(el, "strike"),
            Expiry = AttrOrNull(el, "expiry"),
            PutCall = AttrOrNull(el, "putCall"),
            Multiplier = AttrIntOrDefault(el, "multiplier", 1),
            Currency = AttrOrNull(el, "currency"),
            Exchange = AttrOrNull(el, "exchange"),
            OrderType = AttrOrNull(el, "orderType"),
            FxRateToBase = AttrDec(el, "fxRateToBase"),
            FifoPnlRealized = AttrDec(el, "fifoPnlRealized"),
            MtmPnl = AttrDec(el, "mtmPnl"),
            UnderlyingSymbol = AttrOrNull(el, "underlyingSymbol"),
            UnderlyingConid = AttrIntNullable(el, "underlyingConid"),
            OpenCloseIndicator = AttrOrNull(el, "openCloseIndicator"),
            Taxes = AttrDec(el, "taxes"),
        }).ToList();
    }

    public List<FlexPositionDto> ParseOpenPositions(string xml)
    {
        var doc = XDocument.Parse(xml);
        var positions = doc.Descendants("OpenPosition").ToList();
        _logger.LogInformation("Parsing {Count} open positions from Flex XML", positions.Count);

        return positions.Select(el => new FlexPositionDto
        {
            ConId = AttrInt(el, "conid"),
            Symbol = Attr(el, "symbol"),
            AssetCategory = Attr(el, "assetCategory"),
            Quantity = AttrDec(el, "position"),
            CostBasisPrice = AttrDec(el, "costBasisPrice"),
            CostBasisMoney = AttrDec(el, "costBasisMoney"),
            Strike = AttrDec(el, "strike"),
            Expiry = AttrOrNull(el, "expiry"),
            PutCall = AttrOrNull(el, "putCall"),
            Multiplier = AttrIntOrDefault(el, "multiplier", 1),
            FifoPnlUnrealized = AttrDec(el, "fifoPnlUnrealized"),
            Currency = AttrOrNull(el, "currency"),
            MarkPrice = AttrDec(el, "markPrice"),
            PercentOfNAV = AttrDec(el, "percentOfNAV"),
        }).ToList();
    }

    public List<FlexEquitySummaryDto> ParseEquitySummary(string xml)
    {
        var doc = XDocument.Parse(xml);
        var rows = doc.Descendants("EquitySummaryByReportDateInBase").ToList();
        _logger.LogInformation("Parsing {Count} equity summary rows from Flex XML", rows.Count);

        return rows.Select(el => new FlexEquitySummaryDto
        {
            ReportDate = ParseFlexDate(Attr(el, "reportDate")),
            Total = AttrDec(el, "total"),
            LongOptionValue = AttrDec(el, "optionsLong"),
            ShortOptionValue = AttrDec(el, "optionsShort"),
        }).ToList();
    }

    public List<FlexOptionEventDto> ParseOptionEvents(string xml)
    {
        var doc = XDocument.Parse(xml);
        var events = doc.Descendants("OptionEAE").ToList();
        _logger.LogInformation("Parsing {Count} option events from Flex XML", events.Count);

        return events.Select(el => new FlexOptionEventDto
        {
            ConId = AttrInt(el, "conid"),
            Symbol = Attr(el, "symbol"),
            AssetCategory = Attr(el, "assetCategory"),
            UnderlyingSymbol = AttrOrNull(el, "underlyingSymbol"),
            UnderlyingConid = AttrIntNullable(el, "underlyingConid"),
            TransactionType = Attr(el, "transactionType"),
            Date = ParseFlexDate(Attr(el, "date")),
            Quantity = AttrDec(el, "quantity"),
            TradePrice = AttrDec(el, "tradePrice"),
            RealizedPnl = AttrDec(el, "realizedPnl"),
            MtmPnl = AttrDec(el, "mtmPnl"),
            Commission = AttrDec(el, "commisionsAndTax"),
            Strike = AttrDec(el, "strike"),
            Expiry = AttrOrNull(el, "expiry"),
            PutCall = AttrOrNull(el, "putCall"),
            Multiplier = AttrIntOrDefault(el, "multiplier", 1),
            TradeId = AttrOrNull(el, "tradeID"),
        }).ToList();
    }

    /// <summary>Validates that the XML contains the expected Flex sections.</summary>
    public void ValidateSections(string xml)
    {
        var doc = XDocument.Parse(xml);
        var missing = new List<string>();

        if (!doc.Descendants("Trade").Any() && !doc.Descendants("Trades").Any())
            missing.Add("Trades");
        if (!doc.Descendants("OpenPosition").Any() && !doc.Descendants("OpenPositions").Any())
            missing.Add("OpenPositions");
        if (!doc.Descendants("EquitySummaryByReportDateInBase").Any() && !doc.Descendants("EquitySummaryInBase").Any())
            missing.Add("EquitySummaryInBase");
        if (!doc.Descendants("OptionEAE").Any())
            missing.Add("OptionEAE");

        if (missing.Count > 0)
            _logger.LogWarning("Flex XML missing sections: {Sections}. The query may not be configured correctly.", string.Join(", ", missing));
    }

    // ── Attribute helpers ────────────────────────────

    private static string Attr(XElement el, string name) =>
        el.Attribute(name)?.Value ?? "";

    private static string? AttrOrNull(XElement el, string name)
    {
        var val = el.Attribute(name)?.Value;
        return string.IsNullOrEmpty(val) ? null : val;
    }

    private static int AttrInt(XElement el, string name) =>
        int.TryParse(el.Attribute(name)?.Value, out var v) ? v : 0;

    private static int? AttrIntNullable(XElement el, string name) =>
        int.TryParse(el.Attribute(name)?.Value, out var v) ? v : null;

    private static int AttrIntOrDefault(XElement el, string name, int defaultValue) =>
        int.TryParse(el.Attribute(name)?.Value, out var v) ? v : defaultValue;

    private static decimal AttrDec(XElement el, string name) =>
        decimal.TryParse(el.Attribute(name)?.Value, NumberStyles.Any, CultureInfo.InvariantCulture, out var v) ? v : 0;

    private static DateTime ParseFlexDateTime(string value)
    {
        if (string.IsNullOrEmpty(value)) return default;
        return DateTime.TryParseExact(value, DateTimeFormats, CultureInfo.InvariantCulture,
            DateTimeStyles.None, out var dt) ? dt : default;
    }

    private static DateTime ParseFlexDate(string value)
    {
        if (string.IsNullOrEmpty(value)) return default;
        return DateTime.TryParseExact(value, DateFormats, CultureInfo.InvariantCulture,
            DateTimeStyles.None, out var dt) ? dt : default;
    }
}
