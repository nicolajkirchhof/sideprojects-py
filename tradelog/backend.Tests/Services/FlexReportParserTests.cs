using Microsoft.Extensions.Logging.Abstractions;
using tradelog.Services;

namespace tradelog.Tests.Services;

public class FlexReportParserTests
{
    private readonly FlexReportParser _parser = new(NullLogger<FlexReportParser>.Instance);

    // ── Date format tests ────────────────────────────

    [Fact]
    public void ParseTrades_IsoDateTime_ParsesCorrectly()
    {
        var xml = WrapInFlex("""<Trade tradeID="1" conid="123" symbol="AAPL" assetCategory="STK" dateTime="2026-03-30 11:35:55" quantity="10" tradePrice="150" ibCommission="-1.5" netCash="-1501.5" buySell="BUY" strike="" expiry="" putCall="" multiplier="1" currency="USD" exchange="NYSE" orderType="LMT" fxRateToBase="0.87" fifoPnlRealized="0" mtmPnl="-5" underlyingSymbol="" underlyingConid="" openCloseIndicator="O" taxes="0" />""");

        var trades = _parser.ParseTrades(xml);

        Assert.Single(trades);
        Assert.Equal(new DateTime(2026, 3, 30, 11, 35, 55), trades[0].DateTime);
    }

    [Fact]
    public void ParseTrades_CompactDateTime_ParsesCorrectly()
    {
        var xml = WrapInFlex("""<Trade tradeID="1" conid="123" symbol="AAPL" assetCategory="STK" dateTime="20260330;113555" quantity="10" tradePrice="150" ibCommission="-1.5" netCash="-1501.5" buySell="BUY" strike="" expiry="" putCall="" multiplier="1" currency="USD" exchange="" orderType="" fxRateToBase="0" fifoPnlRealized="0" mtmPnl="0" underlyingSymbol="" underlyingConid="" openCloseIndicator="" taxes="0" />""");

        var trades = _parser.ParseTrades(xml);

        Assert.Single(trades);
        Assert.Equal(new DateTime(2026, 3, 30, 11, 35, 55), trades[0].DateTime);
    }

    [Fact]
    public void ParseEquitySummary_IsoDate_ParsesCorrectly()
    {
        var xml = """<FlexQueryResponse><FlexStatements><FlexStatement><EquitySummaryInBase><EquitySummaryByReportDateInBase reportDate="2026-03-31" total="50000.50" optionsLong="3000" optionsShort="-2000" /></EquitySummaryInBase></FlexStatement></FlexStatements></FlexQueryResponse>""";

        var rows = _parser.ParseEquitySummary(xml);

        Assert.Single(rows);
        Assert.Equal(new DateTime(2026, 3, 31), rows[0].ReportDate);
        Assert.Equal(50000.50m, rows[0].Total);
        Assert.Equal(3000m, rows[0].LongOptionValue);
        Assert.Equal(-2000m, rows[0].ShortOptionValue);
    }

    [Fact]
    public void ParseEquitySummary_CompactDate_ParsesCorrectly()
    {
        var xml = """<FlexQueryResponse><FlexStatements><FlexStatement><EquitySummaryInBase><EquitySummaryByReportDateInBase reportDate="20260331" total="50000" optionsLong="0" optionsShort="0" /></EquitySummaryInBase></FlexStatement></FlexStatements></FlexQueryResponse>""";

        var rows = _parser.ParseEquitySummary(xml);

        Assert.Single(rows);
        Assert.Equal(new DateTime(2026, 3, 31), rows[0].ReportDate);
    }

    // ── Tier 1 field parsing ────────────────────────────

    [Fact]
    public void ParseTrades_Tier1Fields_ParsedCorrectly()
    {
        var xml = WrapInFlex("""<Trade tradeID="999" conid="42" symbol="SPY" assetCategory="STK" dateTime="2026-01-15 10:00:00" quantity="5" tradePrice="480" ibCommission="-1.05" netCash="-2401.05" buySell="BUY" strike="" expiry="" putCall="" multiplier="1" currency="USD" exchange="DARK" orderType="LMT" fxRateToBase="0.92" fifoPnlRealized="125.50" mtmPnl="-8.02" underlyingSymbol="SPY" underlyingConid="756733" openCloseIndicator="O" taxes="0.35" />""");

        var trades = _parser.ParseTrades(xml);
        var t = trades[0];

        Assert.Equal("DARK", t.Exchange);
        Assert.Equal("LMT", t.OrderType);
        Assert.Equal(0.92m, t.FxRateToBase);
        Assert.Equal(125.50m, t.FifoPnlRealized);
        Assert.Equal(-8.02m, t.MtmPnl);
        Assert.Equal("SPY", t.UnderlyingSymbol);
        Assert.Equal(756733, t.UnderlyingConid);
        Assert.Equal("O", t.OpenCloseIndicator);
        Assert.Equal(0.35m, t.Taxes);
    }

    // ── OptionEAE parsing ────────────────────────────

    [Fact]
    public void ParseOptionEvents_Assignment_ParsedCorrectly()
    {
        var xml = """
        <FlexQueryResponse><FlexStatements><FlexStatement>
            <OptionEAE conid="723197" symbol="PEP 155 P" assetCategory="OPT" underlyingSymbol="PEP" underlyingConid="11017" transactionType="Assignment" date="2025-04-14" quantity="1" tradePrice="0" realizedPnl="0" mtmPnl="1216.58" commisionsAndTax="0" strike="155" expiry="2025-04-17" putCall="P" multiplier="100" tradeID="100" />
            <OptionEAE conid="11017" symbol="PEP" assetCategory="STK" underlyingSymbol="PEP" underlyingConid="" transactionType="Buy" date="2025-04-14" quantity="100" tradePrice="155" realizedPnl="0" mtmPnl="-1216" commisionsAndTax="0" strike="" expiry="" putCall="" multiplier="1" tradeID="101" />
        </FlexStatement></FlexStatements></FlexQueryResponse>
        """;

        var events = _parser.ParseOptionEvents(xml);

        Assert.Equal(2, events.Count);

        var assignment = events[0];
        Assert.Equal("Assignment", assignment.TransactionType);
        Assert.Equal(723197, assignment.ConId);
        Assert.Equal("OPT", assignment.AssetCategory);
        Assert.Equal("PEP", assignment.UnderlyingSymbol);
        Assert.Equal(new DateTime(2025, 4, 14), assignment.Date);
        Assert.Equal(1216.58m, assignment.MtmPnl);

        // Stock delivery leg is also parsed (filtering happens in sync service)
        var stockLeg = events[1];
        Assert.Equal("Buy", stockLeg.TransactionType);
        Assert.Equal("STK", stockLeg.AssetCategory);
    }

    [Fact]
    public void ParseOptionEvents_Expiration_ParsedCorrectly()
    {
        var xml = """
        <FlexQueryResponse><FlexStatements><FlexStatement>
            <OptionEAE conid="900001" symbol="SPY 500 C" assetCategory="OPT" underlyingSymbol="SPY" underlyingConid="756733" transactionType="Expiration" date="2026-01-17" quantity="-1" tradePrice="0" realizedPnl="150" mtmPnl="0" commisionsAndTax="0" strike="500" expiry="2026-01-17" putCall="C" multiplier="100" tradeID="200" />
        </FlexStatement></FlexStatements></FlexQueryResponse>
        """;

        var events = _parser.ParseOptionEvents(xml);

        Assert.Single(events);
        Assert.Equal("Expiration", events[0].TransactionType);
        Assert.Equal(150m, events[0].RealizedPnl);
    }

    // ── Section validation ────────────────────────────

    [Fact]
    public void ValidateSections_AllPresent_NoWarning()
    {
        var xml = """
        <FlexQueryResponse><FlexStatements><FlexStatement>
            <Trades><Trade tradeID="1" /></Trades>
            <OpenPositions><OpenPosition conid="1" /></OpenPositions>
            <EquitySummaryInBase><EquitySummaryByReportDateInBase reportDate="2026-01-01" /></EquitySummaryInBase>
            <OptionEAE conid="1" transactionType="Expiration" />
        </FlexStatement></FlexStatements></FlexQueryResponse>
        """;

        // Should not throw — just verifies no exception
        _parser.ValidateSections(xml);
    }

    // ── Helper ────────────────────────────

    private static string WrapInFlex(string tradeElements) =>
        $"<FlexQueryResponse><FlexStatements><FlexStatement><Trades>{tradeElements}</Trades></FlexStatement></FlexStatements></FlexQueryResponse>";
}
