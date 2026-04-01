namespace tradelog.Dtos;

public class FlexTradeDto
{
    public string TradeId { get; set; } = "";
    public int ConId { get; set; }
    public string Symbol { get; set; } = "";
    public string AssetCategory { get; set; } = ""; // STK, FUT, OPT, FOP
    public DateTime DateTime { get; set; }
    public decimal Quantity { get; set; }
    public decimal TradePrice { get; set; }
    public decimal Commission { get; set; }
    public decimal NetCash { get; set; }
    public string BuySell { get; set; } = ""; // BUY, SELL
    public decimal Strike { get; set; }
    public string? Expiry { get; set; }
    public string? PutCall { get; set; } // P, C
    public int Multiplier { get; set; } = 1;
    public string? Currency { get; set; }

    // Tier 1 fields
    public string? Exchange { get; set; }
    public string? OrderType { get; set; }
    public decimal FxRateToBase { get; set; }
    public decimal FifoPnlRealized { get; set; }
    public decimal MtmPnl { get; set; }
    public string? UnderlyingSymbol { get; set; }
    public int? UnderlyingConid { get; set; }
    public string? OpenCloseIndicator { get; set; } // O, C
    public decimal Taxes { get; set; }
}

public class FlexPositionDto
{
    public int ConId { get; set; }
    public string Symbol { get; set; } = "";
    public string AssetCategory { get; set; } = "";
    public decimal Quantity { get; set; }
    public decimal CostBasisPrice { get; set; }
    public decimal CostBasisMoney { get; set; }
    public decimal Strike { get; set; }
    public string? Expiry { get; set; }
    public string? PutCall { get; set; }
    public int Multiplier { get; set; } = 1;
    public decimal FifoPnlUnrealized { get; set; }
    public string? Currency { get; set; }
    public decimal MarkPrice { get; set; }
    public decimal PercentOfNAV { get; set; }
}

public class FlexEquitySummaryDto
{
    public DateTime ReportDate { get; set; }
    public decimal Total { get; set; } // Net Liquidation
    public decimal LongOptionValue { get; set; }
    public decimal ShortOptionValue { get; set; }
}

public class FlexOptionEventDto
{
    public int ConId { get; set; }
    public string Symbol { get; set; } = "";
    public string AssetCategory { get; set; } = ""; // OPT, FOP, STK
    public string? UnderlyingSymbol { get; set; }
    public int? UnderlyingConid { get; set; }
    public string TransactionType { get; set; } = ""; // Assignment, Exercise, Expiration, Buy, Sell
    public DateTime Date { get; set; }
    public decimal Quantity { get; set; }
    public decimal TradePrice { get; set; }
    public decimal RealizedPnl { get; set; }
    public decimal MtmPnl { get; set; }
    public decimal Commission { get; set; }
    public decimal Strike { get; set; }
    public string? Expiry { get; set; }
    public string? PutCall { get; set; }
    public int Multiplier { get; set; } = 1;
    public string? TradeId { get; set; }
}
