namespace tradelog.Models;

public enum Budget
{
    Core,
    Speculative
}

public enum Strategy
{
    PositiveDrift,
    RangeBoundCommodities,
    PEADS,
    Momentum,
    IVMeanReversion,
    SectorStrength,
    SectorWeakness,
    Breakout,
    GreenLineBreakout,
    Slingshot,
    PreEarnings
}

public enum TypeOfTrade
{
    ShortStrangle,
    ShortPutSpread,
    ShortCallSpread,
    LongCall,
    LongPut,
    LongCallVertical,
    LongPutVertical,
    SyntheticLong,
    CoveredStrangle,
    Butterfly,
    RatioDiagonalSpread,
    LongStrangle,
    ShortPut,
    ShortCall,
    LongStock,
    ShortStock
}

public enum DirectionalBias
{
    Bullish,
    Neutral,
    Bearish
}

public enum Timeframe
{
    OneDay,
    OneWeek,
    DeltaBand
}

public enum ManagementRating
{
    AsPlanned,
    MinorAdjustments,
    Rogue
}

public enum PositionRight
{
    Call,
    Put
}

[Flags]
public enum CloseReasons
{
    None = 0,
    TakeLoss = 1,
    TakeProfit = 2,
    Roll = 4,
    AssumptionInvalidated = 8,
    TimeLimit = 16,
    Other = 32
}
