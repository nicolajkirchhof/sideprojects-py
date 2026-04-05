namespace tradelog.Models;

public enum Budget
{
    LongTerm,
    Drift,
    Swing,
    Speculative
}

public enum Strategy
{
    PositiveDrift,
    RangeBound,
    PEAD,
    BreakoutMomentum,
    IVMeanReversion,
    SectorStrength,
    SectorWeakness,
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
    ShortStock,
    IronCondor,
    XYZ,
    PMCC,
    CalendarSpread
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

public enum TradeEventType
{
    ScaleIn,
    ProfitTake,
    Roll,
    Stop
}
