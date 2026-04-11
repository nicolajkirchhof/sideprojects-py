namespace tradelog.Models;

// Budget, Strategy, TypeOfTrade, DirectionalBias, Timeframe, ManagementRating
// are now DB-backed lookup values — see LookupValue.cs and LookupCategory.

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
