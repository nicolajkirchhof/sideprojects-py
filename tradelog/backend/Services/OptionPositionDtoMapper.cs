using tradelog.Dtos;
using tradelog.Models;

namespace tradelog.Services;

public static class OptionPositionDtoMapper
{
    public static OptionPositionDto ToDto(OptionPosition p, OptionPositionsLog? log)
    {
        var isOpen = p.Closed == null;
        var today = DateTime.UtcNow.Date;

        decimal? lastPrice = isOpen ? log?.Price : null;
        decimal? margin = log?.Margin;

        decimal? unrealizedPnl = null;
        decimal? unrealizedPnlPct = null;
        decimal? realizedPnl = null;
        decimal? realizedPnlPct = null;

        if (isOpen && lastPrice.HasValue)
        {
            unrealizedPnl = p.Pos * (lastPrice.Value - p.Cost);
            unrealizedPnlPct = p.Cost != 0
                ? Math.Round(unrealizedPnl.Value / p.Cost, 2) * 100
                : null;
        }

        if (!isOpen && p.ClosePrice.HasValue)
        {
            realizedPnl = (p.ClosePrice.Value - p.Cost) * p.Multiplier * p.Pos - p.Commission;
            realizedPnlPct = (p.Cost * p.Multiplier) != 0
                ? Math.Round(realizedPnl.Value / (p.Cost * p.Multiplier), 1) * 100
                : null;
        }

        var endDate = p.Closed ?? today;
        var totalSpan = (p.Expiry - p.Opened).TotalDays;
        var elapsedSpan = (endDate - p.Opened).TotalDays;
        decimal? durationPct = totalSpan > 0 ? (decimal)(elapsedSpan / totalSpan * 100) : null;

        decimal? roic = (realizedPnl.HasValue && margin.HasValue && margin.Value != 0)
            ? realizedPnl.Value * 100 / margin.Value
            : null;

        return new OptionPositionDto
        {
            Id = p.Id,
            Symbol = p.Symbol,
            ContractId = p.ContractId,
            Opened = p.Opened,
            Expiry = p.Expiry,
            Closed = p.Closed,
            Pos = p.Pos,
            Right = p.Right,
            Strike = p.Strike,
            Cost = p.Cost,
            ClosePrice = p.ClosePrice,
            Commission = p.Commission,
            Multiplier = p.Multiplier,
            CloseReasons = p.CloseReasons,
            LastPrice = lastPrice,
            LastValue = lastPrice.HasValue ? lastPrice.Value * p.Multiplier : null,
            TimeValue = isOpen ? log?.TimeValue : null,
            Delta = isOpen ? (log?.Delta * p.Pos) : null,
            Theta = isOpen ? (log?.Theta * p.Pos) : null,
            Gamma = isOpen ? log?.Gamma : null,
            Vega = isOpen ? log?.Vega : null,
            Iv = isOpen ? (log?.Iv * 100) : null,
            Margin = margin,
            UnrealizedPnl = unrealizedPnl,
            UnrealizedPnlPct = unrealizedPnlPct,
            RealizedPnl = realizedPnl,
            RealizedPnlPct = realizedPnlPct,
            DurationPct = durationPct,
            Roic = roic,
            LogCount = p.LogCount,
            TradeId = p.TradeId,
            Notes = p.Notes,
        };
    }
}
