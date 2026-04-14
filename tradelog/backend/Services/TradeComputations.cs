using tradelog.Dtos;
using tradelog.Models;

namespace tradelog.Services;

public static class StockPositionComputations
{
    /// <summary>
    /// Walk through trades in chronological order, maintaining running position and avg price.
    /// PnL is realized only when reducing (sign of posChange != sign of lastPos).
    /// Trades MUST be pre-sorted by Symbol, Date, Id.
    /// </summary>
    public static List<StockPositionDto> ComputeRunningFields(List<StockPosition> trades)
    {
        var result = new List<StockPositionDto>(trades.Count);
        var state = new Dictionary<string, (int totalPos, decimal avgPrice)>();

        foreach (var t in trades)
        {
            var (prevTotal, prevAvg) = state.GetValueOrDefault(t.Symbol, (0, 0m));
            var lastPos = prevTotal;

            var newTotal = prevTotal + t.PosChange;
            decimal newAvg;
            decimal pnl = 0;

            if (prevTotal == 0)
            {
                newAvg = t.Price;
            }
            else if (Math.Sign(t.PosChange) == Math.Sign(prevTotal))
            {
                newAvg = (prevTotal * prevAvg + t.PosChange * t.Price) / (decimal)(prevTotal + t.PosChange);
            }
            else
            {
                pnl = t.PosChange * (prevAvg - t.Price) * t.Multiplier;
                newAvg = newTotal == 0 ? 0m : prevAvg;

                if (Math.Sign(newTotal) != Math.Sign(prevTotal) && newTotal != 0)
                    newAvg = t.Price;
            }

            state[t.Symbol] = (newTotal, newAvg);

            result.Add(new StockPositionDto
            {
                Id = t.Id,
                Symbol = t.Symbol,
                Date = t.Date,
                PosChange = t.PosChange,
                Price = t.Price,
                Commission = t.Commission,
                Multiplier = t.Multiplier,
                TradeId = t.TradeId,
                Notes = t.Notes,
                LastPos = lastPos,
                TotalPos = newTotal,
                AvgPrice = Math.Round(newAvg, 6),
                Pnl = Math.Round(pnl, 2),
            });
        }

        return result;
    }
}
