using IBApi;

namespace tradelog.Services;

/// <summary>
/// Manages a TWS API socket connection and wraps callback-based API into async methods.
/// Inherits from DefaultEWrapper so only the callbacks we care about need overriding.
/// Each sync creates a fresh connection: Connect → do work → Disconnect.
/// </summary>
public class IbkrConnectionManager : DefaultEWrapper, IDisposable
{
    private EClientSocket? _client;
    private EReaderSignal? _signal;
    private Thread? _readerThread;
    private int _nextReqId = 1;

    // Account summary
    private TaskCompletionSource<Dictionary<string, string>>? _accountSummaryTcs;
    private Dictionary<string, string>? _accountSummaryData;

    // Positions
    private TaskCompletionSource<List<PortfolioItem>>? _positionsTcs;
    private List<PortfolioItem>? _positionsData;

    // Executions
    private TaskCompletionSource<List<ExecutionMessage>>? _executionsTcs;
    private List<ExecutionMessage>? _executionsData;

    // Market data (per request)
    private readonly Dictionary<int, TaskCompletionSource<MarketDataSnapshot>> _mktDataTcs = new();
    private readonly Dictionary<int, MarketDataSnapshot> _mktDataSnapshots = new();

    // Contract details
    private readonly Dictionary<int, TaskCompletionSource<ContractDetails>> _contractDetailsTcs = new();

    // What-if order (margin)
    private readonly Dictionary<int, TaskCompletionSource<OrderState>> _whatIfTcs = new();

    // Connection error
    private string? _connectionError;

    // Managed accounts (received on connect)
    public List<string> ManagedAccounts { get; private set; } = new();

    public bool IsConnected => _client?.IsConnected() ?? false;

    public void Connect(string host, int port, int clientId)
    {
        _connectionError = null;
        _signal = new EReaderMonitorSignal();
        _client = new EClientSocket(this, _signal);
        _client.eConnect(host, port, clientId);

        if (!_client.IsConnected())
            throw new InvalidOperationException($"Failed to connect to TWS at {host}:{port}");

        var reader = new EReader(_client, _signal);
        reader.Start();

        _readerThread = new Thread(() =>
        {
            while (_client.IsConnected())
            {
                _signal.waitForSignal();
                reader.processMsgs();
            }
        })
        { IsBackground = true };
        _readerThread.Start();

        // Give TWS a moment to send initial messages
        Thread.Sleep(1000);

        if (_connectionError != null)
            throw new InvalidOperationException($"TWS connection error: {_connectionError}");
    }

    public void Disconnect()
    {
        _client?.eDisconnect();
    }

    public void Dispose()
    {
        Disconnect();
    }

    // ────────────────────────────────────────────
    // Async request methods
    // ────────────────────────────────────────────

    public async Task<Dictionary<string, string>> RequestAccountSummaryAsync(TimeSpan? timeout = null)
    {
        _accountSummaryData = new Dictionary<string, string>();
        _accountSummaryTcs = new TaskCompletionSource<Dictionary<string, string>>();
        var reqId = _nextReqId++;
        _client!.reqAccountSummary(reqId, "All", "NetLiquidation,MaintMarginReq,AvailableFunds,BuyingPower");
        var result = await WaitWithTimeout(_accountSummaryTcs.Task, timeout ?? TimeSpan.FromSeconds(10));
        _client.cancelAccountSummary(reqId);
        return result;
    }

    public async Task<List<PortfolioItem>> RequestPositionsAsync(TimeSpan? timeout = null)
    {
        _positionsData = new List<PortfolioItem>();
        _positionsTcs = new TaskCompletionSource<List<PortfolioItem>>();
        _client!.reqPositions();
        var result = await WaitWithTimeout(_positionsTcs.Task, timeout ?? TimeSpan.FromSeconds(15));
        _client.cancelPositions();
        return result;
    }

    public async Task<List<ExecutionMessage>> RequestExecutionsAsync(string? sinceDateTime = null, TimeSpan? timeout = null)
    {
        _executionsData = new List<ExecutionMessage>();
        _executionsTcs = new TaskCompletionSource<List<ExecutionMessage>>();
        var reqId = _nextReqId++;
        var filter = new ExecutionFilter { Time = sinceDateTime ?? "" };
        _client!.reqExecutions(reqId, filter);
        return await WaitWithTimeout(_executionsTcs.Task, timeout ?? TimeSpan.FromSeconds(15));
    }

    public async Task<MarketDataSnapshot> RequestMarketDataAsync(Contract contract, TimeSpan? timeout = null)
    {
        var reqId = _nextReqId++;
        var tcs = new TaskCompletionSource<MarketDataSnapshot>();
        _mktDataTcs[reqId] = tcs;
        _mktDataSnapshots[reqId] = new MarketDataSnapshot();

        // Request frozen data (type 2) — works outside market hours
        _client!.reqMarketDataType(2);
        _client.reqMktData(reqId, contract, "", true, false, []);

        try
        {
            return await WaitWithTimeout(tcs.Task, timeout ?? TimeSpan.FromSeconds(15));
        }
        finally
        {
            _client.cancelMktData(reqId);
            _mktDataTcs.Remove(reqId);
            _mktDataSnapshots.Remove(reqId);
        }
    }

    public async Task<ContractDetails> RequestContractDetailsAsync(Contract contract, TimeSpan? timeout = null)
    {
        var reqId = _nextReqId++;
        var tcs = new TaskCompletionSource<ContractDetails>();
        _contractDetailsTcs[reqId] = tcs;
        _client!.reqContractDetails(reqId, contract);

        try
        {
            return await WaitWithTimeout(tcs.Task, timeout ?? TimeSpan.FromSeconds(10));
        }
        finally
        {
            _contractDetailsTcs.Remove(reqId);
        }
    }

    public async Task<OrderState> RequestWhatIfOrderAsync(Contract contract, Order order, TimeSpan? timeout = null)
    {
        var orderId = _nextReqId++;
        var tcs = new TaskCompletionSource<OrderState>();
        _whatIfTcs[orderId] = tcs;
        order.WhatIf = true;
        _client!.placeOrder(orderId, contract, order);

        try
        {
            return await WaitWithTimeout(tcs.Task, timeout ?? TimeSpan.FromSeconds(10));
        }
        finally
        {
            _whatIfTcs.Remove(orderId);
        }
    }

    /// <summary>Get the last traded price for a stock/futures contract.</summary>
    public async Task<decimal> RequestLastPriceAsync(Contract contract, TimeSpan? timeout = null)
    {
        var snapshot = await RequestMarketDataAsync(contract, timeout);
        if (snapshot.Last > 0) return snapshot.Last;
        if (snapshot.Close > 0) return snapshot.Close;
        if (snapshot.Bid > 0 && snapshot.Ask > 0) return (snapshot.Bid + snapshot.Ask) / 2;
        return 0;
    }

    private static async Task<T> WaitWithTimeout<T>(Task<T> task, TimeSpan timeout)
    {
        if (await Task.WhenAny(task, Task.Delay(timeout)) == task)
            return await task;
        throw new TimeoutException($"TWS request timed out after {timeout.TotalSeconds}s");
    }

    // ────────────────────────────────────────────
    // EWrapper callback overrides (only the ones we use)
    // ────────────────────────────────────────────

    public override void accountSummary(int reqId, string account, string tag, string value, string currency)
    {
        _accountSummaryData?.TryAdd(tag, value);
    }

    public override void accountSummaryEnd(int reqId)
    {
        _accountSummaryTcs?.TrySetResult(_accountSummaryData ?? new());
    }

    public override void position(string account, Contract contract, decimal pos, double avgCost)
    {
        _positionsData?.Add(new PortfolioItem
        {
            Account = account,
            Contract = contract,
            Position = pos,
            AvgCost = avgCost,
        });
    }

    public override void positionEnd()
    {
        _positionsTcs?.TrySetResult(_positionsData ?? new());
    }

    public override void execDetails(int reqId, Contract contract, Execution execution)
    {
        _executionsData?.Add(new ExecutionMessage(reqId, contract, execution));
    }

    public override void execDetailsEnd(int reqId)
    {
        _executionsTcs?.TrySetResult(_executionsData ?? new());
    }

    public override void tickPrice(int tickerId, int field, double price, TickAttrib attribs)
    {
        if (!_mktDataSnapshots.TryGetValue(tickerId, out var snap)) return;

        switch (field)
        {
            case 1: snap.Bid = (decimal)price; break;    // BID
            case 2: snap.Ask = (decimal)price; break;    // ASK
            case 4: snap.Last = (decimal)price; break;   // LAST
            case 9: snap.Close = (decimal)price; break;  // CLOSE
        }
    }

    public override void tickOptionComputation(int tickerId, int field, int tickAttrib,
        double impliedVolatility, double delta, double optPrice, double pvDividend,
        double gamma, double vega, double theta, double undPrice)
    {
        if (!_mktDataSnapshots.TryGetValue(tickerId, out var snap)) return;

        // field 13 = model, field 10 = bid, field 11 = ask, field 12 = last
        if (field is 13 or 12 or 10 or 11)
        {
            // Prefer last (12), then model (13)
            if (field == 12 || !snap.HasGreeks)
            {
                snap.Delta = delta;
                snap.Gamma = gamma;
                snap.Vega = vega;
                snap.Theta = theta;
                snap.ImpliedVol = impliedVolatility;
                snap.UnderlyingPrice = undPrice;
                snap.HasGreeks = true;
            }
        }
    }

    public override void tickSnapshotEnd(int tickerId)
    {
        if (_mktDataTcs.TryGetValue(tickerId, out var tcs))
        {
            _mktDataSnapshots.TryGetValue(tickerId, out var snap);
            tcs.TrySetResult(snap ?? new MarketDataSnapshot());
        }
    }

    public override void contractDetails(int reqId, ContractDetails contractDetails)
    {
        if (_contractDetailsTcs.TryGetValue(reqId, out var tcs))
            tcs.TrySetResult(contractDetails);
    }

    public override void openOrder(int orderId, Contract contract, Order order, OrderState orderState)
    {
        if (order.WhatIf && _whatIfTcs.TryGetValue(orderId, out var tcs))
            tcs.TrySetResult(orderState);
    }

    public override void error(int id, long errorTime, int errorCode, string errorMsg, string advancedOrderRejectJson)
    {
        // Connection-level errors
        if (id == -1)
        {
            _connectionError = errorMsg;
            return;
        }

        // Not-critical informational messages (2100-2199 range)
        if (errorCode >= 2100 && errorCode <= 2199)
            return;

        // Fail any pending TCS for this request ID
        var ex = new InvalidOperationException($"TWS error {errorCode}: {errorMsg}");

        if (_mktDataTcs.TryGetValue(id, out var mktTcs))
            mktTcs.TrySetException(ex);
        if (_contractDetailsTcs.TryGetValue(id, out var cdTcs))
            cdTcs.TrySetException(ex);
        if (_whatIfTcs.TryGetValue(id, out var wiTcs))
            wiTcs.TrySetException(ex);
    }

    public override void nextValidId(int orderId)
    {
        _nextReqId = Math.Max(_nextReqId, orderId);
    }

    public override void connectAck()
    {
        if (_client?.AsyncEConnect == true) _client.startApi();
    }

    public override void managedAccounts(string accountsList)
    {
        ManagedAccounts = accountsList.Split(',', StringSplitOptions.RemoveEmptyEntries)
            .Select(a => a.Trim())
            .ToList();
    }
}

// ────────────────────────────────────────────
// Helper types
// ────────────────────────────────────────────

public class PortfolioItem
{
    public string Account { get; set; } = "";
    public Contract Contract { get; set; } = new();
    public decimal Position { get; set; }
    public double AvgCost { get; set; }
}

public class ExecutionMessage
{
    public int ReqId { get; set; }
    public Contract Contract { get; set; }
    public Execution Execution { get; set; }

    public ExecutionMessage(int reqId, Contract contract, Execution execution)
    {
        ReqId = reqId;
        Contract = contract;
        Execution = execution;
    }
}

public class MarketDataSnapshot
{
    public decimal Bid { get; set; }
    public decimal Ask { get; set; }
    public decimal Last { get; set; }
    public decimal Close { get; set; }
    public double Delta { get; set; }
    public double Gamma { get; set; }
    public double Vega { get; set; }
    public double Theta { get; set; }
    public double ImpliedVol { get; set; }
    public double UnderlyingPrice { get; set; }
    public bool HasGreeks { get; set; }
}
