using IBApi;
using tradelog.Services;

namespace tradelog.Tests.Services;

public class IbkrConnectionManagerErrorTests
{
    private readonly IbkrConnectionManager _sut = new();

    [Theory]
    [InlineData(2104, "Market data farm connection is OK:usfarm.nj")]
    [InlineData(2106, "HMDS data farm connection is OK:ushmds")]
    [InlineData(2158, "Sec-def data farm connection is OK:secdefeu")]
    [InlineData(2100, "API client has been unsubscribed from account data")]
    [InlineData(2199, "Upper bound of info range")]
    public void InformationalCodes_AreIgnored(int code, string message)
    {
        _sut.error(-1, 0, code, message, "");

        Assert.Null(_sut._connectionError);
    }

    [Theory]
    [InlineData(1100, "Connectivity between IB and TWS has been lost")]
    [InlineData(504, "Not connected")]
    [InlineData(502, "Couldn't connect to TWS")]
    public void RealConnectionErrors_AreStored(int code, string message)
    {
        _sut.error(-1, 0, code, message, "");

        Assert.Equal(message, _sut._connectionError);
    }

    [Fact]
    public void ErrorForAccountSummaryReqId_FaultsTcs()
    {
        var tcs = new TaskCompletionSource<Dictionary<string, string>>();
        _sut._accountSummaryTcs = tcs;
        _sut._accountSummaryReqId = 5;

        _sut.error(5, 0, 321, "Error validating request", "");

        Assert.True(tcs.Task.IsFaulted);
        Assert.Contains("321", tcs.Task.Exception!.InnerException!.Message);
    }

    [Fact]
    public void ErrorForExecutionsReqId_FaultsTcs()
    {
        var tcs = new TaskCompletionSource<List<ExecutionMessage>>();
        _sut._executionsTcs = tcs;
        _sut._executionsReqId = 7;

        _sut.error(7, 0, 321, "Error validating request", "");

        Assert.True(tcs.Task.IsFaulted);
        Assert.Contains("321", tcs.Task.Exception!.InnerException!.Message);
    }

    [Fact]
    public void ErrorForMarketDataReqId_FaultsTcs()
    {
        var tcs = new TaskCompletionSource<MarketDataSnapshot>();
        _sut._mktDataTcs[10] = tcs;

        _sut.error(10, 0, 200, "No security definition found", "");

        Assert.True(tcs.Task.IsFaulted);
        Assert.Contains("200", tcs.Task.Exception!.InnerException!.Message);
    }

    [Fact]
    public void ErrorForContractDetailsReqId_FaultsTcs()
    {
        var tcs = new TaskCompletionSource<ContractDetails>();
        _sut._contractDetailsTcs[12] = tcs;

        _sut.error(12, 0, 200, "No security definition found", "");

        Assert.True(tcs.Task.IsFaulted);
    }

    [Fact]
    public void ErrorForWhatIfReqId_FaultsTcs()
    {
        var tcs = new TaskCompletionSource<OrderState>();
        _sut._whatIfTcs[15] = tcs;

        _sut.error(15, 0, 200, "No security definition found", "");

        Assert.True(tcs.Task.IsFaulted);
    }

    [Fact]
    public void ErrorForUnknownReqId_DoesNotThrow()
    {
        // Should log but not throw
        var ex = Record.Exception(() => _sut.error(999, 0, 200, "Unknown request", ""));

        Assert.Null(ex);
    }

    [Fact]
    public void InformationalCode_DoesNotFaultActiveTcs()
    {
        var tcs = new TaskCompletionSource<Dictionary<string, string>>();
        _sut._accountSummaryTcs = tcs;
        _sut._accountSummaryReqId = 5;

        // Even if id matches, informational codes should be ignored
        _sut.error(5, 0, 2104, "Market data farm OK", "");

        Assert.False(tcs.Task.IsFaulted);
        Assert.False(tcs.Task.IsCompleted);
    }
}
