using tradelog.Controllers;
using tradelog.Models;
using tradelog.Tests.Fixtures;
using Microsoft.AspNetCore.Mvc;

namespace tradelog.Tests.Controllers;

public class CapitalControllerTests : IDisposable
{
    private readonly TestDbFixture _fixture;

    public CapitalControllerTests()
    {
        _fixture = new TestDbFixture();
        SeedAccount();
    }

    private void SeedAccount()
    {
        using var ctx = _fixture.CreateContext();
        ctx.Accounts.Add(new Account { Id = _fixture.TestAccountId, IbkrAccountId = "U1234", Name = "Test" });
        ctx.SaveChanges();
    }

    [Fact]
    public async Task Create_StoresCapitalSnapshot()
    {
        using var ctx = _fixture.CreateContext();
        var controller = new CapitalController(ctx);

        var capital = new Capital
        {
            Date = new(2025, 6, 15),
            NetLiquidity = 100000m,
            Maintenance = 30000m,
            ExcessLiquidity = 70000m,
            Bpr = 200000m,
        };

        var result = await controller.Create(capital);
        var created = (result.Result as CreatedAtActionResult)?.Value as Capital;

        Assert.NotNull(created);
        Assert.Equal(30m, created!.MaintenancePct); // 30000/100000 * 100
    }

    [Fact]
    public async Task GetAll_ReturnsDescendingByDate()
    {
        using (var ctx = _fixture.CreateContext())
        {
            ctx.Capitals.Add(new Capital { Date = new(2025, 1, 1), NetLiquidity = 90000m, AccountId = _fixture.TestAccountId });
            ctx.Capitals.Add(new Capital { Date = new(2025, 1, 3), NetLiquidity = 110000m, AccountId = _fixture.TestAccountId });
            ctx.Capitals.Add(new Capital { Date = new(2025, 1, 2), NetLiquidity = 100000m, AccountId = _fixture.TestAccountId });
            await ctx.SaveChangesAsync();
        }

        using var queryCtx = _fixture.CreateContext();
        var controller = new CapitalController(queryCtx);

        var result = await controller.GetAll();
        var capitals = result.Value!.ToList();

        Assert.Equal(3, capitals.Count);
        Assert.Equal(new(2025, 1, 3), capitals[0].Date);
        Assert.Equal(new(2025, 1, 2), capitals[1].Date);
        Assert.Equal(new(2025, 1, 1), capitals[2].Date);
    }

    [Fact]
    public async Task Delete_RemovesCapital()
    {
        using var ctx = _fixture.CreateContext();
        var controller = new CapitalController(ctx);

        var capital = new Capital { Date = new(2025, 6, 15), NetLiquidity = 100000m };
        var createResult = await controller.Create(capital);
        var created = (createResult.Result as CreatedAtActionResult)?.Value as Capital;

        var deleteResult = await controller.Delete(created!.Id);
        Assert.IsType<NoContentResult>(deleteResult);

        var getResult = await controller.GetById(created.Id);
        Assert.IsType<NotFoundResult>(getResult.Result);
    }

    public void Dispose() => _fixture.Dispose();
}
