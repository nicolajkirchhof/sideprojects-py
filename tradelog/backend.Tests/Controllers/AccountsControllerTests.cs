using tradelog.Controllers;
using tradelog.Models;
using tradelog.Tests.Fixtures;
using Microsoft.AspNetCore.Mvc;

namespace tradelog.Tests.Controllers;

public class AccountsControllerTests : IDisposable
{
    private readonly TestDbFixture _fixture;

    public AccountsControllerTests()
    {
        _fixture = new TestDbFixture();
    }

    [Fact]
    public async Task Create_FirstAccount_IsDefault()
    {
        using var ctx = _fixture.CreateContext();
        var controller = new AccountsController(ctx);

        var account = new Account { IbkrAccountId = "U1234", Name = "Real", Host = "127.0.0.1", Port = 8497 };
        var result = await controller.Create(account);

        var created = (result.Result as CreatedAtActionResult)?.Value as Account;
        Assert.NotNull(created);
        Assert.True(created!.IsDefault);
    }

    [Fact]
    public async Task Create_SecondAccount_NotDefault()
    {
        using var ctx = _fixture.CreateContext();
        var controller = new AccountsController(ctx);

        await controller.Create(new Account { IbkrAccountId = "U1111", Name = "First" });
        var result = await controller.Create(new Account { IbkrAccountId = "U2222", Name = "Second" });

        var created = (result.Result as CreatedAtActionResult)?.Value as Account;
        Assert.NotNull(created);
        Assert.False(created!.IsDefault);
    }

    [Fact]
    public async Task GetAll_ReturnsAllAccounts()
    {
        using var ctx = _fixture.CreateContext();
        var controller = new AccountsController(ctx);

        await controller.Create(new Account { IbkrAccountId = "U1111", Name = "Alpha" });
        await controller.Create(new Account { IbkrAccountId = "U2222", Name = "Beta" });

        var result = await controller.GetAll();

        Assert.Equal(2, result.Value!.Count());
    }

    [Fact]
    public async Task Update_ChangesFields()
    {
        using var ctx = _fixture.CreateContext();
        var controller = new AccountsController(ctx);

        var createResult = await controller.Create(new Account { IbkrAccountId = "U1111", Name = "Old Name", Port = 7497 });
        var created = (createResult.Result as CreatedAtActionResult)?.Value as Account;

        created!.Name = "New Name";
        created.Port = 8497;
        await controller.Update(created.Id, created);

        var getResult = await controller.GetById(created.Id);
        var updated = getResult.Value!;
        Assert.Equal("New Name", updated.Name);
        Assert.Equal(8497, updated.Port);
    }

    [Fact]
    public async Task Delete_RemovesAccount()
    {
        using var ctx = _fixture.CreateContext();
        var controller = new AccountsController(ctx);

        var createResult = await controller.Create(new Account { IbkrAccountId = "U1111", Name = "ToDelete" });
        var created = (createResult.Result as CreatedAtActionResult)?.Value as Account;

        var deleteResult = await controller.Delete(created!.Id);
        Assert.IsType<NoContentResult>(deleteResult);

        var getResult = await controller.GetById(created.Id);
        Assert.IsType<NotFoundResult>(getResult.Result);
    }

    [Fact]
    public async Task DuplicateIbkrAccountId_ThrowsOnSave()
    {
        using var ctx = _fixture.CreateContext();
        var controller = new AccountsController(ctx);

        await controller.Create(new Account { IbkrAccountId = "U1111", Name = "First" });

        // SQLite enforces the unique index
        await Assert.ThrowsAnyAsync<Exception>(async () =>
            await controller.Create(new Account { IbkrAccountId = "U1111", Name = "Duplicate" }));
    }

    public void Dispose() => _fixture.Dispose();
}
