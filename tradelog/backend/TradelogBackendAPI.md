# Tradelog Backend (.NET)

This project contains the backend API for the Tradelog application, built with ASP.NET Core.

## Prerequisites

*   [.NET 9.0 SDK](https://dotnet.microsoft.com/download/dotnet/9.0) or later.
*   A running SQL Server instance.

## Getting Started

1.  **Clone the repository.**
2.  **Navigate to the backend directory:**
    ```shell
    cd tradelog/backend.net
    ```
3.  **Restore dependencies:**
    ```shell
    dotnet restore
    ```
4.  **Configure your connection string:**
    Update the `ConnectionStrings:DefaultConnection` value in `appsettings.json` to point to your SQL Server instance.

## Entity Framework Core Migrations

This project uses EF Core to manage the database schema. The following commands should be run from the `tradelog/backend.net` directory.

### Creating a new migration

When you change your model classes, you'll need to create a new migration to apply those changes to the database.

```shell
dotnet ef migrations add <YourMigrationName>
```
