-- ==========================================================
-- Reset all data for an IBKR account, preserving the Account row
-- Usage: Set @IbkrAccountId below, then run against tradelog DB
-- ==========================================================

SET NOCOUNT ON;
BEGIN TRANSACTION;

-- ========== CONFIGURE HERE ==========
DECLARE @IbkrAccountId NVARCHAR(20) = N'8497';
-- =====================================

DECLARE @AccountId INT;
SELECT @AccountId = Id FROM Accounts WHERE IbkrAccountId = @IbkrAccountId;

IF @AccountId IS NULL
BEGIN
    PRINT 'ERROR: No account found for IbkrAccountId ' + @IbkrAccountId;
    ROLLBACK;
    RETURN;
END

PRINT 'Resetting data for AccountId ' + CAST(@AccountId AS VARCHAR) + ' (IBKR: ' + @IbkrAccountId + ')';

-- Delete in FK-safe order
DELETE FROM OptionPositionsLogs WHERE AccountId = @AccountId;
PRINT '  OptionPositionsLogs: ' + CAST(@@ROWCOUNT AS VARCHAR) + ' rows deleted';

DELETE FROM OptionPositions WHERE AccountId = @AccountId;
PRINT '  OptionPositions:     ' + CAST(@@ROWCOUNT AS VARCHAR) + ' rows deleted';

DELETE FROM Trades WHERE AccountId = @AccountId;
PRINT '  Trades:              ' + CAST(@@ROWCOUNT AS VARCHAR) + ' rows deleted';

DELETE FROM TradeEntries WHERE AccountId = @AccountId;
PRINT '  TradeEntries:        ' + CAST(@@ROWCOUNT AS VARCHAR) + ' rows deleted';

DELETE FROM Capitals WHERE AccountId = @AccountId;
PRINT '  Capitals:            ' + CAST(@@ROWCOUNT AS VARCHAR) + ' rows deleted';

DELETE FROM WeeklyPreps WHERE AccountId = @AccountId;
PRINT '  WeeklyPreps:         ' + CAST(@@ROWCOUNT AS VARCHAR) + ' rows deleted';

DELETE FROM Portfolios WHERE AccountId = @AccountId;
PRINT '  Portfolios:          ' + CAST(@@ROWCOUNT AS VARCHAR) + ' rows deleted';

DELETE FROM StockPriceCaches;
PRINT '  StockPriceCaches:    ' + CAST(@@ROWCOUNT AS VARCHAR) + ' rows deleted';

-- Reset sync state
UPDATE Accounts
SET LastSyncAt = NULL,
    LastSyncResult = NULL
WHERE Id = @AccountId;

PRINT '';
PRINT 'Account reset complete. Account row preserved with cleared sync state.';

COMMIT;
