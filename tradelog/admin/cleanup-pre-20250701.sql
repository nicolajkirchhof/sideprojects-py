-- ==========================================================
-- One-off cleanup: delete every row with date < 2025-07-01
-- across all time-dependent tables, scoped to a single account.
--
-- Usage:
--     sqlcmd -S ocin.database.windows.net -d tradelog_stage \
--            -U ocin -P <pwd> -C -l 120 \
--            -v IbkrAccountId=U16408041 \
--            -i tradelog/admin/cleanup-pre-20250701.sql
--
-- FK-safe deletion order: children before parents.
-- ==========================================================

SET NOCOUNT ON;
BEGIN TRANSACTION;

DECLARE @AccountId INT;
DECLARE @Cutoff DATETIME2 = '2025-07-01';

SELECT @AccountId = Id FROM Accounts WHERE IbkrAccountId = N'$(IbkrAccountId)';

IF @AccountId IS NULL
BEGIN
    PRINT 'ERROR: No account found for IbkrAccountId $(IbkrAccountId)';
    ROLLBACK;
    RETURN;
END

PRINT 'Cleaning account ' + CAST(@AccountId AS VARCHAR) + ' — cutoff ' + CONVERT(VARCHAR, @Cutoff, 23);

-- 1. OptionPositionsLogs (references OptionPositions via ContractId, not FK — safe to delete directly)
DELETE FROM OptionPositionsLogs
WHERE AccountId = @AccountId AND DateTime < @Cutoff;
PRINT '  OptionPositionsLogs: ' + CAST(@@ROWCOUNT AS VARCHAR) + ' rows deleted';

-- 2. TradeEvents (FK → Trades, CASCADE) — delete directly for dates before cutoff
DELETE FROM TradeEvents
WHERE AccountId = @AccountId AND Date < @Cutoff;
PRINT '  TradeEvents: ' + CAST(@@ROWCOUNT AS VARCHAR) + ' rows deleted';

-- 3. OptionPositions — FK to Trades is ON DELETE SET NULL, so no cascade issue
DELETE FROM OptionPositions
WHERE AccountId = @AccountId AND Opened < @Cutoff;
PRINT '  OptionPositions: ' + CAST(@@ROWCOUNT AS VARCHAR) + ' rows deleted';

-- 4. StockPositions — same FK relationship to Trades (SET NULL)
DELETE FROM StockPositions
WHERE AccountId = @AccountId AND Date < @Cutoff;
PRINT '  StockPositions: ' + CAST(@@ROWCOUNT AS VARCHAR) + ' rows deleted';

-- 5. Trades (thesis) — must come after OptionPositions/StockPositions since those reference it
DELETE FROM Trades
WHERE AccountId = @AccountId AND Date < @Cutoff;
PRINT '  Trades: ' + CAST(@@ROWCOUNT AS VARCHAR) + ' rows deleted';

-- 6. Capitals
DELETE FROM Capitals
WHERE AccountId = @AccountId AND Date < @Cutoff;
PRINT '  Capitals: ' + CAST(@@ROWCOUNT AS VARCHAR) + ' rows deleted';

-- 7. WeeklyPreps
DELETE FROM WeeklyPreps
WHERE AccountId = @AccountId AND Date < @Cutoff;
PRINT '  WeeklyPreps: ' + CAST(@@ROWCOUNT AS VARCHAR) + ' rows deleted';

COMMIT TRANSACTION;
PRINT 'Cleanup complete.';
