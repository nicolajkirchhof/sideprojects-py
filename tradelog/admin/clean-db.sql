-- Wipes every user object from the current database so a sqlpackage
-- Publish can recreate the schema from scratch. Runs entirely inside
-- the target database, so no master-level permissions are required.

DECLARE @SQL NVARCHAR(2000);

-- 1. Triggers
WHILE EXISTS(SELECT TOP(1) * FROM SYS.TRIGGERS WHERE is_ms_shipped = 0)
BEGIN
    SELECT TOP(1) @SQL = 'DROP TRIGGER IF EXISTS [' + [name] + '] ON ' + parent_class_desc COLLATE database_default
    FROM SYS.TRIGGERS
    WHERE is_ms_shipped = 0;
    PRINT @SQL;
    EXEC (@SQL);
END

-- 2. Stored procedures
WHILE EXISTS(SELECT TOP(1) * FROM INFORMATION_SCHEMA.ROUTINES WHERE ROUTINE_TYPE = 'PROCEDURE')
BEGIN
    SELECT TOP(1) @SQL = 'DROP PROCEDURE IF EXISTS [' + ROUTINE_SCHEMA + '].[' + ROUTINE_NAME + ']'
    FROM INFORMATION_SCHEMA.ROUTINES
    WHERE ROUTINE_TYPE = 'PROCEDURE';
    PRINT @SQL;
    EXEC (@SQL);
END

-- 3. Functions
WHILE EXISTS(SELECT TOP(1) * FROM INFORMATION_SCHEMA.ROUTINES WHERE ROUTINE_TYPE = 'FUNCTION')
BEGIN
    SELECT TOP(1) @SQL = 'DROP FUNCTION IF EXISTS [' + ROUTINE_SCHEMA + '].[' + ROUTINE_NAME + ']'
    FROM INFORMATION_SCHEMA.ROUTINES
    WHERE ROUTINE_TYPE = 'FUNCTION';
    PRINT @SQL;
    EXEC (@SQL);
END

-- 4. Foreign key constraints
WHILE EXISTS(SELECT TOP(1) * FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS WHERE CONSTRAINT_TYPE = 'FOREIGN KEY')
BEGIN
    SELECT TOP(1) @SQL = 'ALTER TABLE [' + TABLE_SCHEMA + '].[' + TABLE_NAME + '] DROP CONSTRAINT [' + CONSTRAINT_NAME + ']'
    FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS
    WHERE CONSTRAINT_TYPE = 'FOREIGN KEY';
    PRINT @SQL;
    EXEC (@SQL);
END

-- 5. Views
WHILE EXISTS(SELECT TOP(1) * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'VIEW' AND TABLE_SCHEMA != 'sys')
BEGIN
    SELECT TOP(1) @SQL = 'DROP VIEW [' + TABLE_SCHEMA + '].[' + TABLE_NAME + ']'
    FROM INFORMATION_SCHEMA.TABLES
    WHERE TABLE_TYPE = 'VIEW' AND TABLE_SCHEMA != 'sys';
    PRINT @SQL;
    EXEC (@SQL);
END

-- 6. Temporal tables — turn versioning off before dropping
WHILE EXISTS(
    SELECT TOP(1) * FROM sys.tables
    WHERE [type] = 'U' AND temporal_type = 2)
BEGIN
    SELECT TOP(1)
        @SQL = 'ALTER TABLE [' + s.name + '].[' + t.name + '] SET (SYSTEM_VERSIONING = OFF);'
    FROM sys.tables AS t
    INNER JOIN sys.schemas AS s ON t.schema_id = s.schema_id
    WHERE t.[type] = 'U' AND t.[name] != '__MigrationHistory' AND temporal_type = 2;
    PRINT @SQL;
    EXEC (@SQL);
END

-- 7. Base tables (incl. EF migration history — sqlpackage will recreate it)
WHILE EXISTS(SELECT TOP(1) * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE')
BEGIN
    SELECT TOP(1) @SQL = 'DROP TABLE [' + TABLE_SCHEMA + '].[' + TABLE_NAME + ']'
    FROM INFORMATION_SCHEMA.TABLES
    WHERE TABLE_TYPE = 'BASE TABLE';
    PRINT @SQL;
    EXEC (@SQL);
END
