-- ON Master
CREATE LOGIN ocin WITH PASSWORD = 'SET_ME'
-- ON master and DB
CREATE USER ocin FROM LOGIN ocin
-- ON db
ALTER ROLE db_owner ADD MEMBER ocin