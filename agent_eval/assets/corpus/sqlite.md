# SQLite Snapshot
Source: https://sqlite.org/
Reference-Date: 2026-03-31

## parameter-binding
SQLite supports parameter binding with placeholders such as `?` or named parameters. Binding values separately from the SQL string is the standard way to avoid SQL injection and to keep queries readable. Applications should avoid string concatenation when building SQL from external inputs.

## transactions
SQLite transactions group multiple statements into one atomic unit of work. An application can explicitly `BEGIN`, then `COMMIT` to persist the changes or `ROLLBACK` to discard them. Wrapping related updates in a transaction avoids partially applied state when an error occurs midway through the sequence.

## wal-mode
Write-ahead logging, commonly called WAL mode, stores recent writes in a WAL file before they are checkpointed back into the main database file. WAL improves concurrency because readers can continue while a writer appends to the log. It is especially useful for services that need frequent reads with occasional writes.

## fts5
FTS5 is SQLite's full-text search extension. It supports indexed text search with the `MATCH` syntax and ranking helpers such as `bm25`. FTS5 is appropriate for compact local search features, including document chunk retrieval in an offline evaluation platform.

## indexes
Indexes speed up read queries by helping SQLite locate rows without scanning the full table. The tradeoff is that inserts and updates become a bit more expensive because the index must also be maintained. Indexes are most helpful on columns that are frequently filtered, joined, or sorted.

## busy-timeout
SQLite can use `busy_timeout` to wait for a short period when the database is locked instead of failing immediately. This is helpful in lightweight apps where brief lock contention may happen during overlapping access. The timeout is a mitigation, not a substitute for keeping transactions short.
