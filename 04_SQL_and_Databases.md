# SQL and Databases — Interview Prep Syllabus

SQL and database fundamentals sit underneath almost every practical task a **Data Scientist**, **Machine Learning Engineer**, or **AI Engineer** performs. Data Scientists spend a large fraction of their time extracting, joining, and aggregating data from relational and analytical stores before any modeling begins — a wrong `JOIN` or a misunderstood `GROUP BY` silently corrupts an entire analysis. Machine Learning Engineers build feature pipelines, feature stores, and training-data extraction jobs that must run efficiently at scale, so understanding indexing, query plans, and OLTP/OLAP tradeoffs directly affects pipeline latency and cost. AI Engineers increasingly build retrieval-augmented generation (RAG) systems, agentic tool-use pipelines, and context stores that blend relational databases, vector search, and NoSQL stores — and they are frequently asked to design schemas for chat history, embeddings metadata, and tool-call logs. Across all three roles, interviewers use SQL as a fast, objective signal of a candidate's ability to reason precisely about data — making it one of the highest-leverage topics to master deeply.

---

## Table of Contents

1. [SQL Fundamentals](#1-sql-fundamentals)
   - [SELECT, WHERE, ORDER BY, LIMIT, DISTINCT](#11-select-where-order-by-limit-distinct)
   - [Aggregate Functions, GROUP BY, HAVING vs WHERE](#12-aggregate-functions-group-by-having-vs-where)
   - [JOIN Types](#13-join-types-inner-leftrightfull-outer-cross-self-join)
   - [Set Operations](#14-set-operations-union-union-all-intersect-except)
   - [Subqueries: Correlated vs Non-Correlated, EXISTS vs IN](#15-subqueries-correlated-vs-non-correlated-exists-vs-in)
   - [Finding and Deleting Duplicate Rows](#16-finding-and-deleting-duplicate-rows)
   - [Interview Questions](#17-interview-questions--sql-fundamentals)
2. [Advanced SQL](#2-advanced-sql)
   - [Window Functions](#21-window-functions)
   - [Common Table Expressions (CTEs), Recursive CTEs](#22-common-table-expressions-ctes-recursive-ctes)
   - [CASE Statements, Pivoting/Unpivoting](#23-case-statements-pivotingunpivoting-data)
   - [Query Optimization](#24-query-optimization-planner-explain-indexing)
   - [Query Performance Pitfalls](#25-query-performance-pitfalls)
   - [GROUP BY Extensions: ROLLUP, CUBE, GROUPING SETS](#26-group-by-extensions-rollup-cube-grouping-sets)
   - [Gotcha Interview Problems: Nth Highest Salary, Gaps and Islands](#27-gotcha-interview-problems-nth-highest-salary-gaps-and-islands)
   - [Views: Regular vs Materialized Views](#28-views-regular-vs-materialized-views)
   - [Stored Procedures, Functions, and Triggers](#29-stored-procedures-functions-and-triggers)
   - [Interview Questions](#210-interview-questions--advanced-sql)
3. [Database Design and Theory](#3-database-design-and-theory)
   - [Normalization and Denormalization](#31-normalization-1nf-2nf-3nf-bcnf-and-denormalization)
   - [ACID, Transactions, Isolation Levels, Deadlocks](#32-acid-transactions-isolation-levels-deadlocks)
   - [Primary/Foreign Keys, Constraints](#33-primaryforeign-keys-constraints)
   - [OLTP vs OLAP, Star/Snowflake Schema, SCDs](#34-oltp-vs-olap-star-vs-snowflake-schema-scds)
   - [Interview Questions](#35-interview-questions--database-design-and-theory)
4. [NoSQL and Modern Data Stores](#4-nosql-and-modern-data-stores)
   - [Key-Value, Document, Columnar, Graph Stores](#41-key-value-document-columnar-and-graph-stores)
   - [CAP Theorem and Eventual Consistency](#42-cap-theorem-and-eventual-consistency)
   - [Cloud Data Warehouses](#43-cloud-data-warehouses-snowflake-bigquery-redshift)
   - [Interview Questions](#44-interview-questions--nosql-and-modern-data-stores)
5. [Rapid-Fire Interview Q&A](#5-rapid-fire-interview-qa)

---

## 1. SQL Fundamentals

### 1.1 SELECT, WHERE, ORDER BY, LIMIT, DISTINCT

**Concept.** `SELECT` retrieves columns/expressions from a table. `WHERE` filters rows *before* grouping/aggregation. `ORDER BY` sorts the result set (happens near the very end of logical query processing). `LIMIT` (or `TOP`/`FETCH FIRST` depending on dialect) restricts the number of rows returned. `DISTINCT` removes duplicate rows from the output.

**Logical order of operations** (not the order you type them!):

| Step | Clause | Purpose |
|---|---|---|
| 1 | `FROM` / `JOIN` | Build the base row set |
| 2 | `WHERE` | Filter rows |
| 3 | `GROUP BY` | Group rows |
| 4 | `HAVING` | Filter groups |
| 5 | `SELECT` | Compute output expressions |
| 6 | `DISTINCT` | Remove duplicate output rows |
| 7 | `ORDER BY` | Sort |
| 8 | `LIMIT`/`OFFSET` | Restrict row count |

```sql
-- Sample table: employees(emp_id, name, dept, salary, hire_date)

SELECT name, dept, salary
FROM employees
WHERE dept = 'Engineering'
ORDER BY salary DESC
LIMIT 5;

-- Distinct departments
SELECT DISTINCT dept
FROM employees;
```

**Pitfalls**
- `ORDER BY` a column not in `SELECT` works in most engines, but combined with `DISTINCT` it must appear in the `SELECT` list (Postgres enforces this).
- `LIMIT` without `ORDER BY` returns an *arbitrary* subset — never rely on it for deterministic pagination.
- `DISTINCT` is applied to the **entire row**, not a single column, when multiple columns are selected — a common source of "why do I still have duplicates" confusion.
- `WHERE` cannot reference aggregate results or column aliases defined in `SELECT` in most engines (aliases are resolved after `WHERE`).

**Practical notes:** Use `LIMIT ... OFFSET ...` for naive pagination (slow on large offsets — the engine still scans/discards skipped rows); prefer keyset pagination (`WHERE id > last_seen_id ORDER BY id LIMIT n`) for large datasets.

---

### 1.2 Aggregate Functions, GROUP BY, HAVING vs WHERE

**Concept.** Aggregate functions (`COUNT`, `SUM`, `AVG`, `MIN`, `MAX`, `STDDEV`, `VARIANCE`) collapse multiple rows into one value per group. `GROUP BY` defines the grouping keys. `HAVING` filters *after* aggregation; `WHERE` filters *before*.

```sql
-- Average salary per department, only for departments with >5 employees
SELECT dept, COUNT(*) AS headcount, AVG(salary) AS avg_salary
FROM employees
WHERE hire_date >= '2020-01-01'   -- filters rows before grouping
GROUP BY dept
HAVING COUNT(*) > 5               -- filters groups after aggregation
ORDER BY avg_salary DESC;
```

| | `WHERE` | `HAVING` |
|---|---|---|
| Applies to | Individual rows | Groups (after `GROUP BY`) |
| Can use aggregates? | No | Yes |
| Runs | Before grouping | After grouping |
| Can filter on raw columns? | Yes | Yes (but usually redundant with `WHERE`) |

**Pitfalls**
- Every column in `SELECT` that is not wrapped in an aggregate function must appear in `GROUP BY` (strict SQL / Postgres / MySQL with `ONLY_FULL_GROUP_BY`). MySQL in legacy mode silently allows this and returns an arbitrary row — a classic bug source.
- `COUNT(*)` counts all rows including NULLs; `COUNT(column)` skips NULLs in that column; `COUNT(DISTINCT column)` counts unique non-null values.
- `AVG()` ignores NULLs in the denominator too — `AVG` over a column with NULLs is NOT the same as `SUM(col)/COUNT(*)`.
- Putting a condition that could be a `WHERE` filter into `HAVING` instead works but is slower — the engine aggregates all rows first, then discards groups.

```sql
-- NULL handling illustration
SELECT COUNT(*) AS all_rows, COUNT(bonus) AS non_null_bonus, AVG(bonus) AS avg_bonus
FROM employees;
```

---

### 1.3 JOIN Types: INNER, LEFT/RIGHT/FULL OUTER, CROSS, SELF JOIN

**Concept.** Joins combine rows from two or more tables based on a related column.

**Sample schema**

```sql
CREATE TABLE customers (customer_id INT PRIMARY KEY, name VARCHAR(50));
CREATE TABLE orders (
  order_id INT PRIMARY KEY,
  customer_id INT,
  amount DECIMAL(10,2),
  FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);
```

| Join Type | "Venn Diagram in Words" | Rows Returned |
|---|---|---|
| `INNER JOIN` | Only the overlapping middle of two circles | Rows where the join key matches in **both** tables |
| `LEFT OUTER JOIN` | The whole left circle, plus the overlap | All left rows; unmatched right columns are `NULL` |
| `RIGHT OUTER JOIN` | The whole right circle, plus the overlap | All right rows; unmatched left columns are `NULL` |
| `FULL OUTER JOIN` | Both circles entirely | All rows from both sides; unmatched columns from either side are `NULL` |
| `CROSS JOIN` | No circles — every combination | Cartesian product: `N x M` rows |
| `SELF JOIN` | One circle joined to a copy of itself | Same table aliased twice, joined on a relationship within itself |

```sql
-- INNER JOIN: only customers who have placed an order
SELECT c.name, o.order_id, o.amount
FROM customers c
INNER JOIN orders o ON c.customer_id = o.customer_id;

-- LEFT JOIN: every customer, even those with zero orders
SELECT c.name, o.order_id, o.amount
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id;

-- Customers with NO orders (anti-join pattern)
SELECT c.name
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
WHERE o.order_id IS NULL;

-- FULL OUTER JOIN: all customers and all orders, matched where possible
SELECT c.name, o.order_id
FROM customers c
FULL OUTER JOIN orders o ON c.customer_id = o.customer_id;

-- CROSS JOIN: every product paired with every store (e.g., building a price matrix)
SELECT p.product_name, s.store_name
FROM products p
CROSS JOIN stores s;

-- SELF JOIN: find employees who earn more than their manager
SELECT e.name AS employee, m.name AS manager, e.salary, m.salary AS mgr_salary
FROM employees e
JOIN employees m ON e.manager_id = m.emp_id
WHERE e.salary > m.salary;
```

**Pitfalls**
- MySQL has no `FULL OUTER JOIN` keyword — emulate with `LEFT JOIN UNION RIGHT JOIN` (dedupe with `UNION`, not `UNION ALL`, or handle carefully to avoid duplicating the matched rows).
- Filtering a `LEFT JOIN`'s right-table column in `WHERE` (instead of the `ON` clause) silently converts it into an `INNER JOIN` — this is one of the most common real-world bugs.
  ```sql
  -- BUG: this becomes an INNER JOIN because WHERE drops unmatched (NULL) rows
  SELECT c.name, o.amount
  FROM customers c LEFT JOIN orders o ON c.customer_id = o.customer_id
  WHERE o.amount > 100;

  -- FIX: put the condition in ON to preserve LEFT JOIN semantics
  SELECT c.name, o.amount
  FROM customers c LEFT JOIN orders o
       ON c.customer_id = o.customer_id AND o.amount > 100;
  ```
- Joining on a non-unique key on either side can silently multiply row counts ("fan-out") — always sanity-check `COUNT(*)` before and after a join.
- `CROSS JOIN` on large tables can explode memory/compute — guard with `LIMIT` while testing.

---

### 1.4 Set Operations: UNION, UNION ALL, INTERSECT, EXCEPT

**Concept.** Set operations combine the **results of two queries** (not tables directly) that must have the same number of columns with compatible types.

| Operation | Behavior |
|---|---|
| `UNION` | Combines rows from both queries, **removes duplicates** (implicit `DISTINCT`, so it sorts/hashes — slower) |
| `UNION ALL` | Combines rows, **keeps duplicates** — faster, no dedup pass |
| `INTERSECT` | Rows present in **both** result sets |
| `EXCEPT` (Postgres/SQL Server) / `MINUS` (Oracle) | Rows in the first result set but **not** in the second |

```sql
-- Active customers this year OR last year (deduplicated)
SELECT customer_id FROM orders WHERE order_year = 2026
UNION
SELECT customer_id FROM orders WHERE order_year = 2025;

-- Same, but keep duplicates (faster, useful when you'll aggregate afterward anyway)
SELECT customer_id FROM orders WHERE order_year = 2026
UNION ALL
SELECT customer_id FROM orders WHERE order_year = 2025;

-- Customers who ordered in BOTH years
SELECT customer_id FROM orders WHERE order_year = 2026
INTERSECT
SELECT customer_id FROM orders WHERE order_year = 2025;

-- Customers who ordered in 2026 but NOT in 2025 (churn-risk / new-customer analysis)
SELECT customer_id FROM orders WHERE order_year = 2026
EXCEPT
SELECT customer_id FROM orders WHERE order_year = 2025;
```

**Pitfalls**
- Defaulting to `UNION` when you actually don't expect duplicates wastes a sort/hash-dedup step — use `UNION ALL` whenever you know (or don't care) that duplicates won't occur or don't matter.
- Column names in the final result come from the **first** query; column count and (implicitly comparable) types must match across all queries.
- `INTERSECT`/`EXCEPT` compare whole rows, not just one column — if you `SELECT customer_id, name` from two differently-populated tables, rows only match when *both* columns are identical.

---

### 1.5 Subqueries: Correlated vs Non-Correlated, EXISTS vs IN

**Concept.** A subquery is a query nested inside another. A **non-correlated** subquery runs once, independently of the outer query. A **correlated** subquery references a column from the outer query and conceptually re-executes for each outer row.

```sql
-- Non-correlated subquery: employees earning above the company average
SELECT name, salary
FROM employees
WHERE salary > (SELECT AVG(salary) FROM employees);

-- Correlated subquery: employees earning above THEIR department's average
SELECT e.name, e.salary, e.dept
FROM employees e
WHERE e.salary > (
  SELECT AVG(e2.salary)
  FROM employees e2
  WHERE e2.dept = e.dept        -- references outer table e => correlated
);
```

**EXISTS vs IN**

| | `IN` | `EXISTS` |
|---|---|---|
| Compares | A value against a list/subquery result | Whether a correlated subquery returns **any** row |
| NULL behavior | `IN` list containing NULL can cause `NOT IN` to return zero rows unexpectedly | `EXISTS`/`NOT EXISTS` are NULL-safe |
| Performance | Optimizer may materialize the list first | Optimizer can often short-circuit (stop at first match) |
| Typical use | Small, static, or non-correlated sets | Correlated existence checks, especially with large tables |

```sql
-- IN: customers who placed at least one order over $500
SELECT name FROM customers
WHERE customer_id IN (SELECT customer_id FROM orders WHERE amount > 500);

-- EXISTS: same logic, typically more efficient on large correlated data
SELECT name FROM customers c
WHERE EXISTS (
  SELECT 1 FROM orders o
  WHERE o.customer_id = c.customer_id AND o.amount > 500
);

-- NOT IN danger: if any order.customer_id is NULL, this returns ZERO rows
SELECT name FROM customers
WHERE customer_id NOT IN (SELECT customer_id FROM orders);   -- risky!

-- Safer equivalent using NOT EXISTS
SELECT name FROM customers c
WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.customer_id);
```

**Pitfalls**
- `NOT IN` with a subquery that can return `NULL` is a classic interview trap — a single `NULL` in the list poisons the entire comparison (because `x <> NULL` is `UNKNOWN`, not `TRUE`), causing `NOT IN` to return no rows at all.
- Scalar subqueries (`= (SELECT ...)`) must return **exactly one row/column**; if the subquery can return more than one row, the query errors at runtime.
- Correlated subqueries that could be rewritten as joins are often — but not always — optimized identically by modern planners; always check `EXPLAIN` rather than assuming.
- Subqueries in the `SELECT` list run once per outer row if correlated — a common accidental performance trap ("subquery in SELECT" anti-pattern), often better expressed with a `JOIN` or window function.

---

### 1.6 Finding and Deleting Duplicate Rows

**Concept.** Duplicate rows creep in from failed upserts, re-run ETL jobs, or a missing `UNIQUE` constraint that was supposed to prevent them. *Finding* duplicates is a `GROUP BY`/`HAVING` problem (see Q15 in 1.7). *Deleting* duplicates is a step harder: you must pick exactly one row to keep per duplicate group, which requires a tiebreaker column (a surrogate key, `created_at`, etc.) — "delete the duplicates" is meaningless without deciding which copy survives.

```sql
-- users(user_id, email, created_at) -- email should be unique but isn't enforced

-- Step 1 (always run this first): identify the duplicate groups
SELECT email, COUNT(*) AS occurrences
FROM users
GROUP BY email
HAVING COUNT(*) > 1;

-- Step 2: rank rows within each duplicate group, keep rn = 1, delete the rest.
-- Tiebreaker here is "earliest created_at, then lowest user_id" -- pick whatever
-- business rule defines the "canonical" row.
WITH ranked AS (
  SELECT user_id,
         ROW_NUMBER() OVER (
           PARTITION BY email
           ORDER BY created_at ASC, user_id ASC
         ) AS rn
  FROM users
)
DELETE FROM users
WHERE user_id IN (SELECT user_id FROM ranked WHERE rn > 1);
```

```sql
-- Portable alternative with no window functions: a self-join delete
-- (works on older MySQL; keeps the lowest user_id per email)
DELETE u1 FROM users u1
JOIN users u2
  ON u1.email = u2.email
 AND u1.user_id > u2.user_id;

-- Postgres shortcut using the physical row identifier ctid (Postgres-only,
-- no tiebreaker column needed, but "which row survives" is arbitrary)
DELETE FROM users a
USING users b
WHERE a.ctid < b.ctid
  AND a.email = b.email;
```

**Pitfalls**
- Never run the `DELETE` before validating the corresponding `SELECT`/CTE logic returns exactly the rows you expect — wrap the first attempt in `BEGIN; ... ROLLBACK;` and inspect row counts before committing.
- "Delete duplicates" is ambiguous until you define the keep-rule (oldest? newest? lowest ID?) — always clarify this explicitly in an interview before writing the query; interviewers often leave it open-ended on purpose.
- After cleanup, add the `UNIQUE` constraint that should have existed in the first place — otherwise the same duplicates reappear.
- `DELETE ... WHERE col IN (subquery on the same table)` can be slow on large tables because the subquery re-scans the table; some engines (MySQL) historically disallowed referencing the target table directly in the subquery, which is why the CTE/self-join forms above materialize the row set first.

---

### 1.7 Interview Questions — SQL Fundamentals

**Q1. What is the logical order of execution of SQL clauses, and why does it matter for aliases in `WHERE`?**
A: `FROM/JOIN → WHERE → GROUP BY → HAVING → SELECT → DISTINCT → ORDER BY → LIMIT`. Because `WHERE` is evaluated before `SELECT`, a column alias defined in `SELECT` is not yet known during `WHERE` evaluation, so referencing it in `WHERE` raises an error in most engines (you'd repeat the expression or use a subquery/CTE instead).

**Q2. What's the difference between `WHERE` and `HAVING`?**
A: `WHERE` filters individual rows before any grouping and cannot use aggregate functions. `HAVING` filters groups after `GROUP BY`/aggregation and can use aggregate functions. Filtering as early as possible (in `WHERE`) is more efficient than filtering after aggregation in `HAVING`.

**Q3. Write a query to find the second-highest salary in the `employees` table without using `LIMIT`/`OFFSET` or window functions.**
```sql
SELECT MAX(salary) AS second_highest_salary
FROM employees
WHERE salary < (SELECT MAX(salary) FROM employees);
```

**Q4. Given `customers(customer_id, name)` and `orders(order_id, customer_id, amount)`, write a query returning customers who have never placed an order.**
```sql
SELECT c.customer_id, c.name
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
WHERE o.order_id IS NULL;
```

**Q5. Explain the difference between `UNION` and `UNION ALL`, including a performance consideration.**
A: `UNION` deduplicates the combined result set, which requires a sort or hash pass over all rows — more expensive. `UNION ALL` simply concatenates results without deduplication and is faster. Use `UNION ALL` whenever duplicates are impossible or acceptable.

**Q6. Why can `NOT IN` behave unexpectedly with `NULL` values? Provide an example.**
A: If the subquery/list used with `NOT IN` contains a `NULL`, every comparison against that `NULL` evaluates to `UNKNOWN` rather than `TRUE`/`FALSE`, and SQL's three-valued logic means the entire `NOT IN` predicate can never evaluate to `TRUE` for any row, silently returning zero rows. Example: `WHERE customer_id NOT IN (SELECT customer_id FROM orders)` returns nothing if any `orders.customer_id` is `NULL`. Fix with `NOT EXISTS` or filter out NULLs explicitly (`... WHERE customer_id IS NOT NULL`).

**Q7. What is the difference between a correlated and a non-correlated subquery?**
A: A non-correlated subquery is self-contained and executes once, independent of the outer query. A correlated subquery references a column from the outer query, so conceptually it must be evaluated once per outer row (though optimizers often rewrite it as a join internally).

**Q8. Given `orders(order_id, customer_id, order_date, amount)`, write a query to find each customer's most recent order.**
```sql
SELECT o.*
FROM orders o
WHERE o.order_date = (
  SELECT MAX(o2.order_date)
  FROM orders o2
  WHERE o2.customer_id = o.customer_id
);
```

**Q9. What happens if you `SELECT` a non-aggregated, non-grouped column alongside an aggregate function without including it in `GROUP BY`?**
A: In standard SQL (and Postgres, SQL Server, modern MySQL with `ONLY_FULL_GROUP_BY`), this raises an error because the value of that column is ambiguous per group. Legacy MySQL without strict mode allows it and picks an arbitrary row's value per group — a source of non-deterministic bugs.

**Q10. Write a query using a `CROSS JOIN` to generate every combination of `products` and `regions`, then explain a real use case.**
```sql
SELECT p.product_name, r.region_name
FROM products p
CROSS JOIN regions r;
```
A: Use case: building a complete "expected" matrix (e.g., every product should have a price row in every region) so you can `LEFT JOIN` against actual data and find missing combinations (gaps in coverage).

**Q11. Explain `EXISTS` vs `IN` and when you'd prefer one over the other.**
A: `IN` checks membership against a value list/subquery; `EXISTS` checks whether a correlated subquery returns any row at all, short-circuiting on the first match. `EXISTS` is generally preferred for large correlated existence checks and is NULL-safe, whereas `IN`/`NOT IN` can be fooled by NULLs. For small, static, non-correlated lists, `IN` is perfectly fine and often more readable.

**Q12. Given `employees(emp_id, name, manager_id, salary)`, write a self-join query to list each employee with their manager's name.**
```sql
SELECT e.name AS employee_name, m.name AS manager_name
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.emp_id;
```

**Q13. What's the difference between `COUNT(*)`, `COUNT(column)`, and `COUNT(DISTINCT column)`?**
A: `COUNT(*)` counts all rows regardless of NULLs. `COUNT(column)` counts rows where that column is non-NULL. `COUNT(DISTINCT column)` counts the number of distinct non-NULL values in that column.

**Q14. Why does putting a right-table filter in `WHERE` instead of `ON` change the meaning of a `LEFT JOIN`?**
A: `ON` conditions determine which rows are matched *during* the join, so unmatched left rows still appear with NULLs on the right side. `WHERE` runs *after* the join is materialized, and a condition like `right_col > 100` evaluates to `UNKNOWN`/false for the NULL-filled unmatched rows, discarding them — effectively turning the `LEFT JOIN` into an `INNER JOIN`.

**Q15. Write a query to find duplicate email addresses in a `users(user_id, email)` table.**
```sql
SELECT email, COUNT(*) AS occurrences
FROM users
GROUP BY email
HAVING COUNT(*) > 1;
```

**Q16. Write a query to delete duplicate rows from `users(user_id, email, created_at)`, keeping only the earliest-created row per email.**
```sql
WITH ranked AS (
  SELECT user_id,
         ROW_NUMBER() OVER (
           PARTITION BY email
           ORDER BY created_at ASC, user_id ASC
         ) AS rn
  FROM users
)
DELETE FROM users
WHERE user_id IN (SELECT user_id FROM ranked WHERE rn > 1);
```

**Q17. Why should you always run the "find duplicates" `SELECT` before converting it into a `DELETE`?**
A: A `DELETE` is destructive and, unlike a `SELECT`, gives you no chance to inspect what was removed after the fact. Validating the row set (and the ranking/tiebreaker logic) with a `SELECT` first — ideally inside `BEGIN; ... ROLLBACK;` — catches mistakes like an unintended `PARTITION BY` key or a wrong tiebreaker column before they cause irreversible data loss.

**Q18. How would you delete duplicate rows in a dialect without window function support, using only a self-join?**
```sql
DELETE u1 FROM users u1
JOIN users u2
  ON u1.email = u2.email
 AND u1.user_id > u2.user_id;
```
A: For every pair of rows sharing the same `email`, this deletes the one with the larger `user_id`, leaving exactly the lowest-`user_id` row per email group — equivalent in effect to the `ROW_NUMBER()` approach but expressed purely as a join condition.

---

## 2. Advanced SQL

### 2.1 Window Functions

**Concept.** Window functions compute a value across a set of rows ("window") related to the current row, **without collapsing rows** the way `GROUP BY` does. Syntax: `function() OVER ([PARTITION BY ...] [ORDER BY ...] [frame_clause])`.

| Function | Purpose |
|---|---|
| `ROW_NUMBER()` | Unique sequential integer per row within a partition (no ties) |
| `RANK()` | Rank with **gaps** after ties (1,2,2,4) |
| `DENSE_RANK()` | Rank with **no gaps** after ties (1,2,2,3) |
| `NTILE(n)` | Splits partition into `n` roughly equal buckets |
| `LAG(col, n)` | Value from `n` rows **before** the current row |
| `LEAD(col, n)` | Value from `n` rows **after** the current row |
| `SUM()/AVG()/COUNT() OVER(...)` | Running totals, moving averages, group aggregates without collapsing rows |
| `FIRST_VALUE()/LAST_VALUE()` | First/last value in the window frame |

```sql
-- Sample schema: sales(sale_id, salesperson, region, sale_date, amount)

-- ROW_NUMBER / RANK / DENSE_RANK comparison
SELECT salesperson, region, amount,
       ROW_NUMBER() OVER (PARTITION BY region ORDER BY amount DESC) AS row_num,
       RANK()       OVER (PARTITION BY region ORDER BY amount DESC) AS rnk,
       DENSE_RANK() OVER (PARTITION BY region ORDER BY amount DESC) AS dense_rnk
FROM sales;
```

| region | amount | row_num | rnk | dense_rnk |
|---|---|---|---|---|
| East | 500 | 1 | 1 | 1 |
| East | 500 | 2 | 1 | 1 |
| East | 300 | 3 | 3 | 2 |
| West | 400 | 1 | 1 | 1 |

```sql
-- LAG/LEAD: month-over-month change
SELECT sale_date, amount,
       LAG(amount)  OVER (ORDER BY sale_date) AS prev_amount,
       LEAD(amount) OVER (ORDER BY sale_date) AS next_amount,
       amount - LAG(amount) OVER (ORDER BY sale_date) AS delta
FROM monthly_sales;

-- Running total per salesperson
SELECT salesperson, sale_date, amount,
       SUM(amount) OVER (
         PARTITION BY salesperson
         ORDER BY sale_date
         ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
       ) AS running_total
FROM sales;

-- 7-day moving average (frame clause)
SELECT sale_date, amount,
       AVG(amount) OVER (
         ORDER BY sale_date
         ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
       ) AS moving_avg_7d
FROM daily_sales;

-- Top 3 earners per department
SELECT * FROM (
  SELECT name, dept, salary,
         RANK() OVER (PARTITION BY dept ORDER BY salary DESC) AS rnk
  FROM employees
) ranked
WHERE rnk <= 3;
```

**Frame clauses explained in words:**
- `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` → "from the start of the partition up through this row" (running total).
- `ROWS BETWEEN 6 PRECEDING AND CURRENT ROW` → "the current row plus the 6 rows before it" (7-row moving window).
- `RANGE` frames use logical value ranges (e.g., dates) instead of physical row counts — useful when rows can tie on the `ORDER BY` value.

**Pitfalls**
- Window functions execute logically **after** `WHERE`/`GROUP BY`/`HAVING` but **before** `ORDER BY`/`SELECT DISTINCT` — you cannot filter on a window function's result directly in `WHERE`; wrap it in a subquery/CTE and filter in the outer query (as shown in "Top 3 earners" above).
- Forgetting `PARTITION BY` computes the window across the **entire result set**, not per group — a frequent bug when porting `GROUP BY` logic to window functions.
- Default frame when `ORDER BY` is present but no explicit frame is given is `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`, which can produce surprising results with tied `ORDER BY` values (ties are all included up to "current row's value", not "current row's position").
- `RANK()`/`DENSE_RANK()` are not interchangeable with `ROW_NUMBER()` when picking "the top N per group" — with ties, `ROW_NUMBER()` arbitrarily picks one row per tie, `RANK()` may return more than N rows if there's a tie at the boundary.

---

### 2.2 Common Table Expressions (CTEs), Recursive CTEs

**Concept.** A CTE (`WITH name AS (...)`) is a named, temporary result set scoped to a single query — improves readability and lets you reference the same logical subquery multiple times (though materialization behavior varies by engine).

```sql
-- Basic CTE: readability + reuse
WITH dept_avg AS (
  SELECT dept, AVG(salary) AS avg_salary
  FROM employees
  GROUP BY dept
)
SELECT e.name, e.dept, e.salary, d.avg_salary
FROM employees e
JOIN dept_avg d ON e.dept = d.dept
WHERE e.salary > d.avg_salary;

-- Multiple CTEs chained together
WITH high_value AS (
  SELECT customer_id, SUM(amount) AS total_spent
  FROM orders
  GROUP BY customer_id
  HAVING SUM(amount) > 10000
),
region_map AS (
  SELECT customer_id, region FROM customers
)
SELECT h.customer_id, h.total_spent, r.region
FROM high_value h
JOIN region_map r ON h.customer_id = r.customer_id;
```

**Recursive CTE** — used for hierarchical/graph traversal (org charts, bill-of-materials, category trees).

```sql
-- employees(emp_id, name, manager_id)
-- Find the full management chain (all subordinates) under a given manager

WITH RECURSIVE org_chain AS (
  -- Anchor member: the starting employee
  SELECT emp_id, name, manager_id, 1 AS level
  FROM employees
  WHERE emp_id = 100        -- e.g., the CEO's ID

  UNION ALL

  -- Recursive member: find direct reports of everyone already in org_chain
  SELECT e.emp_id, e.name, e.manager_id, oc.level + 1
  FROM employees e
  JOIN org_chain oc ON e.manager_id = oc.emp_id
)
SELECT * FROM org_chain ORDER BY level;
```

*(Note: MySQL/Postgres/SQL Server use `WITH RECURSIVE`; SQL Server also accepts `WITH` without the `RECURSIVE` keyword. Oracle traditionally uses `CONNECT BY PRIOR` instead of recursive CTEs.)*

**How it executes, in words:** The anchor query runs first, producing the initial row set. The recursive term then repeatedly joins against the *most recently produced* rows (not the whole accumulated table) and unions the new rows in, stopping automatically when the recursive term produces zero new rows.

**Pitfalls**
- A recursive CTE without a terminating condition (e.g., a self-referencing cycle in the data) can loop forever — always test with an explicit level/depth cap (`WHERE level < 20`) during development, and consider `MAXRECURSION` limits (SQL Server) or cycle-detection logic for production.
- The recursive term must be `UNION ALL` (not `UNION`) in most engines, and cannot contain aggregate functions, `GROUP BY`, or another recursive reference beyond one level in some dialects.
- CTEs are **not always materialized/optimized as a single unit** — Postgres pre-v12 always materialized CTEs (an "optimization fence"); Postgres 12+ can inline non-recursive CTEs. This affects performance predictions and is a good "it depends on your engine and version" interview answer.

---

### 2.3 CASE Statements, Pivoting/Unpivoting Data

**Concept.** `CASE` is SQL's inline conditional expression, usable anywhere a value is expected (`SELECT`, `WHERE`, `ORDER BY`, `GROUP BY`). It underlies manual pivoting.

```sql
-- Simple and searched CASE
SELECT name, salary,
  CASE
    WHEN salary >= 100000 THEN 'High'
    WHEN salary >= 60000  THEN 'Medium'
    ELSE 'Low'
  END AS salary_band
FROM employees;

-- Manual PIVOT: rows -> columns using CASE + aggregation
-- sales(salesperson, quarter, amount)  =>  one row per salesperson, one column per quarter
SELECT salesperson,
  SUM(CASE WHEN quarter = 'Q1' THEN amount ELSE 0 END) AS Q1,
  SUM(CASE WHEN quarter = 'Q2' THEN amount ELSE 0 END) AS Q2,
  SUM(CASE WHEN quarter = 'Q3' THEN amount ELSE 0 END) AS Q3,
  SUM(CASE WHEN quarter = 'Q4' THEN amount ELSE 0 END) AS Q4
FROM sales
GROUP BY salesperson;
```

**Native PIVOT (SQL Server / Snowflake syntax):**
```sql
SELECT salesperson, [Q1], [Q2], [Q3], [Q4]
FROM sales
PIVOT (
  SUM(amount) FOR quarter IN ([Q1], [Q2], [Q3], [Q4])
) AS pvt;
```

**UNPIVOT — columns back to rows:**
```sql
-- wide_sales(salesperson, Q1, Q2, Q3, Q4) -> long format
SELECT salesperson, quarter, amount
FROM wide_sales
UNPIVOT (
  amount FOR quarter IN (Q1, Q2, Q3, Q4)
) AS unpvt;

-- Portable UNPIVOT using UNION ALL (works everywhere)
SELECT salesperson, 'Q1' AS quarter, Q1 AS amount FROM wide_sales
UNION ALL
SELECT salesperson, 'Q2', Q2 FROM wide_sales
UNION ALL
SELECT salesperson, 'Q3', Q3 FROM wide_sales
UNION ALL
SELECT salesperson, 'Q4', Q4 FROM wide_sales;
```

**Pitfalls**
- `CASE` evaluates `WHEN` clauses **in order** and stops at the first match — order matters, especially with overlapping range conditions.
- Native `PIVOT`/`UNPIVOT` syntax is **not standard SQL** — it exists in SQL Server, Oracle, Snowflake, BigQuery (`PIVOT`/`UNPIVOT` operators), but not in vanilla MySQL/Postgres, which require the manual `CASE`+`GROUP BY` or `UNION ALL` pattern shown above.
- Forgetting `ELSE` in a `CASE` used for aggregation (e.g., `SUM(CASE WHEN ... THEN amount END)`) implicitly returns `NULL` for non-matching rows, which `SUM` ignores — usually fine, but be explicit about whether you want `0` or `NULL` for empty groups.

---

### 2.4 Query Optimization: Planner, EXPLAIN, Indexing

**Concept.** The **query planner/optimizer** parses SQL into a logical plan, then chooses a physical execution plan (which access paths, join algorithms, and orderings to use) based on table statistics (row counts, value distributions/histograms) — a **cost-based optimizer**.

**How the planner reasons, in words:**
1. Parse SQL into a query tree.
2. Rewrite/normalize (e.g., subquery-to-join transformations, predicate pushdown).
3. Enumerate candidate physical plans (which index to use, which join algorithm — nested loop, hash join, merge join, which join order).
4. Estimate the **cost** of each plan using table/index statistics.
5. Pick the plan with lowest estimated cost; execute it.

```sql
-- Postgres / MySQL 8+ : shows the chosen plan
EXPLAIN SELECT * FROM orders WHERE customer_id = 42;

-- Postgres: actually executes and reports real timing/row counts vs estimates
EXPLAIN ANALYZE SELECT * FROM orders WHERE customer_id = 42;

-- SQL Server equivalent
SET STATISTICS PROFILE ON;
-- or use the graphical "Actual Execution Plan"
```

Reading `EXPLAIN ANALYZE` output — key signals:

| Signal | Meaning |
|---|---|
| `Seq Scan` (Postgres) / `Full Table Scan` | No index used — reads every row |
| `Index Scan` | Uses index to find rows, then fetches full row from table |
| `Index Only Scan` | Satisfies the query entirely from the index (no table lookup) — fastest |
| `Bitmap Heap Scan` | Combines multiple index matches into a bitmap before fetching rows |
| Estimated rows vs. actual rows way off | Stale statistics — consider `ANALYZE`/`UPDATE STATISTICS` |
| `Nested Loop` | Good for small outer sets; bad if both sides are large |
| `Hash Join` | Good for large, roughly-equal-size unsorted sets |
| `Merge Join` | Good when both inputs are already sorted on the join key |

**Indexing strategies**

| Index Type | Best For | Notes |
|---|---|---|
| **B-tree** | Range queries (`<`, `>`, `BETWEEN`), sorting, equality | Default index type in almost every RDBMS; supports ordered traversal |
| **Hash index** | Pure equality lookups (`=`) | Cannot support range queries or sorting; smaller/faster for exact-match only workloads |
| **Composite (multi-column)** | Queries filtering/sorting on multiple columns together | Column **order matters** — see below |
| **Covering index** | Queries where the index itself contains every column needed | Enables index-only scans, avoiding a trip to the table heap |
| **Partial/filtered index** | Queries that always filter on a known subset (e.g., `WHERE active = true`) | Smaller index, faster for that subset |
| **Bitmap index** | Low-cardinality columns in analytical/OLAP workloads | Common in data warehouses, poor for high-write OLTP |

```sql
CREATE INDEX idx_orders_customer_id ON orders (customer_id);

-- Composite index: order matters! This helps queries filtering on
-- customer_id alone, OR customer_id+order_date together,
-- but NOT queries filtering on order_date alone.
CREATE INDEX idx_orders_cust_date ON orders (customer_id, order_date);

-- Covering index: includes amount so the query never touches the table
CREATE INDEX idx_orders_cover ON orders (customer_id, order_date) INCLUDE (amount);  -- SQL Server/Postgres 11+

-- Partial index (Postgres): only indexes active users, much smaller
CREATE INDEX idx_active_users ON users (email) WHERE is_active = true;
```

**Index selectivity.** Selectivity = (distinct values / total rows). A **highly selective** column (e.g., `email`, near-unique) benefits enormously from an index — a lookup narrows to almost one row. A **low-selectivity** column (e.g., a boolean `is_deleted` with 99% `false`) gives the optimizer little to work with; it may ignore the index and do a sequential scan anyway because reading most of the table via random index lookups is slower than one sequential pass.

**When indexes hurt writes.** Every `INSERT`/`UPDATE`/`DELETE` must also update every index on the affected columns. More indexes → slower writes and more storage. This is why OLTP tables are indexed sparingly (targeted at known query patterns) while OLAP/reporting tables can be indexed (or columnar-encoded) more heavily since writes are batched and infrequent.

**Pitfalls**
- Adding an index doesn't guarantee the optimizer uses it — stale statistics, low selectivity, or a small table (planner decides a full scan is cheaper than random I/O) can all cause the planner to ignore it.
- Over-indexing a write-heavy OLTP table degrades `INSERT`/`UPDATE` throughput and bloats storage — every index is a tradeoff, not a free win.
- Index column order in composite indexes must match the **leftmost prefix** of your query's filter/sort columns to be usable.

---

### 2.5 Query Performance Pitfalls

**N+1 problem.** Occurs when application code loops over a result set and issues one additional query per row (e.g., fetching a customer list, then querying orders separately for each customer in a loop) instead of a single joined/batched query.

```sql
-- BAD (N+1): 1 query to get customers, then N queries like this per customer
SELECT * FROM orders WHERE customer_id = 1;
SELECT * FROM orders WHERE customer_id = 2;
-- ... repeated for every customer

-- GOOD: one query, joined or batched with IN
SELECT c.customer_id, o.*
FROM customers c
JOIN orders o ON o.customer_id = c.customer_id
WHERE c.customer_id IN (1, 2, 3, ...);
```

**Implicit type conversion.** Comparing columns/parameters of mismatched types forces the engine to convert one side, which can silently disable index usage.

```sql
-- phone_number stored as VARCHAR; comparing to an integer literal
-- forces an implicit CAST of the column on every row => index on phone_number is unusable
SELECT * FROM customers WHERE phone_number = 5551234567;   -- BAD

SELECT * FROM customers WHERE phone_number = '5551234567'; -- GOOD, matches column type
```

**Function-on-indexed-column ("non-SARGable" predicates).** Wrapping an indexed column in a function prevents the optimizer from using a standard B-tree index scan, because the index stores raw column values, not the function's output.

```sql
-- BAD: YEAR(order_date) must be computed for every row => forces a full scan
SELECT * FROM orders WHERE YEAR(order_date) = 2026;

-- GOOD: rewrite as a range so the index on order_date can be used (SARGable)
SELECT * FROM orders
WHERE order_date >= '2026-01-01' AND order_date < '2027-01-01';

-- BAD: leading wildcard prevents B-tree index use
SELECT * FROM customers WHERE name LIKE '%Smith';

-- OK: trailing wildcard CAN use a B-tree index (prefix match)
SELECT * FROM customers WHERE name LIKE 'Smith%';
```

**SARGability** ("Search ARGument-able") — a predicate is SARGable if the engine can use an index seek/range scan to evaluate it directly, without transforming every row's value first. Rule of thumb: keep the indexed column **bare** on one side of the comparison; push transformations onto the constant/literal side instead.

| Non-SARGable | SARGable Rewrite |
|---|---|
| `WHERE YEAR(date_col) = 2026` | `WHERE date_col >= '2026-01-01' AND date_col < '2027-01-01'` |
| `WHERE UPPER(name) = 'JOHN'` | Store/query with case-insensitive collation, or a functional/expression index on `UPPER(name)` |
| `WHERE salary + 1000 > 50000` | `WHERE salary > 49000` |
| `WHERE col LIKE '%abc%'` | Full-text search index, or trigram index (`pg_trgm`) — a plain B-tree can't help here |

**Pitfalls / practical notes**
- N+1 is primarily an *application/ORM* pitfall (Django, SQLAlchemy, Hibernate all have "lazy loading" defaults that cause it) — interviewers often want you to name it and describe eager-loading/`JOIN`/batching as the fix.
- Some modern optimizers (Postgres with expression indexes, SQL Server computed columns) can make certain function-wrapped predicates SARGable again if you build an index on the expression itself: `CREATE INDEX ON orders (YEAR(order_date));` (dialect-dependent syntax).
- Always validate assumptions with `EXPLAIN` rather than memorized rules — planner behavior varies by engine/version and by table size (a tiny table may get a full scan regardless, and that's correct).

---

### 2.6 GROUP BY Extensions: ROLLUP, CUBE, GROUPING SETS

**Concept.** Plain `GROUP BY` produces exactly one row per unique combination of grouping columns. `ROLLUP`, `CUBE`, and `GROUPING SETS` add **subtotal/grand-total rows** in a single query, avoiding a manual `UNION ALL` of several aggregate queries at different grouping levels — common in financial/BI reports ("subtotal by region, then a grand total").

```sql
-- sales(region, product, amount)

-- ROLLUP: hierarchical subtotals -- (region, product) rows, then a subtotal
-- per region (product = NULL), then one grand total (region = NULL, product = NULL)
SELECT region, product, SUM(amount) AS total
FROM sales
GROUP BY ROLLUP (region, product)
ORDER BY region, product;

-- CUBE: every possible combination of subtotals -- (region, product), region alone,
-- product alone, AND the grand total (order of columns doesn't change what's produced)
SELECT region, product, SUM(amount) AS total
FROM sales
GROUP BY CUBE (region, product);

-- GROUPING SETS: explicitly list exactly which grouping combinations you want,
-- when ROLLUP/CUBE would generate more (or different) rows than you need
SELECT region, product, SUM(amount) AS total
FROM sales
GROUP BY GROUPING SETS ((region, product), (region), ());

-- GROUPING() distinguishes a subtotal row's NULL from a genuine NULL grouping value
SELECT region, product, SUM(amount) AS total,
       GROUPING(region)  AS is_region_subtotal,
       GROUPING(product) AS is_product_subtotal
FROM sales
GROUP BY ROLLUP (region, product);
```

**Pitfalls**
- `ROLLUP (region, product)` is order-sensitive (it rolls up right-to-left: subtotal by `region` first, then grand total) — `ROLLUP (product, region)` produces a *different* set of subtotal rows. `CUBE` and `GROUPING SETS` are not hierarchy-dependent in the same way; `CUBE` always emits every combination.
- A subtotal row's grouping columns come back as `NULL`, which is easy to confuse with an actual `NULL` value in the underlying data (e.g., an unknown region) — use `GROUPING(col)` (returns `1` for a subtotal row, `0` for a real value) to disambiguate reliably.
- Engine support varies: Postgres, SQL Server, Oracle, Snowflake, and BigQuery support `ROLLUP`/`CUBE`/`GROUPING SETS` in standard form; MySQL only supports `ROLLUP` (via `GROUP BY ... WITH ROLLUP` — different syntax position) and has no native `CUBE`/`GROUPING SETS`; SQLite supports neither and needs manual `UNION ALL`.
- Forgetting that subtotal rows exist in the result set is a common bug when the output feeds straight into further aggregation or a chart — filter with `WHERE GROUPING(col) = 0` if you only want the leaf-level rows.

---

### 2.7 Gotcha Interview Problems: Nth Highest Salary, Gaps and Islands

**Nth-highest value.** The second-highest-salary trick in section 1.7 (Q3) only generalizes to "second." For an arbitrary `N`, window functions are the clean solution — and you must decide up front whether ties should consume one rank (`DENSE_RANK`) or several (`RANK`).

```sql
-- Nth-highest salary, generalized, ties treated as one rank (two $100k earners
-- both count as "1st", so the "2nd highest" is the next distinct salary value)
SELECT DISTINCT salary
FROM (
  SELECT salary, DENSE_RANK() OVER (ORDER BY salary DESC) AS rnk
  FROM employees
) ranked
WHERE rnk = 3;   -- 3rd highest; swap in any N

-- Fully portable version with no window functions (correlated subquery,
-- counts distinct salaries strictly greater than the candidate row's salary)
SELECT DISTINCT salary
FROM employees e1
WHERE (
  SELECT COUNT(DISTINCT salary) FROM employees e2 WHERE e2.salary > e1.salary
) = 2;   -- N - 1, i.e. 2 => this is the 3rd-highest salary
```

**Gaps and islands.** "Gaps" = missing values in what should be a contiguous sequence (missing dates, missing invoice numbers). "Islands" = maximal runs of consecutive values (or consecutive same-status rows) that should be collapsed into one group — e.g. a user's consecutive-day login streaks, or a machine's contiguous uptime windows.

```sql
-- logins(user_id, login_date) -- find each user's consecutive-day login streaks ("islands")
WITH numbered AS (
  SELECT user_id, login_date,
         ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY login_date) AS rn
  FROM logins
),
islands AS (
  SELECT user_id, login_date,
         -- subtracting a running row number from the date is constant
         -- for any run of CONSECUTIVE dates -- the classic "islands" trick
         login_date - (rn * INTERVAL '1 day') AS island_key
  FROM numbered
)
SELECT user_id,
       MIN(login_date) AS streak_start,
       MAX(login_date) AS streak_end,
       COUNT(*)        AS streak_length
FROM islands
GROUP BY user_id, island_key
ORDER BY user_id, streak_start;

-- Gaps: find missing calendar dates in an expected daily sequence (Postgres)
WITH expected AS (
  SELECT generate_series(MIN(login_date), MAX(login_date), INTERVAL '1 day')::date AS expected_date
  FROM logins
)
SELECT e.expected_date AS missing_date
FROM expected e
LEFT JOIN logins l ON e.expected_date = l.login_date
WHERE l.login_date IS NULL;
```

**Pitfalls**
- The `date - (rn * INTERVAL)` island trick assumes at most one row per `(user_id, login_date)` — dedupe first, or the running row number drifts and breaks the constant-offset property.
- `generate_series` is Postgres-specific; other engines need an auxiliary calendar/numbers table (or a recursive CTE) to materialize the "expected" sequence for a gaps query.
- `RANK()` vs `DENSE_RANK()` for "Nth highest" is a common trap: with ties at the top, `RANK()` can skip values entirely (1,1,3,...) so `rnk = 2` returns zero rows, whereas `DENSE_RANK()` (1,1,2,...) reliably returns the Nth *distinct* value — clarify which behavior the interviewer wants before coding.

---

### 2.8 Views: Regular vs Materialized Views

**Concept.** A **view** is a named, stored *query* — a virtual table with no data of its own. Every time it's referenced, the engine substitutes (and typically optimizes/inlines) the underlying query, so a view is always as fresh as the base tables but costs exactly what the underlying query costs. A **materialized view** stores the query's *result set* physically on disk, so reads are fast, but the data goes stale the moment the base tables change and must be explicitly (or periodically) refreshed.

```sql
-- Regular view: pure convenience/security -- no data stored, always live
CREATE VIEW active_customers AS
SELECT customer_id, name, region
FROM customers
WHERE is_active = TRUE;

SELECT * FROM active_customers WHERE region = 'East';  -- expands to the query above every time

-- Materialized view: stores the result physically; must be refreshed to see new data
CREATE MATERIALIZED VIEW mv_daily_revenue AS
SELECT order_date, SUM(amount) AS total_revenue
FROM orders
GROUP BY order_date;

-- Refresh (Postgres): default REFRESH locks out reads while it runs;
-- CONCURRENTLY avoids the lock but requires a unique index on the materialized view
REFRESH MATERIALIZED VIEW mv_daily_revenue;
REFRESH MATERIALIZED VIEW CONCURRENTLY mv_daily_revenue;
```

| | View | Materialized View |
|---|---|---|
| Storage | None (virtual, recomputed each query) | Physical (stores the result set) |
| Freshness | Always current | Stale until refreshed |
| Query speed | Same cost as the underlying query | Fast (pre-computed) |
| Write/maintenance overhead | None | Refresh cost (full or incremental, depending on engine) |
| Typical use | Encapsulate complex joins, row/column-level security, readability | Expensive aggregations queried repeatedly (dashboards, reporting) |

**Pitfalls**
- A view does not inherently speed anything up — it's a syntactic/security convenience layered over the same underlying query cost; a materialized view is the actual performance tool.
- Materialized views need an explicit refresh strategy (manual `REFRESH`, a cron/orchestrator job, or engine-native incremental refresh where supported) — forgetting to schedule refreshes silently turns a "fast" dashboard into a stale one.
- Postgres's `REFRESH MATERIALIZED VIEW CONCURRENTLY` requires a unique index on the materialized view or it errors outright.
- Deeply nested views ("a view built on a view built on a view") can obscure what's actually being scanned — the optimizer usually flattens them, but `EXPLAIN` output on deeply nested views can become hard to read and reason about.

---

### 2.9 Stored Procedures, Functions, and Triggers

**Concept.** A **stored procedure**/**function** is a named, precompiled block of SQL/procedural logic stored inside the database and invoked by name, moving some logic from the application layer into the database itself. A **trigger** is a procedure the engine invokes *automatically* in response to a DML event (`INSERT`/`UPDATE`/`DELETE`) on a table, without any explicit call from application code.

```sql
-- Postgres function: encapsulate a computation, callable like a built-in
CREATE FUNCTION get_customer_total(p_customer_id INT)
RETURNS DECIMAL AS $$
  SELECT COALESCE(SUM(amount), 0)
  FROM orders
  WHERE customer_id = p_customer_id;
$$ LANGUAGE sql;

SELECT get_customer_total(42);

-- Trigger: automatically stamp an updated_at column on every row change
CREATE FUNCTION set_updated_at() RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_orders_updated_at
BEFORE UPDATE ON orders
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();

-- SQL Server stored procedure example
CREATE PROCEDURE ApplyDiscount @customer_id INT, @pct DECIMAL(5,2)
AS
BEGIN
  UPDATE orders SET amount = amount * (1 - @pct / 100)
  WHERE customer_id = @customer_id;
END;

EXEC ApplyDiscount @customer_id = 42, @pct = 10;
```

**Pitfalls**
- Business logic hidden inside triggers is harder to version-control, unit-test, and code-review than application code, and creates side effects that surprise the next engineer — a plain-looking `UPDATE` can silently cascade into other writes. Many teams deliberately push logic to the application/orchestration layer for exactly this reason.
- Triggers firing once per row can devastate bulk-load performance (a 1M-row `INSERT` fires the trigger 1M times) — bulk operations often need the trigger temporarily disabled or a batch-aware alternative.
- Stored procedures reduce client-server round trips and can enable centralized validation enforced across every client, but they reduce portability (procedural SQL dialects — PL/pgSQL, T-SQL, PL/SQL — are not interchangeable across engines) and move logic out of the application's normal test/deploy pipeline.
- Interviewers rarely probe deep procedural syntax; they're usually checking that you know the concept, can name realistic use cases (auditing, maintaining a denormalized/cached column, enforcing a constraint too complex for `CHECK`), and can articulate the tradeoffs above.

---

### 2.10 Interview Questions — Advanced SQL

**Q1. What's the difference between `RANK()`, `DENSE_RANK()`, and `ROW_NUMBER()`?**
A: `ROW_NUMBER()` assigns a strictly increasing unique integer per row in partition order, breaking ties arbitrarily. `RANK()` gives tied rows the same rank but leaves a gap in the sequence afterward (1,1,3). `DENSE_RANK()` gives tied rows the same rank with no gap afterward (1,1,2).

**Q2. Given `sales(salesperson, region, amount)`, write a query to get the top 2 salespeople by amount in each region.**
```sql
SELECT salesperson, region, amount FROM (
  SELECT salesperson, region, amount,
         ROW_NUMBER() OVER (PARTITION BY region ORDER BY amount DESC) AS rn
  FROM sales
) t
WHERE rn <= 2;
```

**Q3. Write a query computing a 3-month moving average of `revenue` from `monthly_revenue(month, revenue)`.**
```sql
SELECT month, revenue,
       AVG(revenue) OVER (
         ORDER BY month
         ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
       ) AS moving_avg_3m
FROM monthly_revenue;
```

**Q4. Why can't you filter directly on a window function's output in the same `SELECT`'s `WHERE` clause?**
A: Window functions are logically evaluated after `WHERE`/`GROUP BY`/`HAVING` in the query-processing order, so their results don't exist yet when `WHERE` runs. You must wrap the query in a subquery or CTE and filter in an outer query, or use `QUALIFY` (Snowflake/BigQuery-specific shortcut).

**Q5. What is a recursive CTE, and when would you use one? Write one to find all descendants of a category in `categories(category_id, name, parent_id)`.**
A: A recursive CTE repeatedly unions an anchor query with a recursive term that references the CTE itself, used for traversing hierarchical/graph data (org charts, category trees, bill of materials) of unknown depth.
```sql
WITH RECURSIVE subcategories AS (
  SELECT category_id, name, parent_id
  FROM categories
  WHERE category_id = 10          -- root category

  UNION ALL

  SELECT c.category_id, c.name, c.parent_id
  FROM categories c
  JOIN subcategories s ON c.parent_id = s.category_id
)
SELECT * FROM subcategories;
```

**Q6. What's the difference between a CTE and a subquery in terms of performance?**
A: Logically they can express the same thing, but physical execution differs by engine/version. Historically Postgres (pre-v12) always materialized CTEs as an "optimization fence," meaning the CTE ran independently and couldn't have predicates pushed into it — sometimes slower than an equivalent subquery. Postgres 12+ inlines simple, non-recursive CTEs like subqueries unless marked `MATERIALIZED`. SQL Server and modern MySQL typically inline non-recursive CTEs. The safe interview answer: "it depends on the engine and version — verify with `EXPLAIN`."

**Q7. Write a query to pivot `sales(product, month, revenue)` into one row per product with a column per month, using only standard SQL (no PIVOT keyword).**
```sql
SELECT product,
  SUM(CASE WHEN month = 'Jan' THEN revenue ELSE 0 END) AS Jan,
  SUM(CASE WHEN month = 'Feb' THEN revenue ELSE 0 END) AS Feb,
  SUM(CASE WHEN month = 'Mar' THEN revenue ELSE 0 END) AS Mar
FROM sales
GROUP BY product;
```

**Q8. What does `EXPLAIN ANALYZE` show that plain `EXPLAIN` does not?**
A: Plain `EXPLAIN` shows the planner's *estimated* plan and costs without running the query. `EXPLAIN ANALYZE` actually executes the query and reports real elapsed time and real row counts alongside the estimates, letting you spot cases where the optimizer's row-count estimates are badly wrong (usually due to stale statistics), which is the most common root cause of a bad plan choice.

**Q9. Explain SARGability with an example, and rewrite a non-SARGable predicate to be SARGable.**
A: A predicate is SARGable if the database can evaluate it using an index seek/range scan directly on stored values, without first transforming every row. `WHERE YEAR(order_date) = 2026` is non-SARGable because `YEAR()` must be computed per row, defeating a B-tree index on `order_date`. Rewrite: `WHERE order_date >= '2026-01-01' AND order_date < '2027-01-01'`, which is SARGable and can use the index.

**Q10. What is the N+1 query problem and how do you fix it?**
A: It happens when code fetches a list of N parent rows, then issues one additional query per row to fetch related child data (N extra round-trips), instead of a single batched query. Fix with an explicit `JOIN`, a single `WHERE child_fk IN (...)` batched query, or ORM eager-loading (`select_related`/`prefetch_related` in Django, `joinedload` in SQLAlchemy).

**Q11. What's the difference between a B-tree index and a hash index? When would you choose each?**
A: A B-tree index stores keys in sorted order in a balanced tree, supporting equality, range (`<`,`>`,`BETWEEN`), and ordered-scan queries — the general-purpose default. A hash index stores a hash of the key pointing to the row, supporting only exact-equality lookups (O(1)) but no range queries or sorting. Choose hash indexes only for pure equality-lookup workloads where range queries are never needed (rare in practice; most engines default to B-tree and don't even always support/recommend standalone hash indexes for durability/crash-safety reasons historically in Postgres).

**Q12. Why might adding an index not improve (or even hurt) query performance?**
A: If the column has low selectivity (few distinct values relative to row count), the optimizer may correctly decide a sequential scan is cheaper than many random index lookups. If table statistics are stale, the optimizer may misjudge the index's benefit. Additionally, indexes always add overhead to every `INSERT`/`UPDATE`/`DELETE` that touches the indexed column, which can hurt write-heavy workloads even if reads improve.

**Q13. Given `orders(order_id, customer_id, order_date, amount)`, write a query that returns each order alongside the difference in `amount` from that customer's previous order (by date).**
```sql
SELECT order_id, customer_id, order_date, amount,
       amount - LAG(amount) OVER (
         PARTITION BY customer_id ORDER BY order_date
       ) AS change_from_prev_order
FROM orders;
```

**Q14. What is a covering index, and why does it speed up queries?**
A: A covering index includes every column referenced by a query (in the key and/or as included/non-key columns), letting the engine satisfy the entire query from the index structure itself ("index-only scan") without a second lookup into the base table's heap/pages — reducing I/O significantly, especially for wide tables with narrow, frequent queries.

**Q15. Explain nested loop join, hash join, and merge join, and when the optimizer tends to choose each.**
A: Nested loop join iterates the outer set and, for each row, probes the inner set (often via an index) — efficient when the outer set is small or a good index exists on the inner join key. Hash join builds an in-memory hash table on one (typically smaller) input and probes it with the other — efficient for large, unsorted, roughly equal-size inputs without useful indexes. Merge join walks both inputs in sorted order simultaneously — efficient when both inputs are already sorted (e.g., via an index) on the join key, avoiding an extra sort step.

**Q16. What's the difference between `ROLLUP` and `CUBE`, and what does `GROUPING()` let you do that plain `NULL` checking can't?**
A: `ROLLUP(a, b)` produces hierarchical subtotals right-to-left — rows for `(a,b)`, a subtotal per `a` (with `b = NULL`), and one grand total (both `NULL`) — so column order changes the output. `CUBE(a, b)` produces every possible combination of subtotals — `(a,b)`, `a` alone, `b` alone, and the grand total — regardless of column order. `GROUPING(col)` returns `1` on a subtotal row and `0` on a real value, letting you distinguish "this NULL means a subtotal" from "this NULL is a genuine value in the data," which a plain `WHERE col IS NULL` check cannot do.

**Q17. Write a query for the Nth-highest salary in `employees(emp_id, salary)`, generalized for any N and handling ties sensibly.**
```sql
SELECT DISTINCT salary
FROM (
  SELECT salary, DENSE_RANK() OVER (ORDER BY salary DESC) AS rnk
  FROM employees
) ranked
WHERE rnk = 3;  -- 3rd highest; DENSE_RANK treats tied salaries as one rank
```

**Q18. Explain the "gaps and islands" problem and sketch the general technique for finding islands (consecutive runs).**
A: "Islands" are maximal runs of consecutive or same-status values that should be grouped as one (e.g., a user's consecutive-day login streak); "gaps" are missing values in an otherwise-expected contiguous sequence (e.g., missing dates). The standard islands technique: assign each row a sequential `ROW_NUMBER()` within its partition ordered by the sequence column, then subtract that row number (scaled to the same unit) from the sequence value — the result is constant for any run of truly consecutive values, giving a groupable "island key" (e.g., `login_date - (rn * INTERVAL '1 day')`).

**Q19. What's the difference between a view and a materialized view, and when would you choose a materialized view?**
A: A view is a stored query with no data of its own — always live, but costs exactly what the underlying query costs every time it's referenced. A materialized view physically stores the query's result, so reads are fast but the data is stale until explicitly refreshed. Choose a materialized view when the same expensive aggregation/join is queried repeatedly (dashboards, reporting) and some staleness is acceptable in exchange for speed; choose a plain view when you just want to encapsulate logic, enforce row/column-level security, or improve readability without any performance goal.

**Q20. What is a trigger, name a legitimate use case, and describe one risk of relying on them heavily?**
A: A trigger is a procedure the database invokes automatically on `INSERT`/`UPDATE`/`DELETE` events, without an explicit call from application code. Legitimate use: auto-maintaining an `updated_at` timestamp, an audit log, or a denormalized/cached column that must always match its source. Risk: triggers hide logic and side effects outside the application's normal code path (harder to test/review/version), can cascade unexpectedly across tables, and firing once per row can severely slow down bulk loads.

---

## 3. Database Design and Theory

### 3.1 Normalization (1NF, 2NF, 3NF, BCNF) and Denormalization

**Concept.** Normalization organizes tables to reduce redundancy and prevent update/insert/delete anomalies, by progressively enforcing stricter rules about functional dependencies.

**Unnormalized example:**

| order_id | customer_name | customer_email | product | product_price | quantity |
|---|---|---|---|---|---|
| 1 | Alice | alice@x.com | Widget | 9.99 | 2 |
| 1 | Alice | alice@x.com | Gadget | 19.99 | 1 |
| 2 | Bob | bob@x.com | Widget | 9.99 | 5 |

**1NF (First Normal Form):** Each column holds atomic (indivisible) values; no repeating groups or arrays in a single cell; each row is uniquely identifiable.

*Violation example:* a column storing `"Widget,Gadget"` as a comma-separated string violates 1NF — fix by splitting into separate rows or a separate line-items table.

**2NF (Second Normal Form):** Must be in 1NF, and every non-key column must depend on the **entire** primary key (relevant when the PK is composite). Eliminates *partial* dependencies.

*Violation:* if `(order_id, product)` is the composite key, but `customer_name`/`customer_email` depend only on `order_id` (not on `product`), that's a partial dependency — violates 2NF.

**3NF (Third Normal Form):** Must be in 2NF, and no non-key column depends on another non-key column (eliminates *transitive* dependencies). `product_price` depending on `product` (not directly on the order) is transitive — should live in a separate `products` table.

**Normalized (3NF) result:**

```sql
CREATE TABLE customers (
  customer_id INT PRIMARY KEY,
  name VARCHAR(100),
  email VARCHAR(100)
);

CREATE TABLE products (
  product_id INT PRIMARY KEY,
  product_name VARCHAR(100),
  price DECIMAL(10,2)
);

CREATE TABLE orders (
  order_id INT PRIMARY KEY,
  customer_id INT REFERENCES customers(customer_id),
  order_date DATE
);

CREATE TABLE order_items (
  order_id INT REFERENCES orders(order_id),
  product_id INT REFERENCES products(product_id),
  quantity INT,
  PRIMARY KEY (order_id, product_id)
);
```

**BCNF (Boyce-Codd Normal Form):** A stricter version of 3NF — for every functional dependency `A -> B`, `A` must be a candidate key (super-key). Handles edge cases 3NF misses when a table has multiple overlapping candidate keys.

**Normal Forms summary:**

| Form | Rule | Fixes |
|---|---|---|
| 1NF | Atomic values, no repeating groups | Multi-valued columns |
| 2NF | 1NF + no partial dependency on part of a composite key | Redundancy tied to part of the key |
| 3NF | 2NF + no transitive dependency (non-key depending on non-key) | Redundancy from derived/lookup attributes |
| BCNF | Every determinant is a candidate key | Anomalies 3NF still allows with overlapping keys |

**Denormalization — the tradeoff.** Deliberately reintroducing redundancy (e.g., storing `customer_name` directly on the `orders` row, or precomputed aggregate columns) to avoid expensive joins at read time.

| | Normalized | Denormalized |
|---|---|---|
| Data redundancy | Minimal | Higher (duplicated values) |
| Write complexity | Simple, single-place updates | Must update multiple copies (or accept staleness) |
| Read performance | Requires joins | Faster — data pre-joined/flattened |
| Storage | Smaller | Larger |
| Best for | OLTP (frequent writes, consistency-critical) | OLAP/reporting (read-heavy, latency-sensitive) |

**Pitfalls**
- Over-normalizing an OLAP/reporting schema forces expensive multi-way joins on every dashboard query — analytical schemas intentionally denormalize (star schema) for read speed.
- Denormalizing an OLTP schema risks update anomalies — if `customer_name` is duplicated across 10,000 order rows and the customer renames, you must update all 10,000 rows or accept inconsistency.
- BCNF is rarely tested for beyond a definitional question in interviews, but knowing the distinction from 3NF ("every determinant must be a *candidate* key, not just *the* primary key") signals genuine depth.

---

### 3.2 ACID, Transactions, Isolation Levels, Deadlocks

**Concept — ACID:**

| Property | Meaning |
|---|---|
| **Atomicity** | A transaction's operations either all succeed or all roll back — no partial effects |
| **Consistency** | A transaction moves the database from one valid state to another, respecting all constraints |
| **Isolation** | Concurrent transactions don't see each other's uncommitted intermediate state (degree depends on isolation level) |
| **Durability** | Once committed, changes survive crashes/power loss (typically via write-ahead logging) |

```sql
BEGIN TRANSACTION;

UPDATE accounts SET balance = balance - 100 WHERE account_id = 1;
UPDATE accounts SET balance = balance + 100 WHERE account_id = 2;

-- If both succeed:
COMMIT;
-- If anything fails / business rule violated:
-- ROLLBACK;
```

```sql
-- Savepoints allow partial rollback within a larger transaction
BEGIN;
UPDATE inventory SET qty = qty - 1 WHERE product_id = 5;
SAVEPOINT after_inventory;
UPDATE orders SET status = 'shipped' WHERE order_id = 42;
-- something goes wrong with the shipping update only:
ROLLBACK TO SAVEPOINT after_inventory;
COMMIT;
```

**Isolation levels** — control which "concurrency anomalies" are allowed. Anomalies, defined:

| Anomaly | Definition |
|---|---|
| **Dirty read** | Reading another transaction's *uncommitted* changes |
| **Non-repeatable read** | Re-reading the same row within a transaction returns different data because another transaction committed a change in between |
| **Phantom read** | Re-running the same range query within a transaction returns a different *set of rows* because another transaction inserted/deleted matching rows |

| Isolation Level | Dirty Read | Non-Repeatable Read | Phantom Read | Notes |
|---|---|---|---|---|
| **Read Uncommitted** | Possible | Possible | Possible | Rarely used; fastest, weakest guarantees |
| **Read Committed** | Prevented | Possible | Possible | Default in Postgres, SQL Server, Oracle |
| **Repeatable Read** | Prevented | Prevented | Possible (standard defines it may occur; MySQL InnoDB actually prevents most phantoms via multi-version concurrency control) | Default in MySQL InnoDB |
| **Serializable** | Prevented | Prevented | Prevented | Strongest; transactions behave as if executed one at a time; highest risk of contention/rollback |

**Concrete before/after examples — two interleaved sessions:**

```sql
-- DIRTY READ (only possible at READ UNCOMMITTED)
-- Session A                                     Session B
BEGIN;
UPDATE accounts SET balance = 500
  WHERE account_id = 1;   -- uncommitted!
                                                  BEGIN;
                                                  SELECT balance FROM accounts
                                                    WHERE account_id = 1;  -- reads 500 (dirty)
ROLLBACK;   -- the update never happened
                                                  -- Session B now acted on a balance (500)
                                                  -- that never existed in the committed data
```

```sql
-- NON-REPEATABLE READ (possible at READ COMMITTED; prevented by REPEATABLE READ)
-- Session A                                     Session B
BEGIN;
SELECT balance FROM accounts
  WHERE account_id = 1;   -- returns 1000
                                                  BEGIN;
                                                  UPDATE accounts SET balance = 800
                                                    WHERE account_id = 1;
                                                  COMMIT;
SELECT balance FROM accounts
  WHERE account_id = 1;   -- returns 800 -- same txn, same row, different value!
COMMIT;
```

```sql
-- PHANTOM READ (possible under the standard's REPEATABLE READ; prevented by SERIALIZABLE)
-- Session A                                     Session B
BEGIN;
SELECT COUNT(*) FROM accounts
  WHERE balance > 500;    -- returns 3 rows
                                                  BEGIN;
                                                  INSERT INTO accounts (account_id, balance)
                                                    VALUES (99, 600);
                                                  COMMIT;
SELECT COUNT(*) FROM accounts
  WHERE balance > 500;    -- returns 4 rows -- a new row "phantom-appeared" mid-transaction
COMMIT;
```

**Lost update** — a fourth anomaly worth naming even though it isn't in the classic dirty/non-repeatable/phantom trio: two transactions both read the same row, both compute a new value based on that read, and both write it back — the second write silently overwrites the first, "losing" its update.

```sql
-- LOST UPDATE (possible under READ COMMITTED with naive read-then-write logic)
-- Session A                                     Session B
BEGIN;                                           BEGIN;
SELECT balance FROM accounts                     SELECT balance FROM accounts
  WHERE account_id = 1;   -- reads 1000             WHERE account_id = 1;   -- also reads 1000
-- app computes 1000 + 100 = 1100                -- app computes 1000 + 50 = 1050
UPDATE accounts SET balance = 1100
  WHERE account_id = 1;
COMMIT;
                                                  UPDATE accounts SET balance = 1050
                                                    WHERE account_id = 1;
                                                  COMMIT;
                                                  -- final balance is 1050 -- Session A's
                                                  -- +100 update is silently lost
```
Fix: use an atomic in-place update (`UPDATE accounts SET balance = balance + 100 WHERE account_id = 1`) instead of read-then-write in application code, or take an explicit lock with `SELECT ... FOR UPDATE` before recomputing, or rely on `SERIALIZABLE`/optimistic concurrency (a version column checked on write) to detect the conflict.

```sql
-- Setting isolation level (syntax varies by engine)
SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;
BEGIN;
  SELECT * FROM accounts WHERE account_id = 1;
  -- ... business logic ...
  UPDATE accounts SET balance = balance - 100 WHERE account_id = 1;
COMMIT;
```

**Deadlocks.** Occur when two transactions each hold a lock the other needs, and each waits for the other indefinitely — a circular wait.

*Example scenario, in words:* Transaction A locks row 1 then tries to lock row 2. Transaction B locks row 2 then tries to lock row 1. Neither can proceed. The database's deadlock detector periodically checks the "waits-for" graph for cycles and kills one transaction (the "victim," usually the one with less work invested or lowest cost to roll back) with an error, letting the other proceed.

```sql
-- Transaction A                      -- Transaction B
BEGIN;                                BEGIN;
UPDATE accounts                       UPDATE accounts
  SET balance = balance - 10           SET balance = balance - 20
  WHERE account_id = 1;                 WHERE account_id = 2;
-- (holds lock on row 1)               -- (holds lock on row 2)
UPDATE accounts                       UPDATE accounts
  SET balance = balance + 10           SET balance = balance + 20
  WHERE account_id = 2;  -- BLOCKS      WHERE account_id = 1;  -- BLOCKS
                                        -- => DEADLOCK detected, one txn is aborted
```

**Avoiding deadlocks:** always acquire locks (update rows) in a **consistent global order** across all transactions (e.g., always by ascending `account_id`); keep transactions short; use appropriate isolation levels; consider `SELECT ... FOR UPDATE` explicitly and consistently ordered.

**Pitfalls**
- Higher isolation = stronger correctness guarantees but more blocking/aborts and lower throughput — always a tradeoff, never a free upgrade.
- MySQL's `REPEATABLE READ` (via InnoDB's MVCC + gap locking) behaves more strictly than the SQL standard's minimum requirement and actually prevents most phantom reads, unlike a naive reading of the standard table above — a nuance worth mentioning to show depth.
- Long-running transactions holding locks (e.g., a transaction left open by a client) are a common real-world cause of cascading blocking, not just genuine deadlocks — distinguish "deadlock" (cyclic, detected and auto-resolved) from "blocking/lock contention" (one waits on another, resolves eventually, no error).

---

### 3.3 Primary/Foreign Keys, Constraints

**Concept.**

| Constraint | Purpose |
|---|---|
| `PRIMARY KEY` | Uniquely identifies each row; implies `UNIQUE` + `NOT NULL`; one per table |
| `FOREIGN KEY` | Enforces referential integrity — a column's value must exist in the referenced table's key column (or be `NULL`) |
| `UNIQUE` | No two rows may share the same value in this column(s); unlike PK, multiple `UNIQUE` constraints allowed, and (in most engines) `NULL` values are exempt from the uniqueness check |
| `NOT NULL` | Column may never store `NULL` |
| `CHECK` | Enforces an arbitrary boolean condition on column values |
| `DEFAULT` | Value used when no explicit value is inserted |

```sql
CREATE TABLE departments (
  dept_id INT PRIMARY KEY,
  dept_name VARCHAR(50) NOT NULL UNIQUE
);

CREATE TABLE employees (
  emp_id INT PRIMARY KEY,
  name VARCHAR(100) NOT NULL,
  dept_id INT,
  salary DECIMAL(10,2) CHECK (salary >= 0),
  hire_date DATE DEFAULT CURRENT_DATE,
  email VARCHAR(100) UNIQUE,
  CONSTRAINT fk_dept FOREIGN KEY (dept_id) REFERENCES departments(dept_id)
    ON DELETE SET NULL
    ON UPDATE CASCADE
);
```

**`ON DELETE`/`ON UPDATE` referential actions:**

| Action | Behavior when parent row is deleted/updated |
|---|---|
| `CASCADE` | Automatically delete/update matching child rows |
| `SET NULL` | Set the child's FK column to `NULL` |
| `SET DEFAULT` | Set the child's FK column to its default value |
| `RESTRICT` / `NO ACTION` | Reject the parent delete/update if matching child rows exist |

**Pitfalls**
- A `FOREIGN KEY` does **not** automatically create an index on the child column in every engine (e.g., not automatic in Postgres) — leaving FK columns unindexed causes slow joins and slow cascading deletes; explicitly index FK columns.
- `UNIQUE` constraints typically allow multiple `NULL`s (NULL is not considered equal to another NULL for uniqueness purposes) in most engines — a frequent surprise.
- Composite primary keys (e.g., `(order_id, product_id)`) are common in junction/bridge tables but can complicate ORMs; consider a surrogate key (`id BIGSERIAL`) plus a `UNIQUE` constraint on the natural composite key when tooling requires a single-column PK.
- Dropping/altering a column referenced by a `FOREIGN KEY` or `CHECK` constraint requires dropping the constraint first in most engines.

---

### 3.4 OLTP vs OLAP, Star vs Snowflake Schema, SCDs

**OLTP vs OLAP:**

| | OLTP (Online Transaction Processing) | OLAP (Online Analytical Processing) |
|---|---|---|
| Purpose | Day-to-day operations (orders, payments, bookings) | Analysis, reporting, BI, ML feature extraction |
| Query pattern | Many short reads/writes touching few rows | Fewer, complex queries scanning/aggregating many rows |
| Schema | Highly normalized (3NF) to avoid write anomalies | Denormalized (star/snowflake) for read speed |
| Storage layout | Row-oriented | Often column-oriented |
| Example systems | PostgreSQL, MySQL, Oracle (transactional workloads) | Snowflake, BigQuery, Redshift, Vertica |
| Typical user | Application backend | Analysts, data scientists, dashboards |

**Star schema.** One central **fact table** (numeric measures + FKs) surrounded by denormalized **dimension tables** (descriptive attributes), resembling a star shape.

```
                dim_date
                   |
dim_customer -- fact_sales -- dim_product
                   |
               dim_store
```

```sql
CREATE TABLE dim_date (
  date_key INT PRIMARY KEY,
  full_date DATE,
  month VARCHAR(10),
  quarter VARCHAR(2),
  year INT
);

CREATE TABLE dim_product (
  product_key INT PRIMARY KEY,
  product_name VARCHAR(100),
  category VARCHAR(50),   -- denormalized: category name lives directly here
  brand VARCHAR(50)
);

CREATE TABLE fact_sales (
  sale_id BIGINT PRIMARY KEY,
  date_key INT REFERENCES dim_date(date_key),
  product_key INT REFERENCES dim_product(product_key),
  customer_key INT REFERENCES dim_customer(customer_key),
  store_key INT REFERENCES dim_store(store_key),
  quantity INT,
  revenue DECIMAL(12,2)
);

-- Typical star-schema analytical query
SELECT d.year, p.category, SUM(f.revenue) AS total_revenue
FROM fact_sales f
JOIN dim_date d ON f.date_key = d.date_key
JOIN dim_product p ON f.product_key = p.product_key
GROUP BY d.year, p.category
ORDER BY d.year, total_revenue DESC;
```

**Snowflake schema.** Like a star schema, but dimension tables are **further normalized** into sub-dimensions (e.g., `dim_product` splits out `dim_category` and `dim_brand` as separate normalized tables) — resembles a snowflake's branching arms.

```
dim_category -- dim_product -- fact_sales -- dim_customer
                                    |
                                dim_date
```

| | Star Schema | Snowflake Schema |
|---|---|---|
| Dimension normalization | Denormalized (flat) | Normalized (sub-dimensions) |
| Query complexity | Fewer joins, simpler | More joins, more complex |
| Query performance | Generally faster | Can be slower (more joins) |
| Storage | More redundancy | Less redundancy, smaller |
| Maintenance | Simpler to understand | More flexible for shared sub-dimensions |

**Slowly Changing Dimensions (SCDs)** — strategies for handling dimension attribute changes over time (e.g., a customer moves regions).

| Type | Strategy | Historical tracking? |
|---|---|---|
| **SCD Type 0** | Never update — original value retained forever | No changes allowed |
| **SCD Type 1** | Overwrite the old value in place | No history kept |
| **SCD Type 2** | Insert a new row with a new surrogate key, versioning/date-ranging columns (`effective_date`, `end_date`, `is_current`) | Full history preserved |
| **SCD Type 3** | Add a new column to store the previous value (e.g., `prev_region`) | Only one prior value kept |

```sql
-- SCD Type 2 example: dim_customer tracks history via effective dating
CREATE TABLE dim_customer (
  customer_sk BIGINT PRIMARY KEY,   -- surrogate key, changes each version
  customer_id INT,                   -- natural/business key, stable across versions
  name VARCHAR(100),
  region VARCHAR(50),
  effective_date DATE,
  end_date DATE,                     -- NULL or '9999-12-31' for current row
  is_current BOOLEAN
);

-- Customer 42 moves from 'West' to 'East' on 2026-07-28
UPDATE dim_customer
SET end_date = '2026-07-27', is_current = FALSE
WHERE customer_id = 42 AND is_current = TRUE;

INSERT INTO dim_customer (customer_sk, customer_id, name, region, effective_date, end_date, is_current)
VALUES (9101, 42, 'Alice', 'East', '2026-07-28', NULL, TRUE);

-- Query: what region was customer 42 attributed to for a sale on 2025-05-01?
SELECT * FROM dim_customer
WHERE customer_id = 42
  AND '2025-05-01' BETWEEN effective_date AND COALESCE(end_date, '9999-12-31');
```

**Pitfalls**
- Fact tables should store **foreign keys and measures**, not descriptive text — descriptive attributes belong in dimensions; violating this reintroduces normalization problems into the analytical layer.
- SCD Type 1 is simplest but destroys history — wrong choice if the business needs "what did this look like at the time of the transaction" (common requirement in finance/compliance/ML training-data reproducibility).
- SCD Type 2 tables grow unboundedly and require careful surrogate-key joins in fact tables (join on the surrogate key valid *at transaction time*, not the natural key) — a frequent source of double-counting bugs if done incorrectly.
- Snowflaking looks "more correct" via normalization instinct, but in OLAP contexts it usually just adds join overhead for little benefit — star schema is the more common real-world default.

---

### 3.5 Interview Questions — Database Design and Theory

**Q1. Explain 1NF, 2NF, and 3NF with a concrete example of a table that violates each.**
A: 1NF requires atomic values (a table storing `"Widget,Gadget"` in one cell violates it). 2NF (given a composite key) requires every non-key attribute to depend on the *whole* key, not part of it (a table keyed on `(order_id, product_id)` storing `customer_name`, which depends only on `order_id`, violates 2NF). 3NF forbids non-key attributes depending on other non-key attributes (storing `product_price` in the `orders` table, where price depends on `product` rather than directly on the order, violates 3NF).

**Q2. When would you intentionally denormalize a schema, and what's the tradeoff?**
A: Denormalize in read-heavy, latency-sensitive contexts (dashboards, analytical/OLAP schemas, caching layers) to avoid expensive joins. Tradeoff: redundant data must be kept in sync across copies on every write, risking staleness/inconsistency and using more storage — acceptable in OLAP where writes are batched (ETL), risky in OLTP where writes are frequent and consistency is critical.

**Q3. What are the four ACID properties? Give a real scenario where violating each would cause a problem.**
A: Atomicity — a funds transfer that debits one account but crashes before crediting the other, without atomicity money disappears. Consistency — an insert that violates a `CHECK` constraint (e.g., negative balance) must be rejected, else the DB reaches an invalid state. Isolation — two concurrent withdrawals both reading the same starting balance and independently deducting can overdraw the account without proper isolation. Durability — a committed transaction lost after a server crash because it was never flushed to persistent storage.

**Q4. What is a phantom read, and which isolation level is the minimum required to prevent it per the SQL standard?**
A: A phantom read happens when a transaction re-runs the same range query and sees a different *set of rows* because another committed transaction inserted or deleted rows matching that range in between. `SERIALIZABLE` is the only standard isolation level guaranteed to prevent it (though some engines' `REPEATABLE READ`, e.g., MySQL InnoDB via MVCC/gap locks, prevent most phantoms in practice too).

**Q5. Describe a deadlock scenario and how a database detects/resolves it.**
A: Transaction A locks row 1 and waits for row 2; Transaction B locks row 2 and waits for row 1 — a circular wait. The database periodically builds a "waits-for" graph among blocked transactions and detects the cycle, then aborts one transaction (the "victim," raising a deadlock error to the client) so the other can proceed. Prevention: always acquire locks/update rows in a consistent order across all transactions, and keep transactions short.

**Q6. What's the difference between a `UNIQUE` constraint and a `PRIMARY KEY`?**
A: Both enforce uniqueness, but a table can have only one `PRIMARY KEY` (which also implies `NOT NULL`) versus multiple `UNIQUE` constraints. Most engines allow multiple `NULL` values in a `UNIQUE` column (NULL isn't considered equal to NULL) but never allow any `NULL` in a `PRIMARY KEY` column.

**Q7. Design a star schema for an e-commerce `fact_sales` table. What are the dimensions and what are the measures?**
A: Dimensions: `dim_date` (order date attributes), `dim_customer` (customer attributes), `dim_product` (product/category/brand attributes), `dim_store`/`dim_channel` (sales channel). Measures (numeric, additive facts in `fact_sales`): `quantity`, `unit_price`, `discount`, `revenue`, `cost`, `profit`. The fact table stores FKs to each dimension plus these measures — one row per sale line item.

**Q8. What is a Slowly Changing Dimension, and how does SCD Type 2 differ from Type 1?**
A: An SCD is a dimension attribute that changes over time (e.g., a customer's address or region). Type 1 overwrites the old value in place, losing history. Type 2 inserts a brand-new versioned row (new surrogate key) with `effective_date`/`end_date`/`is_current` columns, preserving full history so historical facts can still be joined to the dimension value that was accurate *at the time of the transaction*.

**Q9. Why is a star schema generally preferred over a snowflake schema for BI/reporting workloads?**
A: A star schema's denormalized dimensions require fewer joins per query, which is faster and simpler for analysts and BI tools to reason about. A snowflake schema normalizes dimensions further, saving some storage and reducing redundancy, but adds extra joins that usually outweigh the storage savings in read-heavy analytical workloads — storage is cheap, join latency is not.

**Q10. Why should foreign key columns typically be indexed even though the FK constraint itself doesn't require it in every engine?**
A: Without an index, every join on the FK column, and every cascading delete/update check against the child table, requires a full table scan of the child table to find matching rows — devastating for performance as the child table grows. Explicitly indexing FK columns keeps joins and referential-integrity checks fast.

**Q11. What is the difference between OLTP and OLAP systems, and how does that difference drive schema design choices?**
A: OLTP systems handle frequent, small, transactional read/write operations (e.g., placing an order) and are normalized to minimize redundancy and avoid update anomalies, since consistency and write speed matter most. OLAP systems handle infrequent, large, complex analytical queries over historical data (e.g., quarterly revenue by region) and are denormalized (star/snowflake) and often column-oriented, since read speed over large scans matters most and writes are batched via ETL rather than continuous.

**Q12. A table has columns `(employee_id, project_id, employee_name, project_name, hours_worked)` with a composite primary key `(employee_id, project_id)`. What normalization violations exist, and how would you fix them?**
A: `employee_name` depends only on `employee_id` (part of the composite key, not the whole key) — a partial dependency violating 2NF. Similarly `project_name` depends only on `project_id`. Fix: split into `employees(employee_id, employee_name)`, `projects(project_id, project_name)`, and a bridge table `assignments(employee_id, project_id, hours_worked)`.

**Q13. What does `ON DELETE CASCADE` do, and when is it dangerous?**
A: It automatically deletes child rows when the referenced parent row is deleted. It's dangerous when cascades chain across multiple tables unexpectedly (deleting a `customers` row silently deletes all their `orders`, which silently deletes all `order_items`, etc.) — a single delete can have far larger blast radius than intended; often `RESTRICT` (require explicit child cleanup first) or `SET NULL` is safer for high-stakes data.

**Q14. Explain Read Committed vs Repeatable Read with a concrete example query sequence.**
A: Under Read Committed, if Transaction A reads a row, then Transaction B commits an update to that same row, and Transaction A reads it again *within the same transaction*, A will see B's new committed value (a non-repeatable read). Under Repeatable Read, A's second read within the same transaction returns the *same* value it saw the first time, regardless of B's intervening commit, because A's transaction operates against a consistent snapshot taken at (engine-dependent) transaction or statement start.

**Q15. What is BCNF and how does it differ from 3NF?**
A: BCNF requires that for every non-trivial functional dependency `X -> Y`, `X` must be a candidate key (a superkey) of the relation. 3NF has a looser exception allowing a non-key attribute to determine another attribute as long as the determining attribute is part of some candidate key. BCNF eliminates the small set of anomalies that can still slip through 3NF when a table has multiple overlapping candidate keys — it's stricter and sometimes sacrifices some dependency-preservation to achieve full anomaly freedom.

**Q16. Walk through a concrete two-session example of a dirty read, and state which isolation level prevents it.**
A: Session A runs `UPDATE accounts SET balance = 500 WHERE account_id = 1` inside an open transaction (uncommitted). Session B, at `READ UNCOMMITTED`, runs `SELECT balance FROM accounts WHERE account_id = 1` and sees `500`. Session A then rolls back, so `500` never actually existed in the committed database — Session B read data that logically never happened. Any isolation level of `READ COMMITTED` or stricter prevents this, since they only ever expose committed data to other transactions.

**Q17. Give a concrete example of a non-repeatable read that `READ COMMITTED` allows but `REPEATABLE READ` prevents.**
A: Session A opens a transaction and runs `SELECT balance FROM accounts WHERE account_id = 1`, reading `1000`. Session B then updates that row to `800` and commits. Session A re-runs the identical `SELECT` within the *same* transaction: under `READ COMMITTED` it now sees `800` (a different value for the same row in the same transaction); under `REPEATABLE READ` it still sees `1000`, because A's transaction reads against a consistent snapshot established at (engine-dependent) transaction start.

**Q18. What is a "lost update" anomaly, and how do you prevent it?**
A: Two transactions both read the same row's value, both independently compute a new value from that read, and both write it back — the second write overwrites the first without ever seeing it, silently discarding the first transaction's change (e.g., two concurrent `balance = balance_read + delta` updates where only the second `delta` survives). Prevent it by making the update atomic in SQL itself (`UPDATE accounts SET balance = balance + 100 ...` instead of read-then-write in application code), by explicitly locking the row first (`SELECT ... FOR UPDATE`), or via optimistic concurrency control (a version/timestamp column checked in the `WHERE` clause of the write, rejecting the write if the row changed since it was read).

---

## 4. NoSQL and Modern Data Stores

### 4.1 Key-Value, Document, Columnar, and Graph Stores

**Concept.** Relational databases assume a fixed tabular schema and strong consistency; NoSQL stores trade some of those guarantees (schema rigidity, joins, sometimes strong consistency) for horizontal scalability, flexible schemas, or access patterns that map more naturally onto specific data shapes.

| Store Type | Example Systems | Data Model | Best For | Weak For |
|---|---|---|---|---|
| **Key-Value** | Redis, DynamoDB (partition key access), Memcached | Simple key -> opaque value (or rich value types in Redis) | Caching, session storage, rate limiting, feature flags, leaderboard (`ZSET`), pub/sub | Complex queries, relationships, ad hoc filtering |
| **Document** | MongoDB, Couchbase, DynamoDB (document mode) | Self-contained JSON/BSON documents, flexible/nested schema per document | Rapidly evolving schemas, content management, catalogs, semi-structured data | Multi-document transactions (improving, but historically weaker), complex joins across collections |
| **Columnar (wide-column)** | Cassandra, HBase, Bigtable | Rows keyed by a partition/row key, columns grouped into "column families," sparse schema per row | Massive write throughput, time-series, IoT telemetry, systems needing linear horizontal scalability | Ad hoc analytical queries without knowing access patterns upfront; secondary-index-heavy workloads |
| **Graph** | Neo4j, Amazon Neptune, ArangoDB | Nodes + edges with properties; relationships are first-class | Social networks, fraud-ring detection, recommendation engines, knowledge graphs, dependency/impact analysis | Simple tabular reporting/aggregation over large uniform datasets |

**Redis (key-value) example:**
```
SET session:abc123 '{"user_id": 42, "role": "admin"}' EX 3600   -- expires in 1 hour
GET session:abc123
INCR page_views:homepage
ZADD leaderboard 1500 "player1"
ZREVRANGE leaderboard 0 2 WITHSCORES     -- top 3 players
```

**MongoDB (document) example:**
```javascript
// Insert a flexible, nested document — no fixed schema required
db.products.insertOne({
  name: "Wireless Mouse",
  price: 29.99,
  specs: { color: "black", wireless: true },
  tags: ["electronics", "accessories"]
});

// Query with filtering, similar in spirit to WHERE
db.products.find({ price: { $lt: 50 }, tags: "electronics" });

// Aggregation pipeline, similar in spirit to GROUP BY
db.orders.aggregate([
  { $match: { status: "completed" } },
  { $group: { _id: "$customer_id", total: { $sum: "$amount" } } },
  { $sort: { total: -1 } }
]);
```

**Cassandra (columnar/wide-column) example:**
```sql
-- CQL looks SQL-like but the underlying model is a distributed, partitioned wide-column store
CREATE TABLE sensor_readings (
  sensor_id UUID,
  reading_time TIMESTAMP,
  temperature DOUBLE,
  PRIMARY KEY (sensor_id, reading_time)   -- sensor_id = partition key, reading_time = clustering key
) WITH CLUSTERING ORDER BY (reading_time DESC);

SELECT * FROM sensor_readings WHERE sensor_id = ? LIMIT 100;
```

**Neo4j (graph) example:**
```cypher
// Model: (Person)-[:FOLLOWS]->(Person), (Person)-[:PURCHASED]->(Product)
CREATE (alice:Person {name: "Alice"})-[:FOLLOWS]->(bob:Person {name: "Bob"});

// Find friends-of-friends who purchased a specific product (2-hop traversal)
MATCH (me:Person {name: "Alice"})-[:FOLLOWS]->(:Person)-[:FOLLOWS]->(fof:Person)
MATCH (fof)-[:PURCHASED]->(p:Product {name: "Wireless Mouse"})
RETURN DISTINCT fof.name;
```

**When to choose relational vs. NoSQL — decision table:**

| Requirement | Prefer |
|---|---|
| Complex multi-table joins, strong consistency, ad hoc reporting | Relational (Postgres/MySQL) |
| Sub-millisecond cache lookups, ephemeral state | Key-value (Redis) |
| Rapidly evolving/semi-structured schema, document-centric app data | Document (MongoDB) |
| Massive write throughput, horizontal scale across data centers, time-series | Columnar (Cassandra/HBase) |
| Deep, multi-hop relationship traversal (friends-of-friends, fraud rings) | Graph (Neo4j) |
| Large-scale historical analytics, BI dashboards | Cloud data warehouse (Snowflake/BigQuery/Redshift) |

**Pitfalls**
- MongoDB's flexible schema is a double-edged sword — without discipline (schema validation rules, application-level conventions), collections drift into inconsistent shapes over time, making downstream analytics painful.
- Cassandra's data modeling is **query-first**: you design tables around the specific queries you'll run (denormalizing heavily, duplicating data across tables), the opposite instinct from relational normalization — modeling it "the relational way" leads to poor performance.
- Graph databases shine for *traversal-heavy* queries but are usually a poor fit for simple aggregate reporting that a columnar warehouse would handle far more efficiently.
- Redis is primarily in-memory — without persistence configuration (RDB snapshots/AOF logs) or if used beyond available memory, data loss or eviction can occur; it is not a system-of-record replacement for a durable primary database in most designs.

---

### 4.2 CAP Theorem and Eventual Consistency

**Concept.** In a distributed data system, when a network partition occurs, you must choose between:

| Guarantee | Meaning |
|---|---|
| **C**onsistency | Every read receives the most recent write or an error (all nodes see the same data at the same time) |
| **A**vailability | Every request receives a (non-error) response, even without the guarantee that it contains the latest write |
| **P**artition tolerance | The system continues operating despite network partitions (dropped/delayed messages between nodes) |

**CAP theorem statement:** during an actual network partition, a distributed system can only provide **either** Consistency **or** Availability, not both — Partition tolerance is essentially mandatory for any real distributed system (networks *will* partition), so the meaningful tradeoff in practice is **CP vs. AP**.

| | CP Systems (favor Consistency) | AP Systems (favor Availability) |
|---|---|---|
| Behavior during partition | Refuse/delay requests on the minority side to avoid returning stale data | Keep serving requests on all sides, possibly returning stale data |
| Examples | HBase, MongoDB (in certain configurations), traditional RDBMS replication setups configured for strict consistency, ZooKeeper, etcd | Cassandra (tunable), DynamoDB (tunable), Riak |
| Tradeoff | Some requests fail/timeout during partition | Data can be temporarily inconsistent across replicas |

**Eventual consistency.** A weaker consistency model common in AP systems: after writes stop arriving, all replicas will *eventually* converge to the same value, but there is no guarantee about *how long* convergence takes, and reads during the convergence window can return stale or even conflicting data. Techniques used to reconcile: **last-write-wins** (timestamp-based), **vector clocks**, **CRDTs** (Conflict-free Replicated Data Types), and **read-repair**/**anti-entropy** background processes.

**Practical illustration, in words:** A user updates their profile picture. The write lands on Node A and begins replicating to Nodes B and C. A friend viewing the profile from a session routed to Node B a few milliseconds later might briefly still see the old picture (a "stale read") until replication catches up — this is eventual consistency in action, and is an accepted tradeoff in AP systems like Cassandra or DynamoDB in exchange for always being able to accept the write/serve a read even if some nodes are unreachable.

**Pitfalls**
- CAP theorem is about behavior **specifically during a network partition** — it says nothing about normal, non-partitioned operation, where most systems can offer both good consistency and availability. This nuance ("CAP only bites during a partition") is a strong signal of understanding in interviews.
- "NoSQL = AP" and "SQL = CP" is an oversimplification — many modern systems are *tunable* (e.g., Cassandra's per-query consistency levels `ONE`, `QUORUM`, `ALL`; DynamoDB's eventually-consistent vs. strongly-consistent reads) rather than fixed to one side of the tradeoff.
- Confusing CAP's "Consistency" with ACID's "Consistency" is a common mistake — CAP consistency means *linearizability/single-copy consistency across replicas*, whereas ACID consistency means *the database always satisfies its integrity constraints*. They are different concepts that happen to share a name.

---

### 4.3 Cloud Data Warehouses (Snowflake, BigQuery, Redshift)

**Concept.** Cloud data warehouses are purpose-built for OLAP: large-scale aggregation and scanning over historical data, using a **columnar storage** layout and **massively parallel processing (MPP)** or serverless compute, decoupled (in modern designs) from storage.

**Why columnar storage is fast for analytics, in words:** A row-oriented store (like typical OLTP Postgres/MySQL tables) stores an entire row contiguously on disk — great for fetching *all columns of one row* (a single order), but wasteful for `SUM(revenue)` across a billion rows, since the engine must still read every other column of every row into memory even though only `revenue` is needed. A column-oriented store instead stores each column's values contiguously, so an aggregate query touching only 2 of 50 columns reads only those 2 columns' data off disk/storage — dramatically less I/O. Columnar layouts also compress far better (a column of dates or a low-cardinality `region` column compresses extremely well when values are stored together), and enable **vectorized execution** (operating on batches of column values via SIMD-style CPU instructions) and **late materialization** (only reconstructing full rows for the final rows returned).

**Architecture basics:**

| System | Storage/Compute Model | Notes |
|---|---|---|
| **Snowflake** | Fully decoupled storage (cloud object storage) and compute ("virtual warehouses" you can scale/pause independently); multi-cluster warehouses for concurrency | Storage billed separately from compute; can spin up multiple independent compute clusters against the same data with no contention |
| **BigQuery** | Serverless — no clusters to manage; storage in Colossus (Google's distributed FS), compute via Dremel's tree-based execution engine, columnar format (Capacitor) | Pay per query (bytes scanned) or flat-rate slots; automatic scaling; no indexes — relies on columnar pruning + partitioning/clustering |
| **Redshift** | Cluster-based MPP; leader node distributes work to compute nodes; data distributed across nodes via distribution keys; columnar storage (with `ENCODE` compression) | More manual tuning (distribution keys, sort keys, `VACUUM`/`ANALYZE`) compared to Snowflake/BigQuery's more automated approach |

```sql
-- Snowflake: scaling compute independently of storage
CREATE WAREHOUSE analytics_wh WITH WAREHOUSE_SIZE = 'LARGE' AUTO_SUSPEND = 60;
USE WAREHOUSE analytics_wh;
SELECT region, SUM(revenue) FROM fact_sales GROUP BY region;

-- BigQuery: partitioning and clustering to minimize bytes scanned
CREATE TABLE analytics.fact_sales
PARTITION BY DATE(order_date)
CLUSTER BY customer_id
AS SELECT * FROM staging.raw_sales;

-- Querying a partitioned table — only relevant partitions are scanned
SELECT customer_id, SUM(revenue)
FROM analytics.fact_sales
WHERE order_date BETWEEN '2026-01-01' AND '2026-01-31'
GROUP BY customer_id;

-- Redshift: distribution and sort keys control data layout across nodes
CREATE TABLE fact_sales (
  sale_id BIGINT,
  customer_id INT,
  order_date DATE,
  revenue DECIMAL(12,2)
)
DISTKEY (customer_id)      -- co-locates rows with the same customer_id on one node
SORTKEY (order_date);      -- enables efficient range pruning on date filters
```

**Why they're fast for analytics — summary of techniques:**

| Technique | Effect |
|---|---|
| Columnar storage | Read only the columns a query needs |
| Compression (run-length, dictionary encoding) | Less I/O, more data fits in memory/cache |
| Partitioning (e.g., by date) | Skip entire partitions that can't match the filter ("partition pruning") |
| Clustering/sort keys | Skip blocks/micro-partitions outside the filtered range |
| Massively parallel processing | Split scan/aggregation work across many nodes/workers concurrently |
| Vectorized execution | Process batches of values per CPU instruction instead of row-by-row interpretation |
| Decoupled storage/compute (Snowflake/BigQuery) | Scale reads and writes independently; multiple workloads don't contend for the same compute |
| Result/query caching | Repeated identical queries return instantly from cache |

**Pitfalls**
- Row-by-row `INSERT`/`UPDATE`/`DELETE` (OLTP-style single-row DML) is slow and expensive on most columnar warehouses — they're optimized for large batch loads/appends, not fine-grained transactional writes; frequent single-row updates should go through a staging + merge/upsert batch pattern instead.
- BigQuery/Snowflake billing is often usage-based (bytes scanned / compute-seconds) — an unpartitioned table or a `SELECT *` on a huge table can be surprisingly expensive; always filter on partition/cluster columns and select only needed columns.
- Redshift requires more manual maintenance (choosing good `DISTKEY`/`SORTKEY`, periodic `VACUUM`/`ANALYZE`) than Snowflake/BigQuery's more automated approaches — a schema design mistake (bad distribution key causing data skew) can degrade performance significantly and is harder to fix after the fact at scale.
- These warehouses are not a drop-in replacement for an OLTP database — high-concurrency, low-latency single-row transactional workloads (e.g., a checkout flow) still belong on a traditional RDBMS; warehouses feed from and complement OLTP systems via ETL/ELT, they don't replace them.

---

### 4.4 Interview Questions — NoSQL and Modern Data Stores

**Q1. When would you choose MongoDB over a relational database for a new application?**
A: When the schema is expected to evolve rapidly or vary per record (e.g., a product catalog with wildly different attributes per category), when data is naturally document-shaped (nested/hierarchical) and typically accessed as a whole document rather than joined across many normalized tables, or when horizontal scalability of writes is a priority over complex multi-table joins and strict multi-row ACID transactions.

**Q2. Explain the CAP theorem and why "pick any two" is a common misconception.**
A: CAP says that during an actual network partition, a distributed system must choose between Consistency and Availability — Partition tolerance isn't really an optional third choice you can trade away, since real networks partition regardless of what you want. "Pick any two" wrongly implies you could choose CA and ignore partitions, but you cannot design away network partitions in a distributed system — the honest framing is "CP vs. AP when a partition happens," with full CA achievable only in the absence of partitions (e.g., a single-node, non-distributed system).

**Q3. What is eventual consistency, and give a real-world example of an acceptable use case for it.**
A: Eventual consistency guarantees that, absent new writes, all replicas of a piece of data will eventually converge to the same value, though reads during the convergence window may be stale. Example: a "like count" on a social media post — briefly showing a slightly stale count across different users' feeds is an acceptable tradeoff for high availability and low latency, unlike, say, a bank account balance, which needs stronger consistency.

**Q4. How does Cassandra's data modeling approach differ fundamentally from relational normalization?**
A: Relational modeling starts from the data's structure and normalizes to avoid redundancy, then writes ad hoc queries with joins as needed. Cassandra modeling is "query-first" — you enumerate the specific queries your application needs to run, then design one denormalized table per query pattern (deliberately duplicating data across tables), because Cassandra has no efficient ad hoc joins and performs best when each query hits a single partition.

**Q5. What's the difference between a partition key and a clustering/sort key in a wide-column store like Cassandra?**
A: The partition key determines which node(s) store the data (all rows sharing a partition key live together on the same replica set) and is used for equality lookups. The clustering key determines the sort order of rows *within* a partition, enabling efficient range scans within that partition (e.g., `sensor_id` as partition key, `reading_time` as clustering key lets you efficiently fetch a time range of readings for one sensor).

**Q6. Why is columnar storage faster than row storage for analytical aggregate queries?**
A: An aggregate query (e.g., `SUM(revenue) GROUP BY region`) only needs a subset of columns. Row-oriented storage must still read entire rows off disk even for unused columns, wasting I/O. Column-oriented storage stores each column's values contiguously, so the engine reads only the needed columns, and same-typed contiguous values also compress much better and enable vectorized (batch) processing — both further reducing I/O and CPU cost.

**Q7. What is partition pruning in a data warehouse, and how does it improve performance/cost?**
A: If a table is partitioned by a column (commonly a date), and a query filters on that column, the engine can skip scanning entire partitions that fall outside the filter range entirely, rather than scanning the whole table. This reduces both query latency and, in usage-based-billing warehouses like BigQuery, the number of bytes scanned (and therefore cost).

**Q8. Compare Redis, MongoDB, Cassandra, and Neo4j in one sentence each, focused on their primary strength.**
A: Redis — in-memory key-value store optimized for extremely low-latency caching, counters, and ephemeral state. MongoDB — document store optimized for flexible, semi-structured, nested application data accessed as whole documents. Cassandra — wide-column store optimized for extremely high write throughput and linear horizontal scalability across data centers. Neo4j — graph database optimized for deep, multi-hop relationship traversal queries that would require many expensive joins in a relational model.

**Q9. Why can't you efficiently run ad hoc `SELECT *` analytical queries against a normalized OLTP schema at scale, and how does a star-schema warehouse solve it?**
A: A normalized OLTP schema spreads related data across many small tables, so any broad analytical question requires many joins across potentially huge tables — expensive and slow, and row-oriented storage compounds this by reading unused columns too. A star-schema warehouse denormalizes dimensions around a central fact table (fewer joins) and stores data column-oriented (only needed columns read), specifically optimized for the scan-heavy, aggregate-heavy access pattern analytical queries need.

**Q10. What does it mean that Snowflake and BigQuery "decouple storage and compute," and why does that matter?**
A: Storage (the actual data, in cloud object storage) and compute (the clusters/warehouses that process queries) scale and bill independently. This means you can spin up multiple, differently-sized compute clusters against the same underlying data without them contending for resources (e.g., one warehouse for ETL jobs, another for BI dashboards, running concurrently without slowing each other down), and you can scale compute up/down (or pause it entirely) without moving or duplicating the underlying data.

**Q11. In DynamoDB/Cassandra terms, what does a "tunable consistency" read mean, and why would an engineer choose a weaker consistency level?**
A: Tunable consistency lets you specify, per query, how many replicas must acknowledge a read/write before it's considered successful (e.g., Cassandra's `ONE`, `QUORUM`, `ALL`). Choosing a weaker level (e.g., reading from just one replica) trades a small risk of reading stale data for lower latency and higher availability (the read can succeed even if some replicas are down or slow) — appropriate when the application can tolerate brief staleness (e.g., a view counter) in exchange for speed and resilience.

**Q12. A team stores chat/session history for an LLM-based AI agent. Would you recommend a relational database, a document store, or both? Justify your answer.**
A: Likely both, playing to each one's strength: a document store (MongoDB) or key-value store (Redis/DynamoDB) is a natural fit for storing variable-shaped conversation turns/messages (roles, tool calls, arbitrary metadata) with fast read/write per session, while a relational database remains a good fit for strongly-structured, relationally-critical data such as user accounts, billing, and audit records that need transactional guarantees and joins. Many production AI systems also add a vector store/index alongside these for embedding-based retrieval, which is a distinct concern from either.

---

## 5. Rapid-Fire Interview Q&A

| # | Question | Answer |
|---|---|---|
| 1 | What does `NULL = NULL` evaluate to in SQL? | `UNKNOWN` (not `TRUE`) — use `IS NULL` to test for NULL |
| 2 | Which runs first: `WHERE` or `GROUP BY`? | `WHERE` (filters rows before grouping) |
| 3 | Which SQL clause can use aggregate functions to filter: `WHERE` or `HAVING`? | `HAVING` |
| 4 | What does `LEFT JOIN` return that `INNER JOIN` doesn't? | All rows from the left table, even those with no match on the right (NULLs filled in) |
| 5 | Fastest way to remove duplicate rows while combining two queries without dedup overhead? | `UNION ALL` |
| 6 | What SQL keyword tests for existence of at least one matching row? | `EXISTS` |
| 7 | Name the four ACID properties. | Atomicity, Consistency, Isolation, Durability |
| 8 | Which isolation level prevents dirty reads, non-repeatable reads, AND phantom reads? | `SERIALIZABLE` |
| 9 | What SQL clause assigns a unique row number within a group? | `ROW_NUMBER() OVER (PARTITION BY ... ORDER BY ...)` |
| 10 | Difference between `RANK()` and `DENSE_RANK()` on ties? | `RANK()` leaves gaps after ties; `DENSE_RANK()` doesn't |
| 11 | What does a covering index avoid? | A second lookup into the base table (enables index-only scan) |
| 12 | What is a "SARGable" predicate? | One the engine can evaluate via an index seek/range scan without transforming column values first |
| 13 | Why is `WHERE YEAR(order_date) = 2026` slow? | It wraps the indexed column in a function, forcing a full scan (non-SARGable) |
| 14 | What normal form eliminates transitive dependencies? | 3NF |
| 15 | What normal form requires every determinant to be a candidate key? | BCNF |
| 16 | OLTP or OLAP: which is typically normalized? | OLTP |
| 17 | OLTP or OLAP: which typically uses star schema? | OLAP |
| 18 | In a star schema, what does the fact table store? | Foreign keys to dimensions + numeric measures |
| 19 | Which SCD type overwrites history with no trace? | Type 1 |
| 20 | Which SCD type preserves full history via new versioned rows? | Type 2 |
| 21 | What's the CAP theorem's core tradeoff during a network partition? | Consistency vs. Availability (Partition tolerance is essentially mandatory) |
| 22 | Key-value store best known for in-memory caching? | Redis |
| 23 | Document store best known for flexible JSON-like schemas? | MongoDB |
| 24 | Wide-column store built for massive write throughput? | Cassandra |
| 25 | Graph database best known for relationship traversal? | Neo4j |
| 26 | Why is columnar storage fast for analytics? | Reads only needed columns, compresses better, enables vectorized execution |
| 27 | What does `EXPLAIN ANALYZE` add over plain `EXPLAIN`? | Actual execution timing and row counts, not just estimates |
| 28 | What is the N+1 query problem? | Issuing one extra query per row instead of one batched/joined query |
| 29 | What causes a deadlock? | Two transactions each holding a lock the other needs (circular wait) |
| 30 | How do databases resolve a detected deadlock? | Abort ("kill") one transaction as the victim, letting the other proceed |
| 31 | Does a `FOREIGN KEY` constraint automatically index the child column in every engine? | No (e.g., not automatic in Postgres) — index it explicitly |
| 32 | Can a `UNIQUE` column contain multiple NULLs? | Yes, in most engines (NULL isn't equal to NULL) |
| 33 | What join type produces a Cartesian product? | `CROSS JOIN` |
| 34 | What's the danger of `NOT IN` with a subquery? | Returns zero rows if the subquery result contains any NULL |
| 35 | What's the difference between `RANGE` and `ROWS` frame types in window functions? | `ROWS` counts physical rows; `RANGE` groups by logical value ties on the `ORDER BY` key |
| 36 | What must a recursive CTE's recursive term use instead of `UNION`? | `UNION ALL` |
| 37 | What's a common cause of a query planner choosing a bad plan? | Stale table statistics |
| 38 | Why do indexes slow down writes? | Every `INSERT`/`UPDATE`/`DELETE` must also update each affected index |
| 39 | What is index selectivity? | Ratio of distinct values to total rows; higher selectivity benefits more from indexing |
| 40 | What does "partition pruning" do in a data warehouse? | Skips scanning partitions that can't match the query's filter, cutting cost/latency |
| 41 | What does `GROUP BY ROLLUP(a, b)` add that plain `GROUP BY a, b` doesn't? | Subtotal rows per `a` plus one grand-total row |
| 42 | Which GROUP BY extension produces every possible subtotal combination, not just a hierarchy? | `CUBE` |
| 43 | What does `GROUPING(col)` return on a subtotal row vs. a real value? | `1` on a subtotal row, `0` on a real value |
| 44 | Does a regular `VIEW` store data on disk? | No — it's a stored query, recomputed each reference |
| 45 | Does a `MATERIALIZED VIEW` store data on disk? | Yes — the result set is physically stored and must be refreshed |
| 46 | What must exist on a Postgres materialized view before you can `REFRESH ... CONCURRENTLY`? | A unique index on the materialized view |
| 47 | What database object runs automatically in response to an `INSERT`/`UPDATE`/`DELETE`? | A trigger |
| 48 | In the "gaps and islands" problem, what is an "island"? | A maximal run of consecutive/same-status values that should be grouped as one |
| 49 | What SQL window function is the standard tool for a generalized "Nth-highest value" query with ties treated as one rank? | `DENSE_RANK()` |
| 50 | What anomaly occurs when two transactions read-then-write the same row and one write silently overwrites the other? | Lost update |

---

*End of syllabus. Recommended next step: practice writing every query above from memory against a local Postgres/MySQL instance, then time yourself running `EXPLAIN ANALYZE` on each to build real intuition for planner behavior.*
