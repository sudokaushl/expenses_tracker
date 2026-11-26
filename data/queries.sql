-- ============================================
-- EXPENSES TRACKER DASHBOARD - SQL QUERIES (DuckDB)
-- Table: expenses.acc_labelled
-- Columns: trans_id, trans_dt, particulars, normalized_trans_src, trans_amt, trans_type,
--          trans_mode, category, subcategory, category_conf, subcategory_conf,
--          label_source, manual_review, comments
-- Spending queries use trans_type = 'debit'.
-- ============================================

-- ============================================
-- 1. OVERVIEW PANEL
-- ============================================

-- Total transactions count
SELECT COUNT(*) AS total_transactions FROM acc_labelled;

-- Total spending amount
SELECT
    SUM(trans_amt) AS total_spending,
    AVG(trans_amt) AS avg_spending,
    MIN(trans_amt) AS min_spending,
    MAX(trans_amt) AS max_spending
FROM acc_labelled
WHERE trans_type = 'debit' AND trans_amt IS NOT NULL;

-- Total credits vs debits
SELECT
    trans_type,
    COUNT(*) AS transaction_count,
    ROUND(SUM(trans_amt), 2) AS total_amount
FROM acc_labelled
GROUP BY trans_type;

-- Date range
SELECT
    MIN(trans_dt) AS earliest_transaction,
    MAX(trans_dt) AS latest_transaction
FROM acc_labelled;

-- Daily spending trend (last 30 days)
SELECT
    trans_dt AS date,
    SUM(trans_amt) AS daily_spending,
    COUNT(*) AS daily_count
FROM acc_labelled
WHERE trans_type = 'debit'
  AND trans_dt >= current_date - INTERVAL 30 DAY
GROUP BY trans_dt
ORDER BY date;

-- Monthly spending summary
SELECT
    DATE_TRUNC('month', trans_dt) AS month,
    SUM(trans_amt) AS monthly_spending,
    COUNT(*) AS monthly_count
FROM acc_labelled
WHERE trans_type = 'debit'
GROUP BY DATE_TRUNC('month', trans_dt)
ORDER BY month DESC
LIMIT 12;

-- ============================================
-- 2. PAYMENT MODE DISTRIBUTION
-- ============================================

-- Payment mode counts and amounts
SELECT
    trans_mode,
    COUNT(*) AS transaction_count,
    ROUND(SUM(trans_amt), 2) AS total_amount,
    ROUND(AVG(trans_amt), 2) AS avg_amount
FROM acc_labelled
WHERE trans_mode IS NOT NULL
GROUP BY trans_mode
ORDER BY total_amount DESC;

-- Payment mode percentages
SELECT
    trans_mode,
    COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() AS percentage,
    ROUND(SUM(trans_amt), 2) AS total_amount
FROM acc_labelled
WHERE trans_mode IS NOT NULL
GROUP BY trans_mode;

-- ============================================
-- 3. CATEGORY/SUBCATEGORY BREAKDOWN
-- ============================================

-- Top categories by spending
SELECT
    category,
    COUNT(*) AS transaction_count,
    ROUND(SUM(trans_amt), 2) AS total_spending,
    ROUND(AVG(trans_amt), 2) AS avg_spending,
    ROUND(SUM(trans_amt) * 100.0 / COALESCE(SUM(SUM(trans_amt)) OVER(), 0), 2) AS percentage
FROM acc_labelled
WHERE trans_type = 'debit' AND category IS NOT NULL AND trans_amt IS NOT NULL
GROUP BY category
ORDER BY total_spending DESC;

-- Subcategories within top categories
SELECT
    category,
    subcategory,
    COUNT(*) AS transaction_count,
    ROUND(SUM(trans_amt), 2) AS total_spending
FROM acc_labelled
WHERE trans_type = 'debit' AND category IS NOT NULL AND subcategory IS NOT NULL AND trans_amt IS NOT NULL
GROUP BY category, subcategory
ORDER BY category, total_spending DESC;

-- Category spending with percentages (for pie chart)
SELECT
    category,
    ROUND(SUM(trans_amt), 2) AS amount,
    ROUND(SUM(trans_amt) * 100.0 / COALESCE(SUM(SUM(trans_amt)) OVER(), 0), 2) AS percentage
FROM acc_labelled
WHERE trans_type = 'debit' AND category IS NOT NULL AND trans_amt IS NOT NULL
GROUP BY category
ORDER BY amount DESC;

-- ============================================
-- 4. ENTITY ANALYSIS
-- ============================================

-- Top vendors/merchants by spending
SELECT
    normalized_trans_src,
    COUNT(*) AS transaction_count,
    ROUND(SUM(trans_amt), 2) AS total_spending,
    trans_mode
FROM acc_labelled
WHERE trans_type = 'debit' AND normalized_trans_src IS NOT NULL AND trans_amt IS NOT NULL
GROUP BY normalized_trans_src, trans_mode
ORDER BY total_spending DESC
LIMIT 50;

-- Entity frequency (top 20)
SELECT
    normalized_trans_src,
    COUNT(*) AS frequency
FROM acc_labelled
WHERE normalized_trans_src IS NOT NULL
GROUP BY normalized_trans_src
ORDER BY frequency DESC
LIMIT 20;

-- ============================================
-- 5. TRANSACTION AMOUNT ANALYSIS
-- ============================================

-- Amount distribution histogram (binned)
SELECT
    CASE
        WHEN trans_amt < 100 THEN '< 100'
        WHEN trans_amt < 500 THEN '100-500'
        WHEN trans_amt < 1000 THEN '500-1000'
        WHEN trans_amt < 5000 THEN '1000-5000'
        ELSE '> 5000'
    END AS amount_range,
    COUNT(*) AS count,
    ROUND(SUM(trans_amt), 2) AS total_amount
FROM acc_labelled
WHERE trans_type = 'debit' AND trans_amt IS NOT NULL
GROUP BY 1
ORDER BY MIN(trans_amt);

-- Daily spending trend (line chart)
SELECT
    trans_dt AS date,
    SUM(trans_amt) AS daily_spending,
    COUNT(*) AS transaction_count
FROM acc_labelled
WHERE trans_type = 'debit'
  AND trans_dt >= current_date - INTERVAL 60 DAY
GROUP BY trans_dt
ORDER BY date;

-- Spending by day of week
SELECT
    DAYOFWEEK(trans_dt) AS day_of_week,
    DAYNAME(trans_dt) AS day_name,
    COUNT(*) AS count,
    ROUND(SUM(trans_amt), 2) AS total_spending
FROM acc_labelled
WHERE trans_type = 'debit'
GROUP BY 1, 2
ORDER BY day_of_week;

-- Spending by month (for time series)
SELECT
    DATE_TRUNC('month', trans_dt) AS month,
    SUM(trans_amt) AS monthly_spending,
    COUNT(*) AS count
FROM acc_labelled
WHERE trans_type = 'debit'
  AND trans_dt >= current_date - INTERVAL 12 MONTH
GROUP BY 1
ORDER BY month;

-- ============================================
-- 6. DATA QUALITY METRICS
-- ============================================

-- Null values in key columns
SELECT 'category' AS column_name,
       COUNT(*) FILTER (WHERE category IS NULL) AS null_count,
       ROUND(COUNT(*) FILTER (WHERE category IS NULL) * 100.0 / NULLIF(COUNT(*), 0), 2) AS null_percentage
FROM acc_labelled
UNION ALL
SELECT 'subcategory',
       COUNT(*) FILTER (WHERE subcategory IS NULL),
       ROUND(COUNT(*) FILTER (WHERE subcategory IS NULL) * 100.0 / NULLIF(COUNT(*), 0), 2)
FROM acc_labelled
UNION ALL
SELECT 'trans_amt',
       COUNT(*) FILTER (WHERE trans_amt IS NULL),
       ROUND(COUNT(*) FILTER (WHERE trans_amt IS NULL) * 100.0 / NULLIF(COUNT(*), 0), 2)
FROM acc_labelled
UNION ALL
SELECT 'normalized_trans_src',
       COUNT(*) FILTER (WHERE normalized_trans_src IS NULL),
       ROUND(COUNT(*) FILTER (WHERE normalized_trans_src IS NULL) * 100.0 / NULLIF(COUNT(*), 0), 2)
FROM acc_labelled
UNION ALL
SELECT 'trans_mode',
       COUNT(*) FILTER (WHERE trans_mode IS NULL),
       ROUND(COUNT(*) FILTER (WHERE trans_mode IS NULL) * 100.0 / NULLIF(COUNT(*), 0), 2)
FROM acc_labelled;

-- Particulars matching vs differing from normalized source
SELECT
    CASE WHEN particulars = normalized_trans_src THEN 'Matching' ELSE 'Differing' END AS particular_status,
    COUNT(*) AS count
FROM acc_labelled
GROUP BY 1;

-- Labeling source and review status
SELECT
    label_source,
    manual_review,
    COUNT(*) AS count
FROM acc_labelled
GROUP BY label_source, manual_review
ORDER BY count DESC;

-- Data completeness score
SELECT
    COUNT(*) AS total_rows,
    COUNT(category) AS rows_with_category,
    COUNT(subcategory) AS rows_with_subcategory,
    COUNT(trans_amt) AS rows_with_amount,
    ROUND((COUNT(category) + COUNT(subcategory) + COUNT(trans_amt)) * 100.0 / (3.0 * COUNT(*)), 2) AS completeness_score
FROM acc_labelled;

-- ============================================
-- FILTER QUERIES (for sidebar filters)
-- ============================================

-- Filter by date range
SELECT * FROM acc_labelled
WHERE trans_dt BETWEEN :start_date AND :end_date;

-- Filter by payment mode
SELECT * FROM acc_labelled
WHERE trans_mode = :trans_mode;

-- Filter by category
SELECT * FROM acc_labelled
WHERE category = :category;

-- Filter by entity
SELECT * FROM acc_labelled
WHERE normalized_trans_src LIKE :entity_name;

-- Filter by amount range
SELECT * FROM acc_labelled
WHERE trans_amt BETWEEN :min_amount AND :max_amount;

-- Combined filters
SELECT * FROM acc_labelled
WHERE
    trans_dt BETWEEN :start_date AND :end_date
    AND (:trans_mode IS NULL OR trans_mode = :trans_mode)
    AND (:category IS NULL OR category = :category)
    AND (:entity_name IS NULL OR normalized_trans_src LIKE :entity_name)
    AND trans_amt BETWEEN :min_amount AND :max_amount;

-- ============================================
-- AGGREGATE QUERIES FOR PANELS
-- ============================================

-- Unique values for filter dropdowns
SELECT DISTINCT trans_mode FROM acc_labelled WHERE trans_mode IS NOT NULL ORDER BY trans_mode;

SELECT DISTINCT category FROM acc_labelled WHERE category IS NOT NULL ORDER BY category;

SELECT DISTINCT trans_type FROM acc_labelled WHERE trans_type IS NOT NULL ORDER BY trans_type;

SELECT DISTINCT normalized_trans_src FROM acc_labelled WHERE normalized_trans_src IS NOT NULL ORDER BY normalized_trans_src LIMIT 100;

-- Date range for time picker
SELECT
    MIN(trans_dt) AS min_date,
    MAX(trans_dt) AS max_date
FROM acc_labelled;
