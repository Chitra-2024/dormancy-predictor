WITH params AS (
    SELECT CAST(? AS DATE) AS snap
),

-- FEATURE INPUTS: only rows strictly BEFORE the snapshot
trades_w AS (
    SELECT tr.client_id, CAST(tr.trade_date AS DATE) AS d,
           tr.value, tr.product, p.snap
    FROM read_parquet('{silver_dir}/trades.parquet') tr
    CROSS JOIN params p
    WHERE CAST(tr.trade_date AS DATE) >= p.snap - INTERVAL 90 DAY
      AND CAST(tr.trade_date AS DATE) <  p.snap
),
logins_w AS (
    SELECT lg.client_id, CAST(lg.login_date AS DATE) AS d, p.snap
    FROM read_parquet('{silver_dir}/logins.parquet') lg
    CROSS JOIN params p
    WHERE CAST(lg.login_date AS DATE) >= p.snap - INTERVAL 90 DAY
      AND CAST(lg.login_date AS DATE) <  p.snap
),

t AS (
    SELECT client_id,
           COUNT(*) FILTER (WHERE d >= snap - INTERVAL 30 DAY) AS trades_30d,
           COUNT(*) FILTER (WHERE d >= snap - INTERVAL 60 DAY) AS trades_60d,
           COUNT(*)                                            AS trades_90d,
           AVG(value)                                          AS avg_trade_value_90d,
           COUNT(DISTINCT product)                             AS distinct_products_90d,
           MAX(d)                                              AS last_trade
    FROM trades_w
    GROUP BY client_id, snap
),
l AS (
    SELECT client_id,
           COUNT(*) FILTER (WHERE d >= snap - INTERVAL 30 DAY) AS logins_30d,
           COUNT(*) FILTER (WHERE d >= snap - INTERVAL 60 DAY) AS logins_60d,
           MAX(d)                                              AS last_login
    FROM logins_w
    GROUP BY client_id, snap
),

-- LABEL: the ONLY block allowed to look at data on or after the snapshot
y AS (
    SELECT tr.client_id, COUNT(*) AS future_trades
    FROM read_parquet('{silver_dir}/trades.parquet') tr
    CROSS JOIN params p
    WHERE CAST(tr.trade_date AS DATE) >= p.snap
      AND CAST(tr.trade_date AS DATE) <  p.snap + INTERVAL 90 DAY
    GROUP BY tr.client_id
)

SELECT c.client_id,
       p.snap AS snapshot_date,
       c.segment,
       t.trades_30d,
       t.trades_60d,
       t.trades_90d,
       t.trades_30d - (t.trades_60d - t.trades_30d)          AS trade_trend,
       DATE_DIFF('day', t.last_trade, p.snap)                AS days_since_last_trade,
       t.avg_trade_value_90d,
       t.distinct_products_90d,
       COALESCE(l.logins_30d, 0)                             AS logins_30d,
       COALESCE(l.logins_30d, 0)
         - (COALESCE(l.logins_60d, 0) - COALESCE(l.logins_30d, 0)) AS login_trend,
       COALESCE(DATE_DIFF('day', l.last_login, p.snap), 90)  AS days_since_last_login,
       DATE_DIFF('day', CAST(c.signup_date AS DATE), p.snap) AS tenure_days,
       CASE WHEN COALESCE(y.future_trades, 0) = 0 THEN 1 ELSE 0 END
                                                             AS dormant_next_90d
FROM read_parquet('{silver_dir}/clients.parquet') c
CROSS JOIN params p
JOIN t      ON c.client_id = t.client_id
LEFT JOIN l ON c.client_id = l.client_id
LEFT JOIN y ON c.client_id = y.client_id;