-- Curated research views. Promote an ad hoc query here once it has been reused
-- several times (see docs/decision-log.md).

-- Latest status row per run.
CREATE VIEW runs AS
SELECT * EXCLUDE (rn) FROM (
    SELECT *, row_number() OVER (PARTITION BY run_id ORDER BY updated_at DESC) AS rn
    FROM runs_history
) WHERE rn = 1;

-- Latest grade per (episode, grader); earlier grades are kept for audit.
CREATE VIEW grades AS
SELECT * EXCLUDE (rn) FROM (
    SELECT *, row_number() OVER (PARTITION BY episode_id, grader ORDER BY graded_at DESC) AS rn
    FROM grades_history
) WHERE rn = 1;

-- Accuracy, cost, and tool usage per run, architecture, and grader.
CREATE VIEW run_accuracy AS
SELECT
    e.run_id,
    e.architecture,
    g.grader,
    count(*)                         AS n_episodes,
    avg(g.score)                     AS mean_score,
    avg(e.n_turns)                   AS mean_turns,
    avg(e.tokens_in + e.tokens_out)  AS mean_tokens,
    avg(e.latency_ms)                AS mean_latency_ms
FROM episodes e
JOIN grades g USING (episode_id)
GROUP BY ALL;

-- Distribution of labeled failure modes per run and architecture.
CREATE VIEW failure_modes AS
SELECT e.run_id, e.architecture, g.failure_mode, count(*) AS n
FROM episodes e
JOIN grades g USING (episode_id)
WHERE g.failure_mode IS NOT NULL
GROUP BY ALL;

-- Summary of each metric per run.
CREATE VIEW metric_summary AS
SELECT
    run_id,
    name,
    count(*)                 AS n_points,
    min(step)                AS first_step,
    max(step)                AS last_step,
    arg_max(value, step)     AS last_value,
    min(value)               AS min_value,
    max(value)               AS max_value
FROM metrics
GROUP BY ALL;
