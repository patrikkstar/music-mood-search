
CREATE TABLE IF NOT EXISTS inference_tasks (
    id UUID PRIMARY KEY,

    status TEXT NOT NULL DEFAULT 'PENDING'
        CHECK (
            status IN (
                'PENDING',
                'PROCESSING',
                'COMPLETED',
                'FAILED'
            )
        ),

    result JSONB,
    model_version TEXT,
    inference_ms DOUBLE PRECISION,
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ,

    CONSTRAINT completed_requires_result
        CHECK (
            status <> 'COMPLETED'
            OR result IS NOT NULL
        )
);
