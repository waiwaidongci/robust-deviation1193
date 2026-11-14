-- Upgrade: multi-group mode and historical single-peak compatibility.
-- Fresh installs get the column from SQLAlchemy metadata. Existing databases run
-- the guarded statements below (idempotent) or the app's startup upgrade_schema().
ALTER TABLE deviation_runs ADD COLUMN IF NOT EXISTS mode VARCHAR(16) NOT NULL DEFAULT 'single';
UPDATE deviation_runs SET mode='single' WHERE mode IS NULL OR mode='';
CREATE INDEX IF NOT EXISTS run_study_mode_idx ON deviation_runs (study_id,mode);
