"""Vision-planner subsystem: grounder -> planner -> executor pipeline.

Includes Meta-Harness auto-prompt evolution (REMEDY-12):
- experiment_store: SQLite-backed experiment tracking
- proposer: Generate prompt variants when success rate drops
- scorer: Evaluate prompt variants against held-out documents
- evolution: Promote winners, retire losers
"""
