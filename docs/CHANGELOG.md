# V2.6

Package version: `2.6.1`.

## 2.6.1 - 19 October 2026

- Select the transport tamper-check image from worker 0's client 0 manifest partition, making the fixture independent of filesystem enumeration order.

## 2.6.0 - 18 October 2026

- Case-level audit records link source observations, descriptors, rendered variants and client-round selections. Training records actual presentations, supervised presentations and attempted/applied/AMP-skipped step IDs per case.
- Verification exports the fixed fidelity mask, intervals, measurements, weights, normalizers, active mass and first failed gate, including early exclusions.
- Evaluation writes source-bound identity/protocol manifests for both local and worker execution.
- Cyclic verification follows the manuscript's complete candidate-tuple shift by 11 positions. Frozen-variant analysis preserves geometry and expert predictions and recalculates class severity from each route posterior.
- Offline tools calculate independent-review reliability, explicit-condition routing and paired-seed differences, and export case audits, actual-result figures and fixed-identity qualitative panels.
- [Manuscript alignment](MANUSCRIPT_ALIGNMENT.md) records the revised source fingerprints and method/analysis contracts.
- Worker task results and round cursors persist in a transactional SQLite cache. Completed tasks can be retried after a worker restart without repeating training.
- The cache binds code, configuration, data and keys. HMAC checks cover results, round cursors and record counts; cache payloads use numeric arrays and JSON.
- Batch experiments support `--resume`, validate completed artifacts, continue round snapshots and retain attempt history. Factorial metric files are rebuilt from committed jobs.
- Batch input binding covers every variant's active rendering inputs and reuses scans for identical input scopes across seeds.
- Training and workers check manifest provenance, model class counts and label ranges before constructing a detector.
- `python -m tefed.main --version` reports the package version.
- Git and Docker exclude worker cache databases.

[Commands](WORKFLOWS.md) · [Executed checks](TEST_REPORT.md)
