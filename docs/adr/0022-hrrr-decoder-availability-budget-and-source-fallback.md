# HRRR decoder-ready availability budget and multi-source fallback

**Status**: accepted

The forward HRRR availability probe recorded 107 complete MAJOR cycles and 537 complete
non-MAJOR cycles from 2026-07-04 through 2026-07-30, excluding left-censored startup
cycles and the interrupted 2026-07-31 tail. A first f01 object appeared after roughly
53--54 minutes, but the complete decoder contract became available after a median 107.5
minutes for MAJOR cycles (f01..f48) and 85.8 minutes for non-MAJOR cycles (native f01..f18
plus the prior-MAJOR dependency); first-object availability must therefore never be used
as a proxy for decoder readiness.

**Decision**: Probe AWS, NOMADS, and Azure in parallel and use the earliest source that
passes the complete object contract. Use 100 minutes for routine non-MAJOR readiness and
120 minutes for routine MAJOR readiness; classify later cycles as late and retain source
fallback/health reporting. For approximately 99% coverage, allow about 125 minutes for
non-MAJOR and 120 minutes for MAJOR cycles. The probe's default statistics exclude
left-censored and incomplete cycles, and first-observed timestamps carry approximately
one poll interval (30 seconds) of timing uncertainty.

The non-MAJOR prior-MAJOR dependency is a correctness gate, not an additional six-hour
wait. In the clean probe window, the prior MAJOR was already ready before the non-MAJOR
terminal fxx for 537/537 AWS rows and 537/537 Azure rows; NOMADS was dependency-gated in
12/537 rows (2.23%). Among those rare NOMADS gates, the median added delay was 11.8
minutes and the P90 was 51.5 minutes. The median slack between dependency readiness and
the non-MAJOR terminal was about 156--158 minutes across sources.

**Considered options**: Scheduling on f01 visibility (~53 minutes) was rejected because
the required decoder objects were not yet complete. AWS-only probing was rejected because
NOMADS won approximately 37% of MAJOR and 43% of non-MAJOR cycles; Azure was retained as
redundant fallback despite rarely winning. `Last-Modified` was rejected as the primary
availability clock because the production contract is mirror visibility plus object
completeness, not origin metadata alone.

**Consequences**: ADR-0006's 65-minute staleness threshold remains a separate
f01/ForecastLoop safety rule and must not be replaced by this full-decoder latency
report. Historical and training downloaders should persist source, init_time, terminal
fxx, first-object time, terminal-object time, decoder-ready time, and censoring status;
late or incomplete cycles must remain visible rather than being silently filled.

Evidence: `data_store/probes/hrrr_availability.sqlite` and
`data_store/probes/hrrr_availability.log`; summary command:
`./.venv/bin/python -m scripts.probe_hrrr_availability summary --db data_store/probes/hrrr_availability.sqlite`.
