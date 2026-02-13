AlphaTrade v0.1 API Specification
Predictive service contract (inputs/outputs) for multi-horizon return distributions
Version: v0.1
Status: Draft (engineering-ready)
Last updated: 2026-02-13 (Asia/Singapore)
1. Scope
This document defines the externally visible contract for AlphaTrade v0.1: request/response schemas, endpoints, error handling, and versioning. AlphaTrade v0.1 is a pure predictor (no trading logic).
2. Definitions
Field	Definition
Base timeframe	1-minute bars (OHLCV) with timestamps at bar end time
asof_bar_end	The end timestamp of the most recent completed 1m bar used as input
Lookback L	Number of 1m bars in the input window (configurable)
Horizon h	Number of 1m steps into the future for a target (e.g., h=5 means 5 minutes)
Target r^(h)	log(C_{t+h} / C_t), where C_t is the close at asof_bar_end
Quantiles Q	Set of quantile levels for the target distribution (default 0.1,0.25,0.5,0.75,0.9)
3. Versioning and Compatibility
All responses include model_version. Breaking changes increment the major version (v1.x -> v2.x). Non-breaking changes (e.g., additional optional fields) increment the minor version.
v0.1 guarantees: base timeframe=1m; outputs are quantiles of log-returns for configured horizons.
4. Authentication and Transport
Transport: HTTPS (JSON) or gRPC (proto equivalent).
Authentication: API key or mTLS (deployment choice).
Idempotency: Requests are idempotent; server may cache identical inputs for a short TTL.
5. Endpoints
5.1 POST /v0.1/predict
Returns multi-horizon predictive distributions (quantiles) of future log-returns.
Request schema
Field	Definition
instrument_id	string; required. Unique identifier for the futures contract
asof_bar_end	RFC3339 timestamp with timezone; required
lookback_L	int; optional. If omitted, server default is used (recommended).
horizons	array[int]; optional. If omitted, server default is used. Must be positive integers.
quantiles	array[float]; optional. If omitted, server default is used; values in (0,1) strictly increasing.
inputs	object; required. Contains the input features window (see Inputs payload)
request_id	string; optional. For tracing; echoed in response
debug	bool; optional. If true, include extra diagnostics (no training-sensitive content)
Inputs payload (features window)
inputs.features is a 2D array with shape [L, F], ordered from oldest to newest, last row corresponds to asof_bar_end.
v0.1 fixed feature order (F=8):
Field	Definition
0 lr_close	log(C_i / C_{i-1})
1 hl_range	log(H_i - L_i + epsilon)
2 oc_return	log(C_i / O_i)
3 log_vol	log(V_i + 1)
4 pos_in_range	(C_i - L_i) / (H_i - L_i + epsilon)
5 time_sin	sin(2π * minute_index / period) (period depends on trading session encoding)
6 time_cos	cos(2π * minute_index / period)
7 is_session_open	0/1 flag indicating within valid trading session (recommended for markets with breaks)
Note: epsilon is a small constant used only to avoid division by zero; it must match training configuration.
Example request (JSON)
{
  "instrument_id": "IF2403",
  "asof_bar_end": "2026-02-13T10:07:00+08:00",
  "lookback_L": 4096,
  "horizons": [1, 5, 20, 60],
  "quantiles": [0.1, 0.25, 0.5, 0.75, 0.9],
  "request_id": "req-9c3f6e6a",
  "inputs": {
    "features": [
      [0.0003, -1.2039, 0.0001, 5.8231, 0.62, 0.11, 0.99, 1],
      "... L rows total ...",
      [-0.0002, -1.3210, -0.0001, 5.9102, 0.41, 0.08, 1.00, 1]
    ]
  }
}
Response schema
Field	Definition
model_version	string; e.g., alphatrade_v0.1
instrument_id	string; echoed
asof_bar_end	timestamp; echoed
lookback_L	int; effective L used by server
horizons	array[int]; effective horizons used by server
quantiles	array[float]; effective quantiles used by server
pred.log_return_quantiles	map<string, array[float]>; keys are horizons as strings; values are quantiles aligned with quantiles[]
quality	object; optional. Calibration/health indicators when available
request_id	string; echoed if provided
latency_ms	number; server-side inference latency
Example response (JSON)
{
  "model_version": "alphatrade_v0.1",
  "instrument_id": "IF2403",
  "asof_bar_end": "2026-02-13T10:07:00+08:00",
  "lookback_L": 4096,
  "horizons": [1, 5, 20, 60],
  "quantiles": [0.1, 0.25, 0.5, 0.75, 0.9],
  "pred": {
    "log_return_quantiles": {
      "1":  [-0.0008, -0.0003, 0.0001, 0.0004, 0.0009],
      "5":  [-0.0019, -0.0008, 0.0002, 0.0010, 0.0021],
      "20": [-0.0042, -0.0017, 0.0005, 0.0022, 0.0046],
      "60": [-0.0089, -0.0038, 0.0011, 0.0042, 0.0094]
    }
  },
  "latency_ms": 12.7,
  "request_id": "req-9c3f6e6a"
}
5.2 GET /v0.1/health
Returns service health status and loaded model metadata.
6. Errors
Errors are returned with HTTP status codes and a structured JSON body.
Field	Definition
code	string; stable error code
message	string; human readable
request_id	string; if provided
details	object; optional; field-level errors
Common error codes
- INVALID_ARGUMENT: schema, shape, or value errors (e.g., quantiles not sorted)
- OUT_OF_RANGE: lookback_L too small/large for this model build
- UNAVAILABLE: model not loaded or temporary outage
- UNAUTHENTICATED / PERMISSION_DENIED: auth failures
7. Observability and Audit
The service must log: instrument_id, asof_bar_end, effective L/H/Q, request_id, latency, and error codes.
Predictions and realized outcomes should be stored separately for evaluation and drift monitoring.
8. Configuration
Runtime configuration is externalized (config file or environment). Recommended settings:
model:
  version: alphatrade_v0.1
  lookback_L_default: 4096
  horizons_default: [1, 5, 20, 60]
  quantiles_default: [0.1, 0.25, 0.5, 0.75, 0.9]
feature:
  epsilon: 1e-9
  session_period_minutes: 1440
limits:
  max_L: 8192
  max_horizons: 16
  max_quantiles: 19
