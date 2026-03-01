"""
Unified feature schema for AlphaTrade.

This module defines the canonical feature columns used across:
- Data generation (build_canonical_bars.py)
- Data loading (dataloader)
- Contract validation (check_m1_data_contract.py)

IMPORTANT: This is the single source of truth for feature definitions.
"""

# Feature columns in strict order (F=8 for AlphaTrade v0.2)
FEATURE_COLS = [
    "ret_1m",           # 0: 1-minute return
    "hl_range",         # 1: (high - low) / close
    "co_change",        # 2: (close - open) / open
    "vol_log1p",        # 3: log(1 + volume)
    "pos_log1p",        # 4: log(1 + position)
    "minute_sin",       # 5: sin(2π * minute / 1440)
    "minute_cos",       # 6: cos(2π * minute / 1440)
    "is_session_open",  # 7: session open indicator (1.0 for now)
]

# Feature dimension
FEATURE_DIM = len(FEATURE_COLS)

# Feature dtype
FEATURE_DTYPE = "float32"


def validate_feature_schema(df, strict=True):
    """
    Validate that DataFrame has correct feature schema.
    
    Args:
        df: DataFrame to validate
        strict: If True, require exact match of FEATURE_COLS
    
    Returns:
        (is_valid, error_message)
    """
    # Check all features present
    missing = [col for col in FEATURE_COLS if col not in df.columns]
    if missing:
        return False, f"Missing features: {missing}"
    
    # Check dtypes
    for col in FEATURE_COLS:
        if df[col].dtype != FEATURE_DTYPE:
            return False, f"Feature {col} has dtype {df[col].dtype}, expected {FEATURE_DTYPE}"
    
    # Check dimension
    if strict and len([c for c in df.columns if c in FEATURE_COLS]) != FEATURE_DIM:
        return False, f"Expected {FEATURE_DIM} features, got {len([c for c in df.columns if c in FEATURE_COLS])}"
    
    return True, None


def get_feature_info():
    """Get feature schema information."""
    return {
        'feature_cols': FEATURE_COLS,
        'feature_dim': FEATURE_DIM,
        'feature_dtype': FEATURE_DTYPE
    }
