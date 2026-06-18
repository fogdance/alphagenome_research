"""Base feature schema helpers.

The historical M1/M2 data contract is the 8D ``m1_f8`` profile. Newer
datasets must declare their active profile through
``data_pipeline.feature_profiles``; this module remains the base8 definition
used by M1 builders and control experiments.
"""

try:
    from .feature_profiles import BASE_FEATURE_COLS, base_feature_profile
except ImportError:  # Direct script execution with src/alphatrade on sys.path.
    from data_pipeline.feature_profiles import BASE_FEATURE_COLS, base_feature_profile


FEATURE_COLS = list(BASE_FEATURE_COLS)
FEATURE_DIM = len(FEATURE_COLS)
DEFAULT_FEATURE_PROFILE = base_feature_profile()

# Feature dtype
FEATURE_DTYPE = "float32"


def validate_feature_schema(df, strict=True, feature_cols=None):
    """
    Validate that DataFrame has correct feature schema.
    
    Args:
        df: DataFrame to validate
        strict: If True, require exact match of feature_cols
    
    Returns:
        (is_valid, error_message)
    """
    cols = list(feature_cols or FEATURE_COLS)
    expected_dim = len(cols)

    # Check all features present
    missing = [col for col in cols if col not in df.columns]
    if missing:
        return False, f"Missing features: {missing}"
    
    # Check dtypes
    for col in cols:
        if df[col].dtype != FEATURE_DTYPE:
            return False, f"Feature {col} has dtype {df[col].dtype}, expected {FEATURE_DTYPE}"
    
    # Check dimension
    if strict and len([c for c in df.columns if c in cols]) != expected_dim:
        return False, f"Expected {expected_dim} features, got {len([c for c in df.columns if c in cols])}"
    
    return True, None


def get_feature_info(feature_cols=None):
    """Get feature schema information."""
    cols = list(feature_cols or FEATURE_COLS)
    return {
        'feature_cols': cols,
        'feature_dim': len(cols),
        'feature_dtype': FEATURE_DTYPE
    }
