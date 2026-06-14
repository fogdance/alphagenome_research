# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AlphaTrade: Multi-horizon financial time series predictor."""

from __future__ import annotations

import importlib


_LAZY_EXPORTS = {
    'api': 'alphatrade.api',
    'losses': 'alphatrade.core.losses',
    'model': 'alphatrade.core.model',
    'quality_metrics': 'alphatrade.quality_metrics',
    'runtime_paths': 'alphatrade.runtime_paths',
    'schemas': 'alphatrade.core.schemas',
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'alphatrade' has no attribute {name!r}")
    module = importlib.import_module(_LAZY_EXPORTS[name])
    globals()[name] = module
    return module
