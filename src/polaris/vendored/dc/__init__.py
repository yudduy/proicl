# Vendored from https://github.com/suzgunmirac/dynamic-cheatsheet.git
# @ 5cfe3c37e8e52b1d858d0f3df46e7f17c50991b9
# Original license: see LICENSE in this directory.
#
# Scope: comparator-only (proposal R6). The `language_model.py` module imports
# `text_generation.UnifiedLLMClient` from a sibling package that polaris does
# not vendor; importing this submodule at runtime requires that dep.
