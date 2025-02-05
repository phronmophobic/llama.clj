**Change Log**

In general, updates should be non-breaking, but due to llama.cpp, some breaking changes are unavoidable. This document tries to highlight known breaking changes where possible.

See also https://github.com/ggerganov/llama.cpp/issues/9289

**Version `0.8.6`**
- Updated llama.cpp compatibility to `b4634`
- Some options to `com.phronemophobic.llama/create-context` were removed.
  - `:seed` was removed. use `:seed` option during generation.
  - `:low-vram`
  - `:mul_mat_q`
  - `:f16-kv`
  - `:gqa`
  - `rms-norm-eps`


