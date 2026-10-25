# Follow-up 6: settings of each arm against its main-study arm (harness/plans/msa/config_diff.py)

Main study: as recorded in its sessions. Follow-up 6: as its plans, harness/configs.yaml, harness/run_af3.py and cloud/af3_run.sh resolve them, launched on demand. The input manifest and the label are the intended differences.

## V5o against V5 (v5e; 4 main-study sessions, 3 follow-up 6 plans)

| setting | main study (recorded) | follow-up 6 (planned) | |
|---|---|---|---|
| platform | `v5e` | `v5e` | same |
| provisioning | `standard` | `standard` | same |
| weights | `gcs` | `gcs` | same |
| machine type | `v5litepod-1` | `v5litepod-1` | same |
| image | `TPU runtime v2-alpha-tpuv5-lite` | `TPU runtime v2-alpha-tpuv5-lite` | same |
| libtpu | `0.0.43.2` | `0.0.43.2` | same |
| label | `main` | `msa` | intended |
| config name | `tpu_xla` | `tpu_xla` | same |
| config (harness/configs.yaml entry) | `{"description": "Cloud TPU (v5e or v6e, single chip), XLA attention, af3_tpu.patch", "flash_attention_implementation": "xla", "jax_backend": "tpu"}` | `{"description": "Cloud TPU (v5e or v6e, single chip), XLA attention, af3_tpu.patch", "flash_attention_implementation": "xla", "jax_backend": "tpu"}` | same |
| environment (xla_env) | `{"JAX_LOG_COMPILES": "1", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none"}` | `{"JAX_LOG_COMPILES": "1", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none"}` | same |
| AlphaFold3 commit | `85c4d20505fd5cef05eac22b534d4e793971ae69` | `85c4d20505fd5cef05eac22b534d4e793971ae69` | same |
| af3_tpu.patch SHA-256 | `e4bb8c326137ebb4903aad98d72592e8b6858c82dc6f4c1517848119e0520a4b` | `e4bb8c326137ebb4903aad98d72592e8b6858c82dc6f4c1517848119e0520a4b` | same |
| run_alphafold.py SHA-256 | `c62bfaca9476c57c519429abc7f57d8db05165487165cbb463308e98b5f7ace4` | `c62bfaca9476c57c519429abc7f57d8db05165487165cbb463308e98b5f7ace4` | same |
| buckets | `null` | `null` | same |
| save_embeddings | `false` | `false` | same |
| recycles, diffusion samples | `[null, null]` | `[null, null]` | same |
| per-process timeout (min), entry, log_compiles, order | `[180.0, true, true, "tokens"]` | `[180.0, true, true, "tokens"]` | same |
| input manifest | `inputs/manifest.csv` | `inputs/official_data/manifest.csv` | intended |
| seeds per process | `1` | `1` | same |
| libtpu build | `libtpu_lts_20260630_b_RC06` | `libtpu_lts_20260630_b_RC06` | same |
| run_alphafold command | `cd third_party/alphafold3 && JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=tpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | `cd third_party/alphafold3 && JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=tpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | same |
| repetitions of the arm (r = fresh, w = warm rerun) | `r1, r2, w` | `r1, w` | intended |

## V6o against V6 (v6e; 3 main-study sessions, 3 follow-up 6 plans)

| setting | main study (recorded) | follow-up 6 (planned) | |
|---|---|---|---|
| platform | `v6e` | `v6e` | same |
| provisioning | `standard` | `standard` | same |
| weights | `gcs` | `gcs` | same |
| machine type | `ct6e-standard-1t` | `ct6e-standard-1t` | same |
| image | `ubuntu-os-accelerator-images/ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e` | `ubuntu-os-accelerator-images/ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e` | same |
| libtpu | `0.0.43.2` | `0.0.43.2` | same |
| label | `main` | `msa` | intended |
| config name | `tpu_xla` | `tpu_xla` | same |
| config (harness/configs.yaml entry) | `{"description": "Cloud TPU (v5e or v6e, single chip), XLA attention, af3_tpu.patch", "flash_attention_implementation": "xla", "jax_backend": "tpu"}` | `{"description": "Cloud TPU (v5e or v6e, single chip), XLA attention, af3_tpu.patch", "flash_attention_implementation": "xla", "jax_backend": "tpu"}` | same |
| environment (xla_env) | `{"JAX_LOG_COMPILES": "1", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none"}` | `{"JAX_LOG_COMPILES": "1", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none"}` | same |
| AlphaFold3 commit | `85c4d20505fd5cef05eac22b534d4e793971ae69` | `85c4d20505fd5cef05eac22b534d4e793971ae69` | same |
| af3_tpu.patch SHA-256 | `e4bb8c326137ebb4903aad98d72592e8b6858c82dc6f4c1517848119e0520a4b` | `e4bb8c326137ebb4903aad98d72592e8b6858c82dc6f4c1517848119e0520a4b` | same |
| run_alphafold.py SHA-256 | `c62bfaca9476c57c519429abc7f57d8db05165487165cbb463308e98b5f7ace4` | `c62bfaca9476c57c519429abc7f57d8db05165487165cbb463308e98b5f7ace4` | same |
| buckets | `null` | `null` | same |
| save_embeddings | `false` | `false` | same |
| recycles, diffusion samples | `[null, null]` | `[null, null]` | same |
| per-process timeout (min), entry, log_compiles, order | `[180.0, true, true, "tokens"]` | `[180.0, true, true, "tokens"]` | same |
| input manifest | `inputs/manifest.csv` | `inputs/official_data/manifest.csv` | intended |
| seeds per process | `1` | `1` | same |
| libtpu build | `libtpu_lts_20260630_b_RC06` | `libtpu_lts_20260630_b_RC06` | same |
| run_alphafold command | `cd third_party/alphafold3 && JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=tpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | `cd third_party/alphafold3 && JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=tpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | same |
| repetitions of the arm (r = fresh, w = warm rerun) | `r1, r2, w` | `r1, w` | intended |

## Go against G (l4; 14 main-study sessions, 14 follow-up 6 plans)

| setting | main study (recorded) | follow-up 6 (planned) | |
|---|---|---|---|
| platform | `l4` | `l4` | same |
| provisioning | `standard` | `standard` | same |
| weights | `gcs` | `gcs` | same |
| machine type | `g2-standard-8` | `g2-standard-8` | same |
| image | `ubuntu-os-accelerator-images/ubuntu-accelerator-2204-amd64-with-nvidia-580` | `ubuntu-os-accelerator-images/ubuntu-accelerator-2204-amd64-with-nvidia-580` | same |
| label | `main` | `msa` | intended |
| config name | `l4_xla` | `l4_xla` | same |
| config (harness/configs.yaml entry) | `{"description": "NVIDIA L4 (compute capability 8.9), XLA attention", "env": {"JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}, "flash_attention_implementation": "xla", "jax_backend": "gpu"}` | `{"description": "NVIDIA L4 (compute capability 8.9), XLA attention", "env": {"JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}, "flash_attention_implementation": "xla", "jax_backend": "gpu"}` | same |
| environment (xla_env) | `{"JAX_LOG_COMPILES": "1", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none", "JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}` | `{"JAX_LOG_COMPILES": "1", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none", "JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}` | same |
| AlphaFold3 commit | `85c4d20505fd5cef05eac22b534d4e793971ae69` | `85c4d20505fd5cef05eac22b534d4e793971ae69` | same |
| af3_tpu.patch SHA-256 | `e4bb8c326137ebb4903aad98d72592e8b6858c82dc6f4c1517848119e0520a4b` | `e4bb8c326137ebb4903aad98d72592e8b6858c82dc6f4c1517848119e0520a4b` | same |
| run_alphafold.py SHA-256 | `c62bfaca9476c57c519429abc7f57d8db05165487165cbb463308e98b5f7ace4` | `c62bfaca9476c57c519429abc7f57d8db05165487165cbb463308e98b5f7ace4` | same |
| buckets | `null` | `null` | same |
| save_embeddings | `false` | `false` | same |
| recycles, diffusion samples | `[null, null]` | `[null, null]` | same |
| per-process timeout (min), entry, log_compiles, order | `[180.0, true, true, "tokens"]` | `[180.0, true, true, "tokens"]` | same |
| input manifest | `inputs/manifest.csv` | `inputs/official_data/manifest.csv` | intended |
| seeds per process | `1` | `1` | same |
| run_alphafold command | `cd third_party/alphafold3 && JAX_PLATFORMS=cuda XLA_FLAGS=--xla_gpu_enable_triton_gemm=false XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_CLIENT_MEM_FRACTION=0.95 JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=gpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | `cd third_party/alphafold3 && JAX_PLATFORMS=cuda XLA_FLAGS=--xla_gpu_enable_triton_gemm=false XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_CLIENT_MEM_FRACTION=0.95 JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=gpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | same |
| repetitions of the arm (r = fresh, w = warm rerun) | `r1, r2, w` | `r1, r2` | intended |

Settings that differ beyond the input: 0
