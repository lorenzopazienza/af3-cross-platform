# Follow-up 13: settings of each new arm against its source arm (harness/plans/config_diff_fu13.py)

Source arm: as recorded in its main-study sessions. New arm: as its plans, harness/configs.yaml, harness/run_af3.py and cloud/af3_run.sh resolve them, with the provisioning of the row (on demand on v5e and L4, Flex-start on v6e). Allowed differences: the label, the follow-up 13 settings (configuration name and description, AF3_FU13_* variables, JAX_DEFAULT_MATMUL_PRECISION=highest for the float32 arms), the seeds, the repetitions and the provisioning. Rows marked "without the follow-up 13 settings" compare the configuration, the environment and the command again with those settings removed, and must be the same.

## L5 against V5 (4 source sessions, 1 plans): capture

Follow-up 13 settings of L5: `AF3_FU13_CAPTURE=1`

| setting | source arm (recorded) | new arm (planned) | |
|---|---|---|---|
| platform | `v5e` | `v5e` | same |
| provisioning | `standard` | `standard` | same |
| weights | `gcs` | `gcs` | same |
| machine type | `v5litepod-1` | `v5litepod-1` | same |
| image | `TPU runtime v2-alpha-tpuv5-lite` | `TPU runtime v2-alpha-tpuv5-lite` | same |
| libtpu | `0.0.43.2` | `0.0.43.2` | same |
| label | `main` | `why` | intended |
| config name | `tpu_xla` | `tpu_xla_fu13_capture` | intended |
| config description | `Cloud TPU (v5e or v6e, single chip), XLA attention, af3_tpu.patch` | `Cloud TPU, XLA attention, af3_tpu.patch, layer-wise capture of the trunk (follow-up 13, L)` | intended |
| config (harness/configs.yaml entry, without its description) | `{"flash_attention_implementation": "xla", "jax_backend": "tpu"}` | `{"env": {"AF3_FU13_CAPTURE": "1"}, "flash_attention_implementation": "xla", "jax_backend": "tpu"}` | intended |
| environment (xla_env) | `{"JAX_LOG_COMPILES": "1", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none"}` | `{"JAX_LOG_COMPILES": "1", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none"}` | same |
| software stack (jax, jaxlib, libtpu or jax-cuda12 plugin and pjrt) | `{"jax": "0.10.2", "jaxlib": "0.10.2", "libtpu": "0.0.43.2"}` | `{"jax": "0.10.2", "jaxlib": "0.10.2", "libtpu": "0.0.43.2"}` | same |
| AlphaFold3 commit | `85c4d20505fd5cef05eac22b534d4e793971ae69` | `85c4d20505fd5cef05eac22b534d4e793971ae69` | same |
| af3_tpu.patch SHA-256 | `e4bb8c326137ebb4903aad98d72592e8b6858c82dc6f4c1517848119e0520a4b` | `e4bb8c326137ebb4903aad98d72592e8b6858c82dc6f4c1517848119e0520a4b` | same |
| run_alphafold.py SHA-256 | `c62bfaca9476c57c519429abc7f57d8db05165487165cbb463308e98b5f7ace4` | `c62bfaca9476c57c519429abc7f57d8db05165487165cbb463308e98b5f7ace4` | same |
| buckets | `null` | `null` | same |
| save_embeddings | `false` | `false` | same |
| recycles, diffusion samples | `[null, null]` | `[null, null]` | same |
| per-process timeout (min), entry, log_compiles, order | `[180.0, true, true, "tokens"]` | `[180.0, true, true, "tokens"]` | same |
| input manifest | `inputs/manifest.csv` | `inputs/manifest.csv` | same |
| seeds per process | `1` | `1` | same |
| libtpu build | `libtpu_lts_20260630_b_RC06` | `libtpu_lts_20260630_b_RC06` | same |
| run_alphafold command | `cd third_party/alphafold3 && JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=tpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | `cd third_party/alphafold3 && AF3_FU13_CAPTURE=1 JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=tpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | intended |
| repetitions of the arm (r = fresh, w = warm rerun) | `r1, r2, w` | `r1` | intended |
| seeds of the arm | `1 to 5` | `1 to 1` | intended |
| config (harness/configs.yaml entry, without its description), without the follow-up 13 settings | `{"flash_attention_implementation": "xla", "jax_backend": "tpu"}` | `{"flash_attention_implementation": "xla", "jax_backend": "tpu"}` | same |
| environment (xla_env), without the follow-up 13 settings | `{"JAX_LOG_COMPILES": "1", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none"}` | `{"JAX_LOG_COMPILES": "1", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none"}` | same |
| run_alphafold command, without the follow-up 13 settings | `cd third_party/alphafold3 && JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=tpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | `cd third_party/alphafold3 && JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=tpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | same |

## L6 against V6 (3 source sessions, 1 plans): capture

Follow-up 13 settings of L6: `AF3_FU13_CAPTURE=1`

| setting | source arm (recorded) | new arm (planned) | |
|---|---|---|---|
| platform | `v6e` | `v6e` | same |
| provisioning | `standard` | `flex_start` | intended |
| weights | `gcs` | `gcs` | same |
| machine type | `ct6e-standard-1t` | `ct6e-standard-1t` | same |
| image | `ubuntu-os-accelerator-images/ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e` | `ubuntu-os-accelerator-images/ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e` | same |
| libtpu | `0.0.43.2` | `0.0.43.2` | same |
| label | `main` | `why` | intended |
| config name | `tpu_xla` | `tpu_xla_fu13_capture` | intended |
| config description | `Cloud TPU (v5e or v6e, single chip), XLA attention, af3_tpu.patch` | `Cloud TPU, XLA attention, af3_tpu.patch, layer-wise capture of the trunk (follow-up 13, L)` | intended |
| config (harness/configs.yaml entry, without its description) | `{"flash_attention_implementation": "xla", "jax_backend": "tpu"}` | `{"env": {"AF3_FU13_CAPTURE": "1"}, "flash_attention_implementation": "xla", "jax_backend": "tpu"}` | intended |
| environment (xla_env) | `{"JAX_LOG_COMPILES": "1", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none"}` | `{"JAX_LOG_COMPILES": "1", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none"}` | same |
| software stack (jax, jaxlib, libtpu or jax-cuda12 plugin and pjrt) | `{"jax": "0.10.2", "jaxlib": "0.10.2", "libtpu": "0.0.43.2"}` | `{"jax": "0.10.2", "jaxlib": "0.10.2", "libtpu": "0.0.43.2"}` | same |
| AlphaFold3 commit | `85c4d20505fd5cef05eac22b534d4e793971ae69` | `85c4d20505fd5cef05eac22b534d4e793971ae69` | same |
| af3_tpu.patch SHA-256 | `e4bb8c326137ebb4903aad98d72592e8b6858c82dc6f4c1517848119e0520a4b` | `e4bb8c326137ebb4903aad98d72592e8b6858c82dc6f4c1517848119e0520a4b` | same |
| run_alphafold.py SHA-256 | `c62bfaca9476c57c519429abc7f57d8db05165487165cbb463308e98b5f7ace4` | `c62bfaca9476c57c519429abc7f57d8db05165487165cbb463308e98b5f7ace4` | same |
| buckets | `null` | `null` | same |
| save_embeddings | `false` | `false` | same |
| recycles, diffusion samples | `[null, null]` | `[null, null]` | same |
| per-process timeout (min), entry, log_compiles, order | `[180.0, true, true, "tokens"]` | `[180.0, true, true, "tokens"]` | same |
| input manifest | `inputs/manifest.csv` | `inputs/manifest.csv` | same |
| seeds per process | `1` | `1` | same |
| libtpu build | `libtpu_lts_20260630_b_RC06` | `libtpu_lts_20260630_b_RC06` | same |
| run_alphafold command | `cd third_party/alphafold3 && JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=tpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | `cd third_party/alphafold3 && AF3_FU13_CAPTURE=1 JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=tpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | intended |
| repetitions of the arm (r = fresh, w = warm rerun) | `r1, r2, w` | `r1` | intended |
| seeds of the arm | `1 to 5` | `1 to 1` | intended |
| config (harness/configs.yaml entry, without its description), without the follow-up 13 settings | `{"flash_attention_implementation": "xla", "jax_backend": "tpu"}` | `{"flash_attention_implementation": "xla", "jax_backend": "tpu"}` | same |
| environment (xla_env), without the follow-up 13 settings | `{"JAX_LOG_COMPILES": "1", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none"}` | `{"JAX_LOG_COMPILES": "1", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none"}` | same |
| run_alphafold command, without the follow-up 13 settings | `cd third_party/alphafold3 && JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=tpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | `cd third_party/alphafold3 && JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=tpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | same |

## LG against G (14 source sessions, 1 plans): capture

Follow-up 13 settings of LG: `AF3_FU13_CAPTURE=1`

| setting | source arm (recorded) | new arm (planned) | |
|---|---|---|---|
| platform | `l4` | `l4` | same |
| provisioning | `standard` | `standard` | same |
| weights | `gcs` | `gcs` | same |
| machine type | `g2-standard-8` | `g2-standard-8` | same |
| image | `ubuntu-os-accelerator-images/ubuntu-accelerator-2204-amd64-with-nvidia-580` | `ubuntu-os-accelerator-images/ubuntu-accelerator-2204-amd64-with-nvidia-580` | same |
| label | `main` | `why` | intended |
| config name | `l4_xla` | `l4_xla_fu13_capture` | intended |
| config description | `NVIDIA L4 (compute capability 8.9), XLA attention` | `NVIDIA L4, XLA attention, layer-wise capture of the trunk (follow-up 13, L)` | intended |
| config (harness/configs.yaml entry, without its description) | `{"env": {"JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}, "flash_attention_implementation": "xla", "jax_backend": "gpu"}` | `{"env": {"AF3_FU13_CAPTURE": "1", "JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}, "flash_attention_implementation": "xla", "jax_backend": "gpu"}` | intended |
| environment (xla_env) | `{"JAX_LOG_COMPILES": "1", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none", "JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}` | `{"JAX_LOG_COMPILES": "1", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none", "JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}` | same |
| software stack (jax, jaxlib, libtpu or jax-cuda12 plugin and pjrt) | `{"jax": "0.10.2", "jax-cuda12-pjrt": "0.10.2", "jax-cuda12-plugin": "0.10.2", "jaxlib": "0.10.2"}` | `{"jax": "0.10.2", "jax-cuda12-pjrt": "0.10.2", "jax-cuda12-plugin": "0.10.2", "jaxlib": "0.10.2"}` | same |
| AlphaFold3 commit | `85c4d20505fd5cef05eac22b534d4e793971ae69` | `85c4d20505fd5cef05eac22b534d4e793971ae69` | same |
| af3_tpu.patch SHA-256 | `e4bb8c326137ebb4903aad98d72592e8b6858c82dc6f4c1517848119e0520a4b` | `e4bb8c326137ebb4903aad98d72592e8b6858c82dc6f4c1517848119e0520a4b` | same |
| run_alphafold.py SHA-256 | `c62bfaca9476c57c519429abc7f57d8db05165487165cbb463308e98b5f7ace4` | `c62bfaca9476c57c519429abc7f57d8db05165487165cbb463308e98b5f7ace4` | same |
| buckets | `null` | `null` | same |
| save_embeddings | `false` | `false` | same |
| recycles, diffusion samples | `[null, null]` | `[null, null]` | same |
| per-process timeout (min), entry, log_compiles, order | `[180.0, true, true, "tokens"]` | `[180.0, true, true, "tokens"]` | same |
| input manifest | `inputs/manifest.csv` | `inputs/manifest.csv` | same |
| seeds per process | `1` | `1` | same |
| run_alphafold command | `cd third_party/alphafold3 && JAX_PLATFORMS=cuda XLA_FLAGS=--xla_gpu_enable_triton_gemm=false XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_CLIENT_MEM_FRACTION=0.95 JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=gpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | `cd third_party/alphafold3 && JAX_PLATFORMS=cuda XLA_FLAGS=--xla_gpu_enable_triton_gemm=false XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_CLIENT_MEM_FRACTION=0.95 AF3_FU13_CAPTURE=1 JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=gpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | intended |
| repetitions of the arm (r = fresh, w = warm rerun) | `r1, r2, w` | `r1, r2` | intended |
| seeds of the arm | `1 to 5` | `1 to 1` | intended |
| config (harness/configs.yaml entry, without its description), without the follow-up 13 settings | `{"env": {"JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}, "flash_attention_implementation": "xla", "jax_backend": "gpu"}` | `{"env": {"JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}, "flash_attention_implementation": "xla", "jax_backend": "gpu"}` | same |
| environment (xla_env), without the follow-up 13 settings | `{"JAX_LOG_COMPILES": "1", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none", "JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}` | `{"JAX_LOG_COMPILES": "1", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none", "JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}` | same |
| run_alphafold command, without the follow-up 13 settings | `cd third_party/alphafold3 && JAX_PLATFORMS=cuda XLA_FLAGS=--xla_gpu_enable_triton_gemm=false XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_CLIENT_MEM_FRACTION=0.95 JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=gpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | `cd third_party/alphafold3 && JAX_PLATFORMS=cuda XLA_FLAGS=--xla_gpu_enable_triton_gemm=false XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_CLIENT_MEM_FRACTION=0.95 JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=gpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | same |

## V6f against V6 (3 source sessions, 9 plans): float32 trunk

Follow-up 13 settings of V6f: `AF3_FU13_BFLOAT16=none`, `JAX_DEFAULT_MATMUL_PRECISION=highest`

| setting | source arm (recorded) | new arm (planned) | |
|---|---|---|---|
| platform | `v6e` | `v6e` | same |
| provisioning | `standard` | `flex_start` | intended |
| weights | `gcs` | `gcs` | same |
| machine type | `ct6e-standard-1t` | `ct6e-standard-1t` | same |
| image | `ubuntu-os-accelerator-images/ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e` | `ubuntu-os-accelerator-images/ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e` | same |
| libtpu | `0.0.43.2` | `0.0.43.2` | same |
| label | `main` | `why` | intended |
| config name | `tpu_xla` | `tpu_xla_fu13_f32` | intended |
| config description | `Cloud TPU (v5e or v6e, single chip), XLA attention, af3_tpu.patch` | `Cloud TPU, XLA attention, af3_tpu.patch, float32 trunk (bfloat16 none, matmuls highest; follow-up 13, V6f)` | intended |
| config (harness/configs.yaml entry, without its description) | `{"flash_attention_implementation": "xla", "jax_backend": "tpu"}` | `{"env": {"AF3_FU13_BFLOAT16": "none", "JAX_DEFAULT_MATMUL_PRECISION": "highest"}, "flash_attention_implementation": "xla", "jax_backend": "tpu"}` | intended |
| environment (xla_env) | `{"JAX_LOG_COMPILES": "1", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none"}` | `{"JAX_DEFAULT_MATMUL_PRECISION": "highest", "JAX_LOG_COMPILES": "1", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none"}` | intended |
| software stack (jax, jaxlib, libtpu or jax-cuda12 plugin and pjrt) | `{"jax": "0.10.2", "jaxlib": "0.10.2", "libtpu": "0.0.43.2"}` | `{"jax": "0.10.2", "jaxlib": "0.10.2", "libtpu": "0.0.43.2"}` | same |
| AlphaFold3 commit | `85c4d20505fd5cef05eac22b534d4e793971ae69` | `85c4d20505fd5cef05eac22b534d4e793971ae69` | same |
| af3_tpu.patch SHA-256 | `e4bb8c326137ebb4903aad98d72592e8b6858c82dc6f4c1517848119e0520a4b` | `e4bb8c326137ebb4903aad98d72592e8b6858c82dc6f4c1517848119e0520a4b` | same |
| run_alphafold.py SHA-256 | `c62bfaca9476c57c519429abc7f57d8db05165487165cbb463308e98b5f7ace4` | `c62bfaca9476c57c519429abc7f57d8db05165487165cbb463308e98b5f7ace4` | same |
| buckets | `null` | `null` | same |
| save_embeddings | `false` | `false` | same |
| recycles, diffusion samples | `[null, null]` | `[null, null]` | same |
| per-process timeout (min), entry, log_compiles, order | `[180.0, true, true, "tokens"]` | `[180.0, true, true, "tokens"]` | same |
| input manifest | `inputs/manifest.csv` | `inputs/manifest.csv` | same |
| seeds per process | `1` | `1` | same |
| libtpu build | `libtpu_lts_20260630_b_RC06` | `libtpu_lts_20260630_b_RC06` | same |
| run_alphafold command | `cd third_party/alphafold3 && JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=tpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | `cd third_party/alphafold3 && JAX_DEFAULT_MATMUL_PRECISION=highest AF3_FU13_BFLOAT16=none JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=tpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | intended |
| repetitions of the arm (r = fresh, w = warm rerun) | `r1, r2, w` | `r1, r2` | intended |
| seeds of the arm | `1 to 5` | `1 to 5` | same |
| config (harness/configs.yaml entry, without its description), without the follow-up 13 settings | `{"flash_attention_implementation": "xla", "jax_backend": "tpu"}` | `{"flash_attention_implementation": "xla", "jax_backend": "tpu"}` | same |
| environment (xla_env), without the follow-up 13 settings | `{"JAX_LOG_COMPILES": "1", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none"}` | `{"JAX_LOG_COMPILES": "1", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none"}` | same |
| run_alphafold command, without the follow-up 13 settings | `cd third_party/alphafold3 && JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=tpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | `cd third_party/alphafold3 && JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=tpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | same |

## P against V6 (3 source sessions, 7 plans): perturbation

Follow-up 13 settings of P: `AF3_FU13_PERTURB_EPS=0`, `AF3_FU13_PERTURB_EPS=2^-11`, `AF3_FU13_PERTURB_EPS=2^-14`, `AF3_FU13_PERTURB_EPS=2^-5`, `AF3_FU13_PERTURB_EPS=2^-6`, `AF3_FU13_PERTURB_EPS=2^-7`, `AF3_FU13_PERTURB_EPS=2^-8`, `AF3_FU13_PERTURB_SEED=2026`

| setting | source arm (recorded) | new arm (planned) | |
|---|---|---|---|
| platform | `v6e` | `v6e` | same |
| provisioning | `standard` | `flex_start` | intended |
| weights | `gcs` | `gcs` | same |
| machine type | `ct6e-standard-1t` | `ct6e-standard-1t` | same |
| image | `ubuntu-os-accelerator-images/ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e` | `ubuntu-os-accelerator-images/ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e` | same |
| libtpu | `0.0.43.2` | `0.0.43.2` | same |
| label | `main` | `why` | intended |
| config name | `tpu_xla` | `tpu_xla_fu13_p0`<br>`tpu_xla_fu13_p11`<br>`tpu_xla_fu13_p14`<br>`tpu_xla_fu13_p5`<br>`tpu_xla_fu13_p6`<br>`tpu_xla_fu13_p7`<br>`tpu_xla_fu13_p8` | intended |
| config description | `Cloud TPU (v5e or v6e, single chip), XLA attention, af3_tpu.patch` | `Cloud TPU, XLA attention, af3_tpu.patch, trunk output perturbed at relative size 0 (follow-up 13, P0)`<br>`Cloud TPU, XLA attention, af3_tpu.patch, trunk output perturbed at relative size 2^-11 (follow-up 13, P11)`<br>`Cloud TPU, XLA attention, af3_tpu.patch, trunk output perturbed at relative size 2^-14 (follow-up 13, P14)`<br>`Cloud TPU, XLA attention, af3_tpu.patch, trunk output perturbed at relative size 2^-5 (follow-up 13, P5)`<br>`Cloud TPU, XLA attention, af3_tpu.patch, trunk output perturbed at relative size 2^-6 (follow-up 13, P6)`<br>`Cloud TPU, XLA attention, af3_tpu.patch, trunk output perturbed at relative size 2^-7 (follow-up 13, P7)`<br>`Cloud TPU, XLA attention, af3_tpu.patch, trunk output perturbed at relative size 2^-8 (follow-up 13, P8)` | intended |
| config (harness/configs.yaml entry, without its description) | `{"flash_attention_implementation": "xla", "jax_backend": "tpu"}` | `{"env": {"AF3_FU13_PERTURB_EPS": "0", "AF3_FU13_PERTURB_SEED": "2026"}, "flash_attention_implementation": "xla", "jax_backend": "tpu"}`<br>`{"env": {"AF3_FU13_PERTURB_EPS": "2^-11", "AF3_FU13_PERTURB_SEED": "2026"}, "flash_attention_implementation": "xla", "jax_backend": "tpu"}`<br>`{"env": {"AF3_FU13_PERTURB_EPS": "2^-14", "AF3_FU13_PERTURB_SEED": "2026"}, "flash_attention_implementation": "xla", "jax_backend": "tpu"}`<br>`{"env": {"AF3_FU13_PERTURB_EPS": "2^-5", "AF3_FU13_PERTURB_SEED": "2026"}, "flash_attention_implementation": "xla", "jax_backend": "tpu"}`<br>`{"env": {"AF3_FU13_PERTURB_EPS": "2^-6", "AF3_FU13_PERTURB_SEED": "2026"}, "flash_attention_implementation": "xla", "jax_backend": "tpu"}`<br>`{"env": {"AF3_FU13_PERTURB_EPS": "2^-7", "AF3_FU13_PERTURB_SEED": "2026"}, "flash_attention_implementation": "xla", "jax_backend": "tpu"}`<br>`{"env": {"AF3_FU13_PERTURB_EPS": "2^-8", "AF3_FU13_PERTURB_SEED": "2026"}, "flash_attention_implementation": "xla", "jax_backend": "tpu"}` | intended |
| environment (xla_env) | `{"JAX_LOG_COMPILES": "1", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none"}` | `{"JAX_LOG_COMPILES": "1", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none"}` | same |
| software stack (jax, jaxlib, libtpu or jax-cuda12 plugin and pjrt) | `{"jax": "0.10.2", "jaxlib": "0.10.2", "libtpu": "0.0.43.2"}` | `{"jax": "0.10.2", "jaxlib": "0.10.2", "libtpu": "0.0.43.2"}` | same |
| AlphaFold3 commit | `85c4d20505fd5cef05eac22b534d4e793971ae69` | `85c4d20505fd5cef05eac22b534d4e793971ae69` | same |
| af3_tpu.patch SHA-256 | `e4bb8c326137ebb4903aad98d72592e8b6858c82dc6f4c1517848119e0520a4b` | `e4bb8c326137ebb4903aad98d72592e8b6858c82dc6f4c1517848119e0520a4b` | same |
| run_alphafold.py SHA-256 | `c62bfaca9476c57c519429abc7f57d8db05165487165cbb463308e98b5f7ace4` | `c62bfaca9476c57c519429abc7f57d8db05165487165cbb463308e98b5f7ace4` | same |
| buckets | `null` | `null` | same |
| save_embeddings | `false` | `false` | same |
| recycles, diffusion samples | `[null, null]` | `[null, null]` | same |
| per-process timeout (min), entry, log_compiles, order | `[180.0, true, true, "tokens"]` | `[180.0, true, true, "tokens"]` | same |
| input manifest | `inputs/manifest.csv` | `inputs/manifest.csv` | same |
| seeds per process | `1` | `1` | same |
| libtpu build | `libtpu_lts_20260630_b_RC06` | `libtpu_lts_20260630_b_RC06` | same |
| run_alphafold command | `cd third_party/alphafold3 && JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=tpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | `cd third_party/alphafold3 && AF3_FU13_PERTURB_EPS=0 AF3_FU13_PERTURB_SEED=2026 JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=tpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>`<br>`cd third_party/alphafold3 && AF3_FU13_PERTURB_EPS=2^-11 AF3_FU13_PERTURB_SEED=2026 JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=tpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>`<br>`cd third_party/alphafold3 && AF3_FU13_PERTURB_EPS=2^-14 AF3_FU13_PERTURB_SEED=2026 JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=tpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>`<br>`cd third_party/alphafold3 && AF3_FU13_PERTURB_EPS=2^-5 AF3_FU13_PERTURB_SEED=2026 JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=tpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>`<br>`cd third_party/alphafold3 && AF3_FU13_PERTURB_EPS=2^-6 AF3_FU13_PERTURB_SEED=2026 JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=tpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>`<br>`cd third_party/alphafold3 && AF3_FU13_PERTURB_EPS=2^-7 AF3_FU13_PERTURB_SEED=2026 JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=tpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>`<br>`cd third_party/alphafold3 && AF3_FU13_PERTURB_EPS=2^-8 AF3_FU13_PERTURB_SEED=2026 JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=tpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | intended |
| repetitions of the arm (r = fresh, w = warm rerun) | `r1, r2, w` | `r1` | intended |
| seeds of the arm | `1 to 5` | `1 to 2` | intended |
| config (harness/configs.yaml entry, without its description), without the follow-up 13 settings | `{"flash_attention_implementation": "xla", "jax_backend": "tpu"}` | `{"flash_attention_implementation": "xla", "jax_backend": "tpu"}` | same |
| environment (xla_env), without the follow-up 13 settings | `{"JAX_LOG_COMPILES": "1", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none"}` | `{"JAX_LOG_COMPILES": "1", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none"}` | same |
| run_alphafold command, without the follow-up 13 settings | `cd third_party/alphafold3 && JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=tpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | `cd third_party/alphafold3 && JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=tpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | same |

## Gf against G (14 source sessions, 60 plans): float32 trunk

Follow-up 13 settings of Gf: `AF3_FU13_BFLOAT16=none`, `JAX_DEFAULT_MATMUL_PRECISION=highest`

| setting | source arm (recorded) | new arm (planned) | |
|---|---|---|---|
| platform | `l4` | `l4` | same |
| provisioning | `standard` | `standard` | same |
| weights | `gcs` | `gcs` | same |
| machine type | `g2-standard-8` | `g2-standard-8` | same |
| image | `ubuntu-os-accelerator-images/ubuntu-accelerator-2204-amd64-with-nvidia-580` | `ubuntu-os-accelerator-images/ubuntu-accelerator-2204-amd64-with-nvidia-580` | same |
| label | `main` | `why` | intended |
| config name | `l4_xla` | `l4_xla_fu13_f32` | intended |
| config description | `NVIDIA L4 (compute capability 8.9), XLA attention` | `NVIDIA L4, XLA attention, float32 trunk (bfloat16 none, matmuls highest; follow-up 13, Gf)` | intended |
| config (harness/configs.yaml entry, without its description) | `{"env": {"JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}, "flash_attention_implementation": "xla", "jax_backend": "gpu"}` | `{"env": {"AF3_FU13_BFLOAT16": "none", "JAX_DEFAULT_MATMUL_PRECISION": "highest", "JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}, "flash_attention_implementation": "xla", "jax_backend": "gpu"}` | intended |
| environment (xla_env) | `{"JAX_LOG_COMPILES": "1", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none", "JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}` | `{"JAX_DEFAULT_MATMUL_PRECISION": "highest", "JAX_LOG_COMPILES": "1", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none", "JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}` | intended |
| software stack (jax, jaxlib, libtpu or jax-cuda12 plugin and pjrt) | `{"jax": "0.10.2", "jax-cuda12-pjrt": "0.10.2", "jax-cuda12-plugin": "0.10.2", "jaxlib": "0.10.2"}` | `{"jax": "0.10.2", "jax-cuda12-pjrt": "0.10.2", "jax-cuda12-plugin": "0.10.2", "jaxlib": "0.10.2"}` | same |
| AlphaFold3 commit | `85c4d20505fd5cef05eac22b534d4e793971ae69` | `85c4d20505fd5cef05eac22b534d4e793971ae69` | same |
| af3_tpu.patch SHA-256 | `e4bb8c326137ebb4903aad98d72592e8b6858c82dc6f4c1517848119e0520a4b` | `e4bb8c326137ebb4903aad98d72592e8b6858c82dc6f4c1517848119e0520a4b` | same |
| run_alphafold.py SHA-256 | `c62bfaca9476c57c519429abc7f57d8db05165487165cbb463308e98b5f7ace4` | `c62bfaca9476c57c519429abc7f57d8db05165487165cbb463308e98b5f7ace4` | same |
| buckets | `null` | `null` | same |
| save_embeddings | `false` | `false` | same |
| recycles, diffusion samples | `[null, null]` | `[null, null]` | same |
| per-process timeout (min), entry, log_compiles, order | `[180.0, true, true, "tokens"]` | `[180.0, true, true, "tokens"]` | same |
| input manifest | `inputs/manifest.csv` | `inputs/manifest.csv` | same |
| seeds per process | `1` | `1` | same |
| run_alphafold command | `cd third_party/alphafold3 && JAX_PLATFORMS=cuda XLA_FLAGS=--xla_gpu_enable_triton_gemm=false XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_CLIENT_MEM_FRACTION=0.95 JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=gpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | `cd third_party/alphafold3 && JAX_PLATFORMS=cuda XLA_FLAGS=--xla_gpu_enable_triton_gemm=false XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_CLIENT_MEM_FRACTION=0.95 JAX_DEFAULT_MATMUL_PRECISION=highest AF3_FU13_BFLOAT16=none JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=gpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | intended |
| repetitions of the arm (r = fresh, w = warm rerun) | `r1, r2, w` | `r1, r2` | intended |
| seeds of the arm | `1 to 5` | `1 to 5` | same |
| config (harness/configs.yaml entry, without its description), without the follow-up 13 settings | `{"env": {"JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}, "flash_attention_implementation": "xla", "jax_backend": "gpu"}` | `{"env": {"JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}, "flash_attention_implementation": "xla", "jax_backend": "gpu"}` | same |
| environment (xla_env), without the follow-up 13 settings | `{"JAX_LOG_COMPILES": "1", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none", "JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}` | `{"JAX_LOG_COMPILES": "1", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none", "JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}` | same |
| run_alphafold command, without the follow-up 13 settings | `cd third_party/alphafold3 && JAX_PLATFORMS=cuda XLA_FLAGS=--xla_gpu_enable_triton_gemm=false XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_CLIENT_MEM_FRACTION=0.95 JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=gpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | `cd third_party/alphafold3 && JAX_PLATFORMS=cuda XLA_FLAGS=--xla_gpu_enable_triton_gemm=false XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_CLIENT_MEM_FRACTION=0.95 JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=gpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | same |

Settings that differ beyond the allowed ones: 0
