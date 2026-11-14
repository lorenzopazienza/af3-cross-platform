# Follow-up 15: settings of each r2 arm against the r1 sessions it repeats (harness/plans/config_diff_rcx.py)

Source: as recorded in the r1 sessions (Af and Hf: follow-up 14; GT: main study), with the image each session's wrapper log printed and the Eigen source of its build. New: as the plans of harness/plans/rcx, harness/configs.yaml, harness/run_af3.py and cloud/af3_run.sh resolve them, Spot, with the image the launch sets (af2-tpu-benchmark/af3-accel-2204-nvidia-580 for Af r2 and Hf r2, af2-tpu-benchmark/af3-accel-2204-nvidia-580-gtr1 for GT r2) and the A100 machine type a2-ultragpu-1g. Allowed differences: the label, the repetitions, the provisioning, the image family and the Eigen source of GT (the archive of the same commit). Every other row must be the same.

## Af r2 against Af r1 (15 r1 sessions, 4 plans)

| setting | r1 (recorded) | r2 (planned) | |
|---|---|---|---|
| platform | `a100` | `a100` | same |
| provisioning | `spot` | `spot` | same |
| weights | `gcs` | `gcs` | same |
| machine type | `a2-ultragpu-1g` | `a2-ultragpu-1g` | same |
| label | `f32` | `rcx` | intended |
| config name | `a100_xla_fu14_f32` | `a100_xla_fu14_f32` | same |
| config description | `NVIDIA A100 80 GB, XLA attention, float32 trunk (bfloat16 none, matmuls highest; follow-up 14, Af)` | `NVIDIA A100 80 GB, XLA attention, float32 trunk (bfloat16 none, matmuls highest; follow-up 14, Af)` | same |
| config (harness/configs.yaml entry, without its description) | `{"env": {"AF3_FU13_BFLOAT16": "none", "JAX_DEFAULT_MATMUL_PRECISION": "highest", "JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}, "flash_attention_implementation": "xla", "jax_backend": "gpu"}` | `{"env": {"AF3_FU13_BFLOAT16": "none", "JAX_DEFAULT_MATMUL_PRECISION": "highest", "JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}, "flash_attention_implementation": "xla", "jax_backend": "gpu"}` | same |
| environment (xla_env) | `{"JAX_DEFAULT_MATMUL_PRECISION": "highest", "JAX_LOG_COMPILES": "1", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none", "JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}` | `{"JAX_DEFAULT_MATMUL_PRECISION": "highest", "JAX_LOG_COMPILES": "1", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none", "JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}` | same |
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
| run_alphafold command | `cd third_party/alphafold3 && JAX_PLATFORMS=cuda XLA_FLAGS=--xla_gpu_enable_triton_gemm=false XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_CLIENT_MEM_FRACTION=0.95 JAX_DEFAULT_MATMUL_PRECISION=highest AF3_FU13_BFLOAT16=none JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=gpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | `cd third_party/alphafold3 && JAX_PLATFORMS=cuda XLA_FLAGS=--xla_gpu_enable_triton_gemm=false XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_CLIENT_MEM_FRACTION=0.95 JAX_DEFAULT_MATMUL_PRECISION=highest AF3_FU13_BFLOAT16=none JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=gpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | same |
| repetitions of the arm (r = fresh, w = warm rerun) | `r1` | `r2` | intended |
| seeds of the arm | `1 to 5` | `1 to 5` | same |
| image (as the wrapper log printed it for each session) | `af2-tpu-benchmark/af3-accel-2204-nvidia-580 (2 sessions)`<br>`ubuntu-os-accelerator-images/ubuntu-accelerator-2204-amd64-with-nvidia-580 (13 sessions)` | `af2-tpu-benchmark/af3-accel-2204-nvidia-580 (set at launch)` | intended |
| Eigen source of the AlphaFold3 build | `archive 3.4.0 commit 3147391d946bb4b6c68edd901f2add6ac1f31f8c` | `archive 3.4.0 commit 3147391d946bb4b6c68edd901f2add6ac1f31f8c` | same |

## Hf r2 against Hf r1 (10 r1 sessions, 3 plans)

| setting | r1 (recorded) | r2 (planned) | |
|---|---|---|---|
| platform | `h100` | `h100` | same |
| provisioning | `spot` | `spot` | same |
| weights | `gcs` | `gcs` | same |
| machine type | `a3-highgpu-1g` | `a3-highgpu-1g` | same |
| label | `f32` | `rcx` | intended |
| config name | `h100_xla_fu14_f32` | `h100_xla_fu14_f32` | same |
| config description | `NVIDIA H100 80 GB, XLA attention, float32 trunk (bfloat16 none, matmuls highest; follow-up 14, Hf)` | `NVIDIA H100 80 GB, XLA attention, float32 trunk (bfloat16 none, matmuls highest; follow-up 14, Hf)` | same |
| config (harness/configs.yaml entry, without its description) | `{"env": {"AF3_FU13_BFLOAT16": "none", "JAX_DEFAULT_MATMUL_PRECISION": "highest", "JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}, "flash_attention_implementation": "xla", "jax_backend": "gpu"}` | `{"env": {"AF3_FU13_BFLOAT16": "none", "JAX_DEFAULT_MATMUL_PRECISION": "highest", "JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}, "flash_attention_implementation": "xla", "jax_backend": "gpu"}` | same |
| environment (xla_env) | `{"JAX_DEFAULT_MATMUL_PRECISION": "highest", "JAX_LOG_COMPILES": "1", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none", "JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}` | `{"JAX_DEFAULT_MATMUL_PRECISION": "highest", "JAX_LOG_COMPILES": "1", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none", "JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}` | same |
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
| run_alphafold command | `cd third_party/alphafold3 && JAX_PLATFORMS=cuda XLA_FLAGS=--xla_gpu_enable_triton_gemm=false XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_CLIENT_MEM_FRACTION=0.95 JAX_DEFAULT_MATMUL_PRECISION=highest AF3_FU13_BFLOAT16=none JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=gpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | `cd third_party/alphafold3 && JAX_PLATFORMS=cuda XLA_FLAGS=--xla_gpu_enable_triton_gemm=false XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_CLIENT_MEM_FRACTION=0.95 JAX_DEFAULT_MATMUL_PRECISION=highest AF3_FU13_BFLOAT16=none JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=gpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | same |
| repetitions of the arm (r = fresh, w = warm rerun) | `r1` | `r2` | intended |
| seeds of the arm | `1 to 5` | `1 to 5` | same |
| image (as the wrapper log printed it for each session) | `ubuntu-os-accelerator-images/ubuntu-accelerator-2204-amd64-with-nvidia-580 (10 sessions)` | `af2-tpu-benchmark/af3-accel-2204-nvidia-580 (set at launch)` | intended |
| Eigen source of the AlphaFold3 build | `archive 3.4.0 commit 3147391d946bb4b6c68edd901f2add6ac1f31f8c` | `archive 3.4.0 commit 3147391d946bb4b6c68edd901f2add6ac1f31f8c` | same |

## GT r2 against GT r1 (7 r1 sessions, 8 plans)

| setting | r1 (recorded) | r2 (planned) | |
|---|---|---|---|
| platform | `l4` | `l4` | same |
| provisioning | `standard` | `spot` | intended |
| weights | `gcs` | `gcs` | same |
| machine type | `g2-standard-8` | `g2-standard-8` | same |
| label | `main` | `rcx` | intended |
| config name | `l4_triton` | `l4_triton` | same |
| config description | `NVIDIA L4, Triton flash attention (the AlphaFold3 GPU default)` | `NVIDIA L4, Triton flash attention (the AlphaFold3 GPU default)` | same |
| config (harness/configs.yaml entry, without its description) | `{"env": {"JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}, "flash_attention_implementation": "triton", "jax_backend": "gpu"}` | `{"env": {"JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}, "flash_attention_implementation": "triton", "jax_backend": "gpu"}` | same |
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
| run_alphafold command | `cd third_party/alphafold3 && JAX_PLATFORMS=cuda XLA_FLAGS=--xla_gpu_enable_triton_gemm=false XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_CLIENT_MEM_FRACTION=0.95 JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=gpu --flash_attention_implementation=triton --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | `cd third_party/alphafold3 && JAX_PLATFORMS=cuda XLA_FLAGS=--xla_gpu_enable_triton_gemm=false XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_CLIENT_MEM_FRACTION=0.95 JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=gpu --flash_attention_implementation=triton --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | same |
| repetitions of the arm (r = fresh, w = warm rerun) | `r1` | `r2` | intended |
| seeds of the arm | `1 to 5` | `1 to 5` | same |
| image (as the wrapper log printed it for each session) | `ubuntu-os-accelerator-images/ubuntu-accelerator-2204-amd64-with-nvidia-580 (7 sessions)` | `af2-tpu-benchmark/af3-accel-2204-nvidia-580-gtr1 (set at launch)` | intended |
| Eigen source of the AlphaFold3 build | `git clone of gitlab.com/libeigen/eigen tag 3.4.0 (before follow-up 7)` | `archive 3.4.0 commit 3147391d946bb4b6c68edd901f2add6ac1f31f8c (plan key eigen_source: archive)` | intended |

Settings that differ beyond the allowed ones: 0
