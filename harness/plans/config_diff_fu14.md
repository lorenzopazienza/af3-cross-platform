# Follow-up 14: settings of each new arm against its source arm (harness/plans/config_diff_fu14.py)

Source arm: as recorded in its sessions (V5: main study; A: follow-up 3; H: follow-up 9, r1). New arm: as its plans, harness/configs.yaml, harness/run_af3.py and cloud/af3_run.sh resolve them (A100 machine type a2-ultragpu-1g, set at launch by A100_MACHINE_TYPE), with the provisioning of item f (on demand on v5e, Spot on the A100 and the H100). Allowed differences: the label, the follow-up 14 settings (configuration name and description; AF3_FU13_BFLOAT16=none and JAX_DEFAULT_MATMUL_PRECISION=highest for the float32 arms; --xla_gpu_deterministic_ops=true in XLA_FLAGS for the deterministic arms), the seeds, the repetitions and the provisioning. Rows marked "without the follow-up 14 settings" compare the configuration, the environment and the command again with those settings removed, and must be the same.

## V5f against V5 (4 source sessions, 9 plans): float32 trunk

Follow-up 14 settings of V5f: `AF3_FU13_BFLOAT16=none`, `JAX_DEFAULT_MATMUL_PRECISION=highest`

| setting | source arm (recorded) | new arm (planned) | |
|---|---|---|---|
| platform | `v5e` | `v5e` | same |
| provisioning | `standard` | `standard` | same |
| weights | `gcs` | `gcs` | same |
| machine type | `v5litepod-1` | `v5litepod-1` | same |
| image | `TPU runtime v2-alpha-tpuv5-lite` | `TPU runtime v2-alpha-tpuv5-lite` | same |
| libtpu | `0.0.43.2` | `0.0.43.2` | same |
| label | `main` | `f32` | intended |
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
| config (harness/configs.yaml entry, without its description), without the follow-up 14 settings | `{"flash_attention_implementation": "xla", "jax_backend": "tpu"}` | `{"flash_attention_implementation": "xla", "jax_backend": "tpu"}` | same |
| environment (xla_env), without the follow-up 14 settings | `{"JAX_LOG_COMPILES": "1", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none"}` | `{"JAX_LOG_COMPILES": "1", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none"}` | same |
| run_alphafold command, without the follow-up 14 settings | `cd third_party/alphafold3 && JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=tpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | `cd third_party/alphafold3 && JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=tpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | same |

## Af against A (7 source sessions, 15 plans): float32 trunk

Follow-up 14 settings of Af: `AF3_FU13_BFLOAT16=none`, `JAX_DEFAULT_MATMUL_PRECISION=highest`

| setting | source arm (recorded) | new arm (planned) | |
|---|---|---|---|
| platform | `a100` | `a100` | same |
| provisioning | `standard` | `spot` | intended |
| weights | `gcs` | `gcs` | same |
| machine type | `a2-ultragpu-1g` | `a2-ultragpu-1g` | same |
| image | `ubuntu-os-accelerator-images/ubuntu-accelerator-2204-amd64-with-nvidia-580` | `ubuntu-os-accelerator-images/ubuntu-accelerator-2204-amd64-with-nvidia-580` | same |
| label | `a100` | `f32` | intended |
| config name | `a100_xla` | `a100_xla_fu14_f32` | intended |
| config description | `NVIDIA A100 80 GB (compute capability 8.0), XLA attention (A arm)` | `NVIDIA A100 80 GB, XLA attention, float32 trunk (bfloat16 none, matmuls highest; follow-up 14, Af)` | intended |
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
| repetitions of the arm (r = fresh, w = warm rerun) | `r1, r2, w` | `r1` | intended |
| seeds of the arm | `1 to 5` | `1 to 5` | same |
| config (harness/configs.yaml entry, without its description), without the follow-up 14 settings | `{"env": {"JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}, "flash_attention_implementation": "xla", "jax_backend": "gpu"}` | `{"env": {"JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}, "flash_attention_implementation": "xla", "jax_backend": "gpu"}` | same |
| environment (xla_env), without the follow-up 14 settings | `{"JAX_LOG_COMPILES": "1", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none", "JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}` | `{"JAX_LOG_COMPILES": "1", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none", "JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}` | same |
| run_alphafold command, without the follow-up 14 settings | `cd third_party/alphafold3 && JAX_PLATFORMS=cuda XLA_FLAGS=--xla_gpu_enable_triton_gemm=false XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_CLIENT_MEM_FRACTION=0.95 JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=gpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | `cd third_party/alphafold3 && JAX_PLATFORMS=cuda XLA_FLAGS=--xla_gpu_enable_triton_gemm=false XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_CLIENT_MEM_FRACTION=0.95 JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=gpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | same |

## Hf against H (3 source sessions, 10 plans): float32 trunk

Follow-up 14 settings of Hf: `AF3_FU13_BFLOAT16=none`, `JAX_DEFAULT_MATMUL_PRECISION=highest`

| setting | source arm (recorded) | new arm (planned) | |
|---|---|---|---|
| platform | `h100` | `h100` | same |
| provisioning | `spot` | `spot` | same |
| weights | `gcs` | `gcs` | same |
| machine type | `a3-highgpu-1g` | `a3-highgpu-1g` | same |
| image | `ubuntu-os-accelerator-images/ubuntu-accelerator-2204-amd64-with-nvidia-580` | `ubuntu-os-accelerator-images/ubuntu-accelerator-2204-amd64-with-nvidia-580` | same |
| label | `h100` | `f32` | intended |
| config name | `h100_xla` | `h100_xla_fu14_f32` | intended |
| config description | `NVIDIA H100 80 GB (compute capability 9.0), XLA attention (H arm)` | `NVIDIA H100 80 GB, XLA attention, float32 trunk (bfloat16 none, matmuls highest; follow-up 14, Hf)` | intended |
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
| repetitions of the arm (r = fresh, w = warm rerun) | `r1` | `r1` | same |
| seeds of the arm | `1 to 5` | `1 to 5` | same |
| config (harness/configs.yaml entry, without its description), without the follow-up 14 settings | `{"env": {"JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}, "flash_attention_implementation": "xla", "jax_backend": "gpu"}` | `{"env": {"JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}, "flash_attention_implementation": "xla", "jax_backend": "gpu"}` | same |
| environment (xla_env), without the follow-up 14 settings | `{"JAX_LOG_COMPILES": "1", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none", "JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}` | `{"JAX_LOG_COMPILES": "1", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none", "JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}` | same |
| run_alphafold command, without the follow-up 14 settings | `cd third_party/alphafold3 && JAX_PLATFORMS=cuda XLA_FLAGS=--xla_gpu_enable_triton_gemm=false XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_CLIENT_MEM_FRACTION=0.95 JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=gpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | `cd third_party/alphafold3 && JAX_PLATFORMS=cuda XLA_FLAGS=--xla_gpu_enable_triton_gemm=false XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_CLIENT_MEM_FRACTION=0.95 JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=gpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | same |

## Ad against A (7 source sessions, 5 plans): deterministic ops

Follow-up 14 settings of Ad: `--xla_gpu_deterministic_ops=true`

| setting | source arm (recorded) | new arm (planned) | |
|---|---|---|---|
| platform | `a100` | `a100` | same |
| provisioning | `standard` | `spot` | intended |
| weights | `gcs` | `gcs` | same |
| machine type | `a2-ultragpu-1g` | `a2-ultragpu-1g` | same |
| image | `ubuntu-os-accelerator-images/ubuntu-accelerator-2204-amd64-with-nvidia-580` | `ubuntu-os-accelerator-images/ubuntu-accelerator-2204-amd64-with-nvidia-580` | same |
| label | `a100` | `f32` | intended |
| config name | `a100_xla` | `a100_xla_det` | intended |
| config description | `NVIDIA A100 80 GB (compute capability 8.0), XLA attention (A arm)` | `NVIDIA A100 80 GB, XLA attention, --xla_gpu_deterministic_ops=true (follow-up 14, Ad)` | intended |
| config (harness/configs.yaml entry, without its description) | `{"env": {"JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}, "flash_attention_implementation": "xla", "jax_backend": "gpu"}` | `{"env": {"JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false --xla_gpu_deterministic_ops=true", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}, "flash_attention_implementation": "xla", "jax_backend": "gpu"}` | intended |
| environment (xla_env) | `{"JAX_LOG_COMPILES": "1", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none", "JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}` | `{"JAX_LOG_COMPILES": "1", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none", "JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false --xla_gpu_deterministic_ops=true", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}` | intended |
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
| run_alphafold command | `cd third_party/alphafold3 && JAX_PLATFORMS=cuda XLA_FLAGS=--xla_gpu_enable_triton_gemm=false XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_CLIENT_MEM_FRACTION=0.95 JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=gpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | `cd third_party/alphafold3 && JAX_PLATFORMS=cuda XLA_FLAGS=--xla_gpu_enable_triton_gemm=false --xla_gpu_deterministic_ops=true XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_CLIENT_MEM_FRACTION=0.95 JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=gpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | intended |
| repetitions of the arm (r = fresh, w = warm rerun) | `r1, r2, w` | `r1, r2` | intended |
| seeds of the arm | `1 to 5` | `1 to 5` | same |
| config (harness/configs.yaml entry, without its description), without the follow-up 14 settings | `{"env": {"JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}, "flash_attention_implementation": "xla", "jax_backend": "gpu"}` | `{"env": {"JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}, "flash_attention_implementation": "xla", "jax_backend": "gpu"}` | same |
| environment (xla_env), without the follow-up 14 settings | `{"JAX_LOG_COMPILES": "1", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none", "JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}` | `{"JAX_LOG_COMPILES": "1", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none", "JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}` | same |
| run_alphafold command, without the follow-up 14 settings | `cd third_party/alphafold3 && JAX_PLATFORMS=cuda XLA_FLAGS=--xla_gpu_enable_triton_gemm=false XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_CLIENT_MEM_FRACTION=0.95 JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=gpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | `cd third_party/alphafold3 && JAX_PLATFORMS=cuda XLA_FLAGS=--xla_gpu_enable_triton_gemm=false XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_CLIENT_MEM_FRACTION=0.95 JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=gpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | same |

## Hd against H (3 source sessions, 5 plans): deterministic ops

Follow-up 14 settings of Hd: `--xla_gpu_deterministic_ops=true`

| setting | source arm (recorded) | new arm (planned) | |
|---|---|---|---|
| platform | `h100` | `h100` | same |
| provisioning | `spot` | `spot` | same |
| weights | `gcs` | `gcs` | same |
| machine type | `a3-highgpu-1g` | `a3-highgpu-1g` | same |
| image | `ubuntu-os-accelerator-images/ubuntu-accelerator-2204-amd64-with-nvidia-580` | `ubuntu-os-accelerator-images/ubuntu-accelerator-2204-amd64-with-nvidia-580` | same |
| label | `h100` | `f32` | intended |
| config name | `h100_xla` | `h100_xla_det` | intended |
| config description | `NVIDIA H100 80 GB (compute capability 9.0), XLA attention (H arm)` | `NVIDIA H100 80 GB, XLA attention, --xla_gpu_deterministic_ops=true (follow-up 14, Hd)` | intended |
| config (harness/configs.yaml entry, without its description) | `{"env": {"JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}, "flash_attention_implementation": "xla", "jax_backend": "gpu"}` | `{"env": {"JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false --xla_gpu_deterministic_ops=true", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}, "flash_attention_implementation": "xla", "jax_backend": "gpu"}` | intended |
| environment (xla_env) | `{"JAX_LOG_COMPILES": "1", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none", "JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}` | `{"JAX_LOG_COMPILES": "1", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none", "JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false --xla_gpu_deterministic_ops=true", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}` | intended |
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
| run_alphafold command | `cd third_party/alphafold3 && JAX_PLATFORMS=cuda XLA_FLAGS=--xla_gpu_enable_triton_gemm=false XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_CLIENT_MEM_FRACTION=0.95 JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=gpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | `cd third_party/alphafold3 && JAX_PLATFORMS=cuda XLA_FLAGS=--xla_gpu_enable_triton_gemm=false --xla_gpu_deterministic_ops=true XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_CLIENT_MEM_FRACTION=0.95 JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=gpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | intended |
| repetitions of the arm (r = fresh, w = warm rerun) | `r1` | `r1, r2` | intended |
| seeds of the arm | `1 to 5` | `1 to 5` | same |
| config (harness/configs.yaml entry, without its description), without the follow-up 14 settings | `{"env": {"JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}, "flash_attention_implementation": "xla", "jax_backend": "gpu"}` | `{"env": {"JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}, "flash_attention_implementation": "xla", "jax_backend": "gpu"}` | same |
| environment (xla_env), without the follow-up 14 settings | `{"JAX_LOG_COMPILES": "1", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none", "JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}` | `{"JAX_LOG_COMPILES": "1", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none", "JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}` | same |
| run_alphafold command, without the follow-up 14 settings | `cd third_party/alphafold3 && JAX_PLATFORMS=cuda XLA_FLAGS=--xla_gpu_enable_triton_gemm=false XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_CLIENT_MEM_FRACTION=0.95 JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=gpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | `cd third_party/alphafold3 && JAX_PLATFORMS=cuda XLA_FLAGS=--xla_gpu_enable_triton_gemm=false XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_CLIENT_MEM_FRACTION=0.95 JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=gpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | same |

Settings that differ beyond the allowed ones: 0
