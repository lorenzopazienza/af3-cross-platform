# Follow-ups 7 to 9: settings of each new arm against its source arm (harness/plans/config_diff_fu7_10.py)

Source arm: as recorded in its sessions. New arm: as its plans, harness/configs.yaml, harness/run_af3.py and cloud/af3_run.sh resolve them, with the planned provisioning (Spot on v5e, L4 and H100; on demand on v6e). Intended differences: the factor under study, the label, the repetitions and, where it is Spot, the provisioning (how the VM is obtained, not the computation).

## V5x against V5 (4 source sessions, 5 plans)

| setting | source arm (recorded) | new arm (planned) | |
|---|---|---|---|
| platform | `v5e` | `v5e` | same |
| provisioning | `standard` | `spot` | intended |
| weights | `gcs` | `gcs` | same |
| machine type | `v5litepod-1` | `v5litepod-1` | same |
| image | `TPU runtime v2-alpha-tpuv5-lite` | `TPU runtime v2-alpha-tpuv5-lite` | same |
| libtpu | `0.0.43.2` | `0.0.43.2` | same |
| label | `main` | `seeds` | intended |
| config name | `tpu_xla` | `tpu_xla` | same |
| config description | `Cloud TPU (v5e or v6e, single chip), XLA attention, af3_tpu.patch` | `Cloud TPU (v5e or v6e, single chip), XLA attention, af3_tpu.patch` | same |
| config (harness/configs.yaml entry, without its description) | `{"flash_attention_implementation": "xla", "jax_backend": "tpu"}` | `{"flash_attention_implementation": "xla", "jax_backend": "tpu"}` | same |
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
| run_alphafold command | `cd third_party/alphafold3 && JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=tpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | `cd third_party/alphafold3 && JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=tpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | same |
| repetitions of the arm (r = fresh, w = warm rerun) | `r1, r2, w` | `r1` | intended |
| seeds of the arm | `1 to 5` | `6 to 20` | intended |

## Gx against G (14 source sessions, 22 plans)

| setting | source arm (recorded) | new arm (planned) | |
|---|---|---|---|
| platform | `l4` | `l4` | same |
| provisioning | `standard` | `spot` | intended |
| weights | `gcs` | `gcs` | same |
| machine type | `g2-standard-8` | `g2-standard-8` | same |
| image | `ubuntu-os-accelerator-images/ubuntu-accelerator-2204-amd64-with-nvidia-580` | `ubuntu-os-accelerator-images/ubuntu-accelerator-2204-amd64-with-nvidia-580` | same |
| label | `main` | `seeds` | intended |
| config name | `l4_xla` | `l4_xla` | same |
| config description | `NVIDIA L4 (compute capability 8.9), XLA attention` | `NVIDIA L4 (compute capability 8.9), XLA attention` | same |
| config (harness/configs.yaml entry, without its description) | `{"env": {"JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}, "flash_attention_implementation": "xla", "jax_backend": "gpu"}` | `{"env": {"JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}, "flash_attention_implementation": "xla", "jax_backend": "gpu"}` | same |
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
| run_alphafold command | `cd third_party/alphafold3 && JAX_PLATFORMS=cuda XLA_FLAGS=--xla_gpu_enable_triton_gemm=false XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_CLIENT_MEM_FRACTION=0.95 JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=gpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | `cd third_party/alphafold3 && JAX_PLATFORMS=cuda XLA_FLAGS=--xla_gpu_enable_triton_gemm=false XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_CLIENT_MEM_FRACTION=0.95 JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=gpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | same |
| repetitions of the arm (r = fresh, w = warm rerun) | `r1, r2, w` | `r1` | intended |
| seeds of the arm | `1 to 5` | `6 to 20` | intended |

## V6s against V6 (3 source sessions, 2 plans)

| setting | source arm (recorded) | new arm (planned) | |
|---|---|---|---|
| platform | `v6e` | `v6e` | same |
| provisioning | `standard` | `standard` | same |
| weights | `gcs` | `gcs` | same |
| machine type | `ct6e-standard-1t` | `ct6e-standard-1t` | same |
| image | `ubuntu-os-accelerator-images/ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e` | `ubuntu-os-accelerator-images/ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e` | same |
| libtpu | `0.0.43.2` | `0.0.41` | intended |
| label | `main` | `sw` | intended |
| config name | `tpu_xla` | `tpu_xla` | same |
| config description | `Cloud TPU (v5e or v6e, single chip), XLA attention, af3_tpu.patch` | `Cloud TPU (v5e or v6e, single chip), XLA attention, af3_tpu.patch` | same |
| config (harness/configs.yaml entry, without its description) | `{"flash_attention_implementation": "xla", "jax_backend": "tpu"}` | `{"flash_attention_implementation": "xla", "jax_backend": "tpu"}` | same |
| environment (xla_env) | `{"JAX_LOG_COMPILES": "1", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none"}` | `{"JAX_LOG_COMPILES": "1", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none"}` | same |
| software stack (jax, jaxlib, libtpu or jax-cuda12 plugin and pjrt) | `{"jax": "0.10.2", "jaxlib": "0.10.2", "libtpu": "0.0.43.2"}` | `{"jax": "0.10.1", "jaxlib": "0.10.1", "libtpu": "0.0.41"}` | intended |
| AlphaFold3 commit | `85c4d20505fd5cef05eac22b534d4e793971ae69` | `85c4d20505fd5cef05eac22b534d4e793971ae69` | same |
| af3_tpu.patch SHA-256 | `e4bb8c326137ebb4903aad98d72592e8b6858c82dc6f4c1517848119e0520a4b` | `e4bb8c326137ebb4903aad98d72592e8b6858c82dc6f4c1517848119e0520a4b` | same |
| run_alphafold.py SHA-256 | `c62bfaca9476c57c519429abc7f57d8db05165487165cbb463308e98b5f7ace4` | `c62bfaca9476c57c519429abc7f57d8db05165487165cbb463308e98b5f7ace4` | same |
| buckets | `null` | `null` | same |
| save_embeddings | `false` | `false` | same |
| recycles, diffusion samples | `[null, null]` | `[null, null]` | same |
| per-process timeout (min), entry, log_compiles, order | `[180.0, true, true, "tokens"]` | `[180.0, true, true, "tokens"]` | same |
| input manifest | `inputs/manifest.csv` | `inputs/manifest.csv` | same |
| seeds per process | `1` | `1` | same |
| libtpu build | `libtpu_lts_20260630_b_RC06` | `the build of libtpu 0.0.41` | intended |
| run_alphafold command | `cd third_party/alphafold3 && JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=tpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | `cd third_party/alphafold3 && JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=tpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | same |
| repetitions of the arm (r = fresh, w = warm rerun) | `r1, r2, w` | `r1` | intended |
| seeds of the arm | `1 to 5` | `1 to 5` | same |

## Gs against G (14 source sessions, 7 plans)

| setting | source arm (recorded) | new arm (planned) | |
|---|---|---|---|
| platform | `l4` | `l4` | same |
| provisioning | `standard` | `spot` | intended |
| weights | `gcs` | `gcs` | same |
| machine type | `g2-standard-8` | `g2-standard-8` | same |
| image | `ubuntu-os-accelerator-images/ubuntu-accelerator-2204-amd64-with-nvidia-580` | `ubuntu-os-accelerator-images/ubuntu-accelerator-2204-amd64-with-nvidia-580` | same |
| label | `main` | `sw` | intended |
| config name | `l4_xla` | `l4_xla` | same |
| config description | `NVIDIA L4 (compute capability 8.9), XLA attention` | `NVIDIA L4 (compute capability 8.9), XLA attention` | same |
| config (harness/configs.yaml entry, without its description) | `{"env": {"JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}, "flash_attention_implementation": "xla", "jax_backend": "gpu"}` | `{"env": {"JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}, "flash_attention_implementation": "xla", "jax_backend": "gpu"}` | same |
| environment (xla_env) | `{"JAX_LOG_COMPILES": "1", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none", "JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}` | `{"JAX_LOG_COMPILES": "1", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none", "JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}` | same |
| software stack (jax, jaxlib, libtpu or jax-cuda12 plugin and pjrt) | `{"jax": "0.10.2", "jax-cuda12-pjrt": "0.10.2", "jax-cuda12-plugin": "0.10.2", "jaxlib": "0.10.2"}` | `{"jax": "0.10.1", "jax-cuda12-pjrt": "0.10.1", "jax-cuda12-plugin": "0.10.1", "jaxlib": "0.10.1"}` | intended |
| AlphaFold3 commit | `85c4d20505fd5cef05eac22b534d4e793971ae69` | `85c4d20505fd5cef05eac22b534d4e793971ae69` | same |
| af3_tpu.patch SHA-256 | `e4bb8c326137ebb4903aad98d72592e8b6858c82dc6f4c1517848119e0520a4b` | `e4bb8c326137ebb4903aad98d72592e8b6858c82dc6f4c1517848119e0520a4b` | same |
| run_alphafold.py SHA-256 | `c62bfaca9476c57c519429abc7f57d8db05165487165cbb463308e98b5f7ace4` | `c62bfaca9476c57c519429abc7f57d8db05165487165cbb463308e98b5f7ace4` | same |
| buckets | `null` | `null` | same |
| save_embeddings | `false` | `false` | same |
| recycles, diffusion samples | `[null, null]` | `[null, null]` | same |
| per-process timeout (min), entry, log_compiles, order | `[180.0, true, true, "tokens"]` | `[180.0, true, true, "tokens"]` | same |
| input manifest | `inputs/manifest.csv` | `inputs/manifest.csv` | same |
| seeds per process | `1` | `1` | same |
| run_alphafold command | `cd third_party/alphafold3 && JAX_PLATFORMS=cuda XLA_FLAGS=--xla_gpu_enable_triton_gemm=false XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_CLIENT_MEM_FRACTION=0.95 JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=gpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | `cd third_party/alphafold3 && JAX_PLATFORMS=cuda XLA_FLAGS=--xla_gpu_enable_triton_gemm=false XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_CLIENT_MEM_FRACTION=0.95 JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=gpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | same |
| repetitions of the arm (r = fresh, w = warm rerun) | `r1, r2, w` | `r1` | intended |
| seeds of the arm | `1 to 5` | `1 to 5` | same |

## H against A (7 source sessions, 4 plans)

| setting | source arm (recorded) | new arm (planned) | |
|---|---|---|---|
| platform | `a100` | `h100` | intended |
| provisioning | `standard` | `spot` | intended |
| weights | `gcs` | `gcs` | same |
| machine type | `a2-ultragpu-1g` | `a3-highgpu-1g` | intended |
| image | `ubuntu-os-accelerator-images/ubuntu-accelerator-2204-amd64-with-nvidia-580` | `ubuntu-os-accelerator-images/ubuntu-accelerator-2204-amd64-with-nvidia-580` | same |
| label | `a100` | `h100` | intended |
| config name | `a100_xla` | `h100_xla` | intended |
| config description | `NVIDIA A100 80 GB (compute capability 8.0), XLA attention (A arm)` | `NVIDIA H100 80 GB (compute capability 9.0), XLA attention (H arm)` | intended |
| config (harness/configs.yaml entry, without its description) | `{"env": {"JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}, "flash_attention_implementation": "xla", "jax_backend": "gpu"}` | `{"env": {"JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": "0.95", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}, "flash_attention_implementation": "xla", "jax_backend": "gpu"}` | same |
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
| run_alphafold command | `cd third_party/alphafold3 && JAX_PLATFORMS=cuda XLA_FLAGS=--xla_gpu_enable_triton_gemm=false XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_CLIENT_MEM_FRACTION=0.95 JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=gpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | `cd third_party/alphafold3 && JAX_PLATFORMS=cuda XLA_FLAGS=--xla_gpu_enable_triton_gemm=false XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_CLIENT_MEM_FRACTION=0.95 JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=gpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | same |
| repetitions of the arm (r = fresh, w = warm rerun) | `r1, r2, w` | `r1, r2` | intended |
| seeds of the arm | `1 to 5` | `1 to 5` | same |

Settings that differ beyond the factor under study: 0
