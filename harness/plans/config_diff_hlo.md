# Follow-up 16: settings of each dump arm against the main-study arm it repeats (harness/plans/config_diff_hlo.py)

Source: as recorded in the main study's V5 and V6 sessions, with the Eigen source of their build. New: as the plans of harness/plans/hlo, harness/configs.yaml, harness/run_af3.py and cloud/af3_run.sh resolve them, on demand. Allowed differences: the label, the configuration (tpu_xla plus AF3_HLO_DUMP_FLAGS only, checked below), the repetitions and seeds, and the Eigen source (the archive of the same commit). Every other row must be the same.

## V5dump against V5 (4 main-study sessions, 1 plan)

| setting | source arm (recorded) | dump arm (planned) | |
|---|---|---|---|
| platform | `v5e` | `v5e` | same |
| provisioning | `standard` | `standard` | same |
| weights | `gcs` | `gcs` | same |
| machine type | `v5litepod-1` | `v5litepod-1` | same |
| image | `TPU runtime v2-alpha-tpuv5-lite` | `TPU runtime v2-alpha-tpuv5-lite` | same |
| libtpu | `0.0.43.2` | `0.0.43.2` | same |
| label | `main` | `hlo` | intended |
| config name | `tpu_xla` | `tpu_xla_hlo` | intended |
| config description | `Cloud TPU (v5e or v6e, single chip), XLA attention, af3_tpu.patch` | `Cloud TPU, XLA attention, af3_tpu.patch, HLO of the model dumped as text (follow-up 16)` | intended |
| config (harness/configs.yaml entry, without its description) | `{"flash_attention_implementation": "xla", "jax_backend": "tpu"}` | `{"env": {"AF3_HLO_DUMP_FLAGS": "--xla_dump_hlo_as_text --xla_dump_hlo_module_re=jit_apply_fn"}, "flash_attention_implementation": "xla", "jax_backend": "tpu"}` | intended (tpu_xla plus AF3_HLO_DUMP_FLAGS) |
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
| run_alphafold command | `cd third_party/alphafold3 && JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=tpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | `cd third_party/alphafold3 && AF3_HLO_DUMP_FLAGS=--xla_dump_hlo_as_text --xla_dump_hlo_module_re=jit_apply_fn JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=tpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | intended (AF3_HLO_DUMP_FLAGS added) |
| repetitions of the arm (r = fresh, w = warm rerun) | `r1, r2, w` | `r1, r2` | intended |
| seeds of the arm | `1 to 5` | `1 to 1` | intended |
| Eigen source of the AlphaFold3 build | `git clone of gitlab.com/libeigen/eigen tag 3.4.0 (before follow-up 7)` | `archive 3.4.0 commit 3147391d946bb4b6c68edd901f2add6ac1f31f8c (plan key eigen_source: archive)` | intended |

## V6dump against V6 (3 main-study sessions, 1 plan)

| setting | source arm (recorded) | dump arm (planned) | |
|---|---|---|---|
| platform | `v6e` | `v6e` | same |
| provisioning | `standard` | `standard` | same |
| weights | `gcs` | `gcs` | same |
| machine type | `ct6e-standard-1t` | `ct6e-standard-1t` | same |
| image | `ubuntu-os-accelerator-images/ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e` | `ubuntu-os-accelerator-images/ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e` | same |
| libtpu | `0.0.43.2` | `0.0.43.2` | same |
| label | `main` | `hlo` | intended |
| config name | `tpu_xla` | `tpu_xla_hlo` | intended |
| config description | `Cloud TPU (v5e or v6e, single chip), XLA attention, af3_tpu.patch` | `Cloud TPU, XLA attention, af3_tpu.patch, HLO of the model dumped as text (follow-up 16)` | intended |
| config (harness/configs.yaml entry, without its description) | `{"flash_attention_implementation": "xla", "jax_backend": "tpu"}` | `{"env": {"AF3_HLO_DUMP_FLAGS": "--xla_dump_hlo_as_text --xla_dump_hlo_module_re=jit_apply_fn"}, "flash_attention_implementation": "xla", "jax_backend": "tpu"}` | intended (tpu_xla plus AF3_HLO_DUMP_FLAGS) |
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
| run_alphafold command | `cd third_party/alphafold3 && JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=tpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | `cd third_party/alphafold3 && AF3_HLO_DUMP_FLAGS=--xla_dump_hlo_as_text --xla_dump_hlo_module_re=jit_apply_fn JAX_LOG_COMPILES=1 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none AF3_ENTRY_STATS=<target>/af3_entry_stats.json .venv/bin/python ../../harness/af3_entry.py --json_path=<target>/work/input_seeds.json --output_dir=<target>/af3_output '--model_dir=<model_dir>' --jax_backend=tpu --flash_attention_implementation=xla --run_data_pipeline=false --jax_compilation_cache_dir=<cache>` | intended (AF3_HLO_DUMP_FLAGS added) |
| repetitions of the arm (r = fresh, w = warm rerun) | `r1, r2, w` | `r1, r2` | intended |
| seeds of the arm | `1 to 5` | `1 to 1` | intended |
| Eigen source of the AlphaFold3 build | `git clone of gitlab.com/libeigen/eigen tag 3.4.0 (before follow-up 7)` | `archive 3.4.0 commit 3147391d946bb4b6c68edd901f2add6ac1f31f8c (plan key eigen_source: archive)` | intended |

Settings that differ beyond the allowed ones: 0
