# Upstream Reference Clones

This directory is for local read-only reference clones used during source checking. The clones themselves are not committed to the POLARIS repository because they are large nested Git repositories.

Current observed references:

| Name | Commit | Remote |
| --- | --- | --- |
| `ace` | `4f679bef3b78e973a0e13a0acc2b4a7f6f7e41a2` | `https://github.com/ace-agent/ace.git` |
| `dynamic-cheatsheet` | `5cfe3c37e8e52b1d858d0f3df46e7f17c50991b9` | `https://github.com/suzgunmirac/dynamic-cheatsheet.git` |
| `evalplus` | `26d6d00bb1fd0fa37f39c99d5290da67891d1c5e` | `https://github.com/evalplus/evalplus.git` |
| `gepa` | `ce51b50cd196b539c25fae99ad0e0255c23004a4` | `https://github.com/gepa-ai/gepa.git` |
| `reasoning-with-sampling` | `720a8e9d084c87a630595e316f5260f1d7c3446c` | `https://github.com/aakaran/reasoning-with-sampling.git` |
| `search-and-learn` | `547502cc5e91925a6d3b57f4705aa4da35425f86` | `https://github.com/huggingface/search-and-learn.git` |
| `sglang-v0.4.7` | `4f723edd3baf3823eddfb9d6426548daba17c687` | `https://github.com/sgl-project/sglang.git` |
| `verifiers` | `c7731bbb6615de59c594e5b1c872cb1ef514c974` | `https://github.com/PrimeIntellect-ai/verifiers.git` |
| `vllm-v0.9.2` | `a5dd03c1ebc5e4f56f3c9d3dc0436e9c582c978f` | `https://github.com/vllm-project/vllm.git` |

Runtime-adapted copies live under `src/polaris/vendored/`.
