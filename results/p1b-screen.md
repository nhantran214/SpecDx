# P1b capability screen (gpu16)

Base specification, scenario `happy`, Tell variants [0, 1, 2] x seeds [1, 2, 3]. Admission: act rate ≥ 0.6 and valid-call rate ≥ 0.8. Population: at most 2 per family and 4 in all, at least 2 families.

| model | family | runs | act rate | valid-call rate | admitted | chosen | note |
|---|---|---:|---:|---:|---|---|---|
| qwen3:14b | qwen | 9 | 0.407 | 1.000 | no |  |  |
| qwen2.5:14b | qwen | 9 | 0.580 | 0.943 | no |  |  |
| qwen3:30b-a3b | qwen | 9 | 0.802 | 0.952 | yes | yes |  |
| gpt-oss:20b | openai | 9 | 0.914 | 0.988 | yes | yes |  |
| mistral-small3.2:24b | mistral | 9 | 0.346 | 0.914 | no |  |  |
| granite3.3:8b | ibm | 9 | 0.025 | 1.000 | no |  |  |
| llama3.1:8b | meta | 9 | 0.148 | 0.828 | no |  |  |

## Population

- openai: gpt-oss:20b
- qwen: qwen3:30b-a3b
