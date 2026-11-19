# replication: Offline replication on SOPBench

| Field | Value |
|---|---|
| Started | 2026-11-19T23:50:18Z |
| Ended | 2026-11-19T23:52:30Z |
| Duration | 0h02m12s |
| Commit | 8fe2e08 (clean) |
| Command | `specdx p3 replication` |
| Exit code | 0 |
| Subject | - |
| Split | - |
| Verdict | DONE |

## Results

provisional: V3Params calibrated on CASE-0031, recalibrated for S3 in P3b

Hidden and contradiction findings were not refuted offline (refutation not attempted (offline)); contradictions are counted apart as unrefuted candidates, and the plain contradiction column, which needs a refutation, reads n/a offline.

## Table 1: findings by domain and rater composition

| domain | composition | raters | runs | excluded runs | θ min | θ median | converged | hidden | ambiguity | weak | contradiction | contradiction (unrefuted) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bank | all | 26 | 3484 | 76 | 0.270 | 0.861 | yes | 110 | 26 | 0 | n/a offline | 0 |
| bank | open | 11 | 1474 | 59 | 0.594 | 0.862 | yes | 157 | 57 | 0 | n/a offline | 0 |
| bank | hosted | 15 | 2010 | 17 | 0.521 | 0.910 | yes | 129 | 104 | 0 | n/a offline | 0 |
| dmv | all | 25 | 2425 | 50 | 0.365 | 0.799 | yes | 75 | 59 | 0 | n/a offline | 0 |
| dmv | open | 11 | 1067 | 28 | 0.628 | 0.871 | yes | 102 | 90 | 0 | n/a offline | 0 |
| dmv | hosted | 14 | 1358 | 22 | 0.596 | 0.869 | yes | 62 | 53 | 0 | n/a offline | 0 |
| healthcare | all | 27 | 3348 | 76 | 0.369 | 0.832 | yes | 74 | 36 | 0 | n/a offline | 0 |
| healthcare | open | 11 | 1364 | 58 | 0.546 | 0.772 | yes | 88 | 51 | 0 | n/a offline | 1 |
| healthcare | hosted | 16 | 1984 | 18 | 0.336 | 0.872 | yes | 51 | 38 | 0 | n/a offline | 0 |
| hotel | all | 25 | 4873 | 403 | 0.342 | 0.809 | yes | 233 | 128 | 0 | n/a offline | 0 |
| hotel | open | 10 | 1950 | 249 | 0.521 | 0.805 | yes | 288 | 234 | 0 | n/a offline | 0 |
| hotel | hosted | 15 | 2923 | 154 | 0.418 | 0.862 | yes | 319 | 240 | 0 | n/a offline | 0 |
| library | all | 26 | 1716 | 70 | 0.517 | 0.865 | yes | 146 | 111 | 0 | n/a offline | 0 |
| library | open | 11 | 726 | 34 | 0.691 | 0.869 | yes | 133 | 73 | 0 | n/a offline | 2 |
| library | hosted | 15 | 990 | 36 | 0.693 | 0.882 | yes | 136 | 109 | 0 | n/a offline | 0 |
| online_market | all | 26 | 4469 | 68 | 0.424 | 0.809 | yes | 219 | 152 | 0 | n/a offline | 0 |
| online_market | open | 11 | 1889 | 68 | 0.650 | 0.884 | yes | 438 | 364 | 0 | n/a offline | 2 |
| online_market | hosted | 15 | 2580 | 0 | 0.593 | 0.848 | yes | 184 | 132 | 0 | n/a offline | 0 |
| university | all | 24 | 1008 | 13 | 0.141 | 0.773 | yes | 39 | 20 | 0 | n/a offline | 0 |
| university | open | 10 | 420 | 13 | 0.275 | 0.757 | yes | 39 | 19 | 0 | n/a offline | 0 |
| university | hosted | 14 | 588 | 0 | 0.383 | 0.859 | yes | 41 | 38 | 0 | n/a offline | 0 |

## Table 2: the 10 highest-ranked findings per domain (all)

Ranked by score, then by the number of readings that deviate from the anchor.

| domain | type | location | score | deviating readings | user goal |
|---|---|---|---|---|---|
| bank | hidden | s3:bank:16:outcome | 1.000 | 26 | cancel_credit_card |
| bank | hidden | s3:bank:12:outcome | 1.000 | 25 | cancel_credit_card |
| bank | hidden | s3:bank:23:outcome | 1.000 | 24 | cancel_credit_card |
| bank | hidden | s3:bank:55:constraints | 1.000 | 24 | open_account |
| bank | hidden | s3:bank:55:db_match | 1.000 | 24 | open_account |
| bank | hidden | s3:bank:55:outcome | 1.000 | 24 | open_account |
| bank | hidden | s3:bank:55:process | 1.000 | 24 | open_account |
| bank | hidden | s3:bank:14:outcome | 1.000 | 23 | cancel_credit_card |
| bank | hidden | s3:bank:19:outcome | 1.000 | 23 | cancel_credit_card |
| bank | hidden | s3:bank:9:outcome | 1.000 | 23 | cancel_credit_card |
| dmv | hidden | s3:dmv:38:outcome | 1.000 | 22 | renew_vehicle |
| dmv | hidden | s3:dmv:36:outcome | 1.000 | 21 | renew_vehicle |
| dmv | hidden | s3:dmv:31:outcome | 1.000 | 20 | renew_dl |
| dmv | hidden | s3:dmv:35:constraints | 1.000 | 19 | renew_dl |
| dmv | hidden | s3:dmv:35:db_match | 1.000 | 19 | renew_dl |
| dmv | hidden | s3:dmv:45:constraints | 1.000 | 18 | renew_vehicle |
| dmv | hidden | s3:dmv:52:constraints | 1.000 | 18 | renew_vehicle |
| dmv | hidden | s3:dmv:45:db_match | 1.000 | 17 | renew_vehicle |
| dmv | hidden | s3:dmv:49:constraints | 1.000 | 17 | renew_vehicle |
| dmv | hidden | s3:dmv:52:db_match | 1.000 | 17 | renew_vehicle |
| healthcare | hidden | s3:healthcare:15:outcome | 1.000 | 17 | get_claim_details |
| healthcare | hidden | s3:healthcare:18:outcome | 1.000 | 17 | get_policy_details |
| healthcare | hidden | s3:healthcare:64:db_match | 1.000 | 19 | submit_claim |
| healthcare | hidden | s3:healthcare:11:outcome | 1.000 | 16 | appeal_claim |
| healthcare | hidden | s3:healthcare:11:db_match | 1.000 | 17 | appeal_claim |
| healthcare | hidden | s3:healthcare:71:db_match | 1.000 | 17 | submit_claim |
| healthcare | hidden | s3:healthcare:7:outcome | 1.000 | 16 | appeal_claim |
| healthcare | hidden | s3:healthcare:7:db_match | 1.000 | 17 | appeal_claim |
| healthcare | hidden | s3:healthcare:92:db_match | 1.000 | 16 | update_policy |
| healthcare | hidden | s3:healthcare:108:db_match | 1.000 | 15 | update_policy |
| hotel | hidden | s3:hotel:163:outcome | 1.000 | 25 | process_guest_checkin |
| hotel | hidden | s3:hotel:165:constraints | 1.000 | 25 | process_guest_checkin |
| hotel | hidden | s3:hotel:165:outcome | 1.000 | 25 | process_guest_checkin |
| hotel | hidden | s3:hotel:167:constraints | 1.000 | 25 | process_guest_checkin |
| hotel | hidden | s3:hotel:167:outcome | 1.000 | 25 | process_guest_checkin |
| hotel | hidden | s3:hotel:194:outcome | 1.000 | 25 | show_available_rooms |
| hotel | hidden | s3:hotel:46:outcome | 1.000 | 25 | find_booking_info |
| hotel | hidden | s3:hotel:184:db_match | 1.000 | 21 | request_room_change |
| hotel | hidden | s3:hotel:184:outcome | 1.000 | 21 | request_room_change |
| hotel | hidden | s3:hotel:186:db_match | 1.000 | 20 | request_room_change |
| library | hidden | s3:library:13:constraints | 1.000 | 25 | borrow_book |
| library | hidden | s3:library:13:db_match | 1.000 | 25 | borrow_book |
| library | hidden | s3:library:13:outcome | 1.000 | 25 | borrow_book |
| library | hidden | s3:library:44:outcome | 1.000 | 25 | reserve_room |
| library | hidden | s3:library:12:process | 1.000 | 24 | borrow_book |
| library | hidden | s3:library:34:constraints | 1.000 | 24 | borrow_book |
| library | hidden | s3:library:35:constraints | 1.000 | 24 | borrow_book |
| library | hidden | s3:library:35:db_match | 1.000 | 24 | borrow_book |
| library | hidden | s3:library:35:outcome | 1.000 | 24 | borrow_book |
| library | hidden | s3:library:36:process | 1.000 | 24 | check_return_date |
| online_market | hidden | s3:online_market:7:constraints | 1.000 | 25 | add_review |
| online_market | hidden | s3:online_market:7:db_match | 1.000 | 25 | add_review |
| online_market | hidden | s3:online_market:7:outcome | 1.000 | 25 | add_review |
| online_market | hidden | s3:online_market:134:constraints | 1.000 | 24 | use_coupon |
| online_market | hidden | s3:online_market:134:db_match | 1.000 | 24 | use_coupon |
| online_market | hidden | s3:online_market:134:outcome | 1.000 | 24 | use_coupon |
| online_market | hidden | s3:online_market:140:constraints | 1.000 | 24 | use_coupon |
| online_market | hidden | s3:online_market:140:db_match | 1.000 | 24 | use_coupon |
| online_market | hidden | s3:online_market:140:outcome | 1.000 | 24 | use_coupon |
| online_market | hidden | s3:online_market:156:constraints | 1.000 | 24 | use_coupon |
| university | hidden | s3:university:22:db_match | 1.000 | 17 | drop_course |
| university | hidden | s3:university:22:outcome | 1.000 | 17 | drop_course |
| university | hidden | s3:university:16:db_match | 1.000 | 16 | declare_minor |
| university | hidden | s3:university:29:db_match | 1.000 | 16 | enroll_course |
| university | hidden | s3:university:32:db_match | 1.000 | 16 | enroll_course |
| university | hidden | s3:university:6:db_match | 1.000 | 14 | apply_financial_aid |
| university | hidden | s3:university:4:db_match | 1.000 | 13 | apply_financial_aid |
| university | hidden | s3:university:25:db_match | 1.000 | 13 | enroll_course |
| university | hidden | s3:university:1:db_match | 1.000 | 12 | apply_financial_aid |
| university | hidden | s3:university:26:db_match | 1.000 | 13 | enroll_course |

## Table 3: saturation and distinct tasks (findings at score ≥0.999)

A task has up to four cells, and an ambiguity on an anchored cell also brings its hidden pick, so the distinct tasks per type are the count to read.

| domain | composition | findings | ≥0.999 | distinct tasks: hidden | distinct tasks: ambiguity | distinct tasks: weak | distinct tasks: contradiction (unrefuted) |
|---|---|---|---|---|---|---|---|
| bank | all | 136 | 105 | 57 | 25 | 0 | 0 |
| bank | open | 214 | 83 | 79 | 50 | 0 | 0 |
| bank | hosted | 233 | 100 | 62 | 58 | 0 | 0 |
| dmv | all | 134 | 109 | 45 | 35 | 0 | 0 |
| dmv | open | 192 | 69 | 56 | 47 | 0 | 0 |
| dmv | hosted | 115 | 38 | 37 | 32 | 0 | 0 |
| healthcare | all | 110 | 78 | 43 | 34 | 0 | 0 |
| healthcare | open | 140 | 52 | 59 | 48 | 0 | 1 |
| healthcare | hosted | 89 | 24 | 31 | 25 | 0 | 0 |
| hotel | all | 361 | 315 | 186 | 122 | 0 | 0 |
| hotel | open | 522 | 209 | 136 | 106 | 0 | 0 |
| hotel | hosted | 559 | 281 | 178 | 122 | 0 | 0 |
| library | all | 257 | 229 | 59 | 51 | 0 | 0 |
| library | open | 208 | 123 | 57 | 38 | 0 | 1 |
| library | hosted | 245 | 111 | 58 | 54 | 0 | 0 |
| online_market | all | 371 | 314 | 99 | 71 | 0 | 0 |
| online_market | open | 804 | 369 | 160 | 138 | 0 | 1 |
| online_market | hosted | 316 | 124 | 81 | 61 | 0 | 0 |
| university | all | 59 | 48 | 24 | 18 | 0 | 0 |
| university | open | 58 | 20 | 23 | 17 | 0 | 0 |
| university | hosted | 79 | 26 | 21 | 20 | 0 | 0 |

## Loading

| domain | files | loaded | excluded | records | runs | missing evaluations |
|---|---|---|---|---|---|---|
| bank | 27 | 26 | 1 | 3618 | 3484 | 134 |
| dmv | 26 | 25 | 1 | 2522 | 2425 | 97 |
| healthcare | 28 | 27 | 1 | 3472 | 3348 | 124 |
| hotel | 26 | 25 | 1 | 5070 | 4873 | 197 |
| library | 27 | 26 | 1 | 1782 | 1716 | 66 |
| online_market | 27 | 26 | 1 | 4644 | 4469 | 175 |
| university | 26 | 24 | 2 | 1092 | 1008 | 84 |

## Excluded files

- `output/bank/ast_gpt-5-mini-mode_fc-dep_full-fmt_structured-tool_full-shuffle_False.json`: no evaluations (unevaluated upstream)
- `output/dmv/ast_gpt-5-mini-mode_fc-dep_full-fmt_structured-tool_full-shuffle_False.json`: no evaluations (unevaluated upstream)
- `output/healthcare/ast_gpt-5-mini-mode_fc-dep_full-fmt_structured-tool_full-shuffle_False.json`: no evaluations (unevaluated upstream)
- `output/hotel/ast_qwen2.5-72b-instruct-mode_react-dep_full-fmt_structured-tool_full-shuffle_False.json`: no evaluations (unevaluated upstream)
- `output/library/ast_gpt-4.1-mini-mode_fc-dep_full-fmt_structured-tool_full-shuffle_False.json`: no evaluations (unevaluated upstream)
- `output/online_market/ast_gpt-5-mini-mode_fc-dep_full-fmt_structured-tool_full-shuffle_False.json`: no evaluations (unevaluated upstream)
- `output/university/ast_gpt-5-mini-mode_fc-dep_full-fmt_structured-tool_full-shuffle_False.json`: no evaluations (unevaluated upstream)
- `output/university/ast_qwen2.5-72b-instruct-mode_react-dep_full-fmt_structured-tool_full-shuffle_False.json`: no evaluations (unevaluated upstream)

## Prompt variants

Within each task the most common rendered prompt is variant 0, and every other distinct prompt is variant 1, 2, … by decreasing frequency, decided per run (published files mix renders). T is built from variant 0. The variant is confounded with model generation: the earlier models saw an older render, so offline a variant effect cannot be separated from a rater's accuracy.

- bank: variant 0: 2303 runs; other variants: 1181 runs (by variant 1: 1181; models: claude-3-5-sonnet-20241022 134, deepseek-r1 111, gemini-1.5-pro 134, gemini-2.0-flash-001 134, gemini-2.0-flash-thinking-exp 132, gpt-4o 134, gpt-4o-mini 134, llama3.1-70b-instruct 134, qwen2.5-72b-instruct 134)
- dmv: variant 0: 1649 runs; other variants: 776 runs (by variant 1: 776; models: claude-3-5-sonnet-20241022 97, gemini-1.5-pro 97, gemini-2.0-flash-001 97, gemini-2.0-flash-thinking-exp 97, gpt-4o 97, gpt-4o-mini 97, llama3.1-70b-instruct 97, qwen2.5-72b-instruct 97)
- healthcare: variant 0: 2035 runs; other variants: 1313 runs (by variant 1: 1238, 2: 75; models: claude-3-5-sonnet-20241022 124, deepseek-r1 122, gemini-1.5-pro 124, gemini-2.0-flash-001 124, gemini-2.0-flash-thinking-exp 124, gpt-4o 124, gpt-4o-mini 124, llama3.1-70b-instruct 124, o1 124, o4-mini 75, qwen2.5-72b-instruct 124)
- hotel: variant 0: 4873 runs; other variants: 0 runs
- library: variant 0: 1184 runs; other variants: 532 runs (by variant 1: 528, 2: 4; models: claude-3-5-sonnet-20241022 66, gemini-1.5-pro 66, gemini-2.0-flash-001 66, gemini-2.0-flash-thinking-exp 66, gpt-4o 66, gpt-4o-mini 66, llama3.1-70b-instruct 66, o4-mini 4, qwen2.5-72b-instruct 66)
- online_market: variant 0: 3093 runs; other variants: 1376 runs (by variant 1: 1376; models: claude-3-5-sonnet-20241022 172, gemini-1.5-pro 172, gemini-2.0-flash-001 172, gemini-2.0-flash-thinking-exp 172, gpt-4o 172, gpt-4o-mini 172, llama3.1-70b-instruct 172, qwen2.5-72b-instruct 172)
- university: variant 0: 1008 runs; other variants: 0 runs
