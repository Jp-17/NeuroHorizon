# Decoder Scheduled Sampling 250ms Screening Summary

Generated: 2026-03-22T17:34:39+08:00
Complete settings: 7/7

## Ranking by rollout test fp_bps

| rank | setting | rollout test | rollout valid | tf test | test gap | tf valid | valid gap | checkpoint epoch |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `decoder_ss_linear_0_to_050` | 0.2245 | 0.2210 | 0.2831 | 0.0586 | 0.2826 | 0.0616 | 159 |
| 2 | `decoder_ss_fixed_025` | 0.2230 | 0.2189 | 0.2867 | 0.0638 | 0.2871 | 0.0682 | 159 |
| 3 | `hybrid_mix035_plus_linear_050` | 0.2227 | 0.2210 | 0.2714 | 0.0487 | 0.2746 | 0.0536 | 159 |
| 4 | `decoder_ss_linear_0_to_075` | 0.2193 | 0.2096 | 0.2744 | 0.0551 | 0.2705 | 0.0609 | 119 |
| 5 | `decoder_ss_fixed_075` | 0.2190 | 0.2104 | 0.2566 | 0.0377 | 0.2554 | 0.0450 | 189 |
| 6 | `memory_only_mix035` | 0.2176 | 0.2109 | 0.2794 | 0.0617 | 0.2790 | 0.0681 | 159 |
| 7 | `decoder_ss_fixed_050` | 0.2173 | 0.2156 | 0.2741 | 0.0568 | 0.2766 | 0.0610 | 159 |

Best rollout test setting: `decoder_ss_linear_0_to_050` (0.2245)
Smallest test gap setting: `decoder_ss_fixed_075` (0.0377)
