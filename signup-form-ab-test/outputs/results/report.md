# Effect of a shorter signup form (8 -> 3 fields) on signup completion

## Sample size and duration
- Required sample: 9,490 per group (18,980 total) [alpha=0.05, power=0.8, baseline=40%, MDE=2.0 pts]
- Duration: 14 days (rounded to full weeks), 10,500 users per group expected
- Smallest detectable effect: 1.89 pts over 14 days, 2.68 pts over one week

## Data quality checks
- Simulated users: 21,000 (seed=2026, true lift = 2.5 pts)
- Duplicate user ids: 0
- Missing values in primary metric/variant: 0
- SRM check: A=10,510, B=10,490, chi2=0.02, p=0.89 -> healthy

## Primary metric: signup completion rate
- A: 40.01%   B: 42.83%
- Effect: 2.82 pts, 95% CI [1.49, 4.15], p=3.3e-05
- With CUPED: 2.81 pts, 95% CI [1.50, 4.12] (variance reduction 3.5%)

## Secondary metrics (Benjamini-Hochberg corrected)
- Form time in seconds (completers): diff=-58.207, CI [-59.868, -56.546], p_adj=0, significant=True
- Active accounts per visitor: diff=0.012, CI [0.000, 0.024], p_adj=0.0419, significant=True

## Guardrail metrics
- Email verification rate: A=0.850, B=0.853, diff=0.0025, CI [-0.0125, 0.0174] -> healthy
- Active after 7 days rate: A=0.615, B=0.603, diff=-0.0121, CI [-0.0326, 0.0084] -> healthy
- Fake/duplicate account rate: A=0.032, B=0.030, diff=-0.0023, CI [-0.0096, 0.0050] -> healthy

## Recommendation
- Decision: Launch: the effect is positive with confidence and guardrails are healthy.
- Estimated extra active accounts per year: 6,677 (CI 244 to 13,109)
- Note: the lower CI bound (1.49 pts) is below the 2-point threshold, so the true effect may be smaller than what matters commercially.
- Limits: simulated data; novelty effects and multi-device users are not modeled in the real world.
