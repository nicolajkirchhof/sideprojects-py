Summarize the following market emails into a structured brief for a swing trader.

Evaluate against the GO/NO-GO regime framework in your system prompt.

# Emails

{emails}

# Required JSON Output

```json
{
  "regime": "GO | CAUTION | NO-GO",
  "regime_reasoning": "Why this regime assessment based on SPY/VIX/breadth",
  "themes": ["Key market themes or sector rotations"],
  "movers": ["Notable stock movers with brief reason"],
  "risks": ["Upcoming risk events (FOMC, CPI, earnings, etc.)"],
  "action_items": ["Specific actions for the trader based on regime"]
}
```
