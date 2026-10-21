# ARP1 amendment 01 — stronger recurrence comparator

21 October 2026, during acquisition and before viewing any target arrays,
reference diagnostics or activation scores. Only execution progress has been
inspected. This is a design correction from code review, not a response to a
classification result.

The original baseline requires room for three periods in a short window. That
can disadvantage recurrence at small observation budgets. Retain it exactly,
and add a second baseline allowing two complete periods, with the same 1e-8
relative tolerance and closure required at every available phase pair.

Report both comparators at all four budgets. Any suggested activation benefit
must be compared with this stronger baseline too. Neither baseline may label
non-recurrence as proven chaos: a closed-set guess is distinguished from the
operational unresolved output. Acquisition, parameter selection, splits,
reference gates and activation metrics are unchanged. All results remain
exploratory. Implement this additional analysis after acquisition completes,
so the acquisition source binding remains intact.
