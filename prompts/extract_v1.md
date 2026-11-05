You extract structured fields from one job post from Hacker News's monthly "Who is hiring?" thread.

Reply with exactly one JSON object and nothing else: no prose, no explanation. It must have exactly these keys:

- "company": the company's name as the post gives it, without legal suffixes like "Inc.", or null.
- "role_title": the first role the post names, as written, or null.
- "seniority": one of "intern", "new_grad", "mid", "senior", "unspecified". new_grad covers new grad, entry level, junior and early career. senior covers senior, staff, principal and lead. Use "unspecified" when the post gives no signal or spans levels.
- "location": where the hire must be (a city like "San Francisco, CA", a list like "NYC / London", or a region like "US" for remote roles restricted to one), or null if the role is remote anywhere or the post does not say.
- "remote": one of "yes" (can be done fully remote, even if only from one region), "no" (onsite only), "hybrid" (some office days required), "unspecified".
- "visa_sponsorship": "yes" only if offered, "no" if ruled out (including "must be authorized to work in the US"), otherwise "unspecified".
- "salary_min", "salary_max": the first role's annual base salary as plain numbers (120000, not "120k"). Convert hourly pay at 2080 hours a year and monthly pay at 12 months. A single figure is both min and max. "150k+" has no max; "up to 200k" has no min. Use null for anything not stated. Ignore equity and bonus.
- "salary_currency": the ISO currency code ("USD", "EUR", "GBP"), with "$" meaning USD unless the post says otherwise. null if there is no salary or no currency is written.
- "tech_stack": a list of the technologies the post says the team uses or the role needs (languages, frameworks, databases, clouds, infrastructure tools), not practices or domains. Use [] if none.
- "new_grad_eligible": "yes" if the post welcomes new grads, entry-level or junior applicants; "no" if it requires 2+ years of experience or the role is senior or above; otherwise "unspecified".

Only use what the post says. Never guess a value the post does not state: use null or "unspecified" instead.
