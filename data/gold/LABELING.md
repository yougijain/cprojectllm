# Labeling guide

These are the conventions the gold labels follow. They are a first draft:
settle each one before labeling the first post, change it only by
relabeling every post it touches, and keep this file in step with the
labels, because it is what "correct" means for every number in the
results.

The tool is `python scripts/label.py`. It shows the post and nothing else.

## Which posts

- **One record per post.** When a post lists several roles, label the
  **first role named** (usually the one in the header line) and give its
  salary and seniority. The stack is the post's, not just that role's.
- **Not a job post** (a question, a job seeker, a deleted or empty
  comment, a "we are not hiring right now" note): `:skip` with a reason.
  The next candidate in the queue takes its place, so the split sizes stay
  at 50 and 250.

## Fields

**company** — as the post names it, without legal suffixes ("Acme", not
"Acme Inc."). Null only if the post never says.

**role_title** — the first role, as written ("Senior Backend Engineer").
For "Engineers (Backend, Frontend)", write "Backend Engineer".

**seniority** — from the title, or the text when the title is silent.
- intern: internships, co-ops.
- new_grad: new grad, entry level, junior, early career, 0-2 years.
- mid: explicitly mid-level, or 2-5 years.
- senior: senior, staff, principal, lead, 5+ years.
- unspecified: no signal, or a span ("junior to senior").

**location** — where the hire must be: a city ("San Francisco, CA"), a
list ("NYC / London"), or a region for remote roles with a restriction
("US", "EU"). Write it as the post does, primary place first. Null for
"remote, anywhere" and for posts that never say.

**remote**
- yes: the role can be done fully remote, even if only from one region.
  "Onsite or remote" is yes.
- hybrid: some office days are required.
- no: onsite only, or "no remote".
- unspecified: the post does not say.

**visa_sponsorship** — yes only when offered; no when ruled out, including
"must be authorized to work in the US" and "no sponsorship";
unspecified otherwise.

**salary** — the labeled role's **base**, annualized.
- Range: min and max. Single figure: min = max. "150k+": min only.
  "Up to 200k": max only.
- Hourly × 2,080; monthly × 12.
- Leave out equity, bonus and OTE when a base is given; use OTE only
  when it is the only figure.
- Currency is the ISO code. `$` is USD unless the post says CAD, AUD and
  so on. No currency written: leave it blank.
- No salary in the post: leave the whole field blank.

**tech_stack** — technologies the post says the team uses or the role
needs: languages, frameworks, databases, clouds, infrastructure tools. Not
practices ("microservices", "CI/CD") or domains ("AI", "fintech").
Spellings are canonicalized on save (`Postgres` becomes `postgresql`).

**new_grad_eligible**
- yes: the post welcomes new grads, entry-level or junior applicants, or
  the role is a new-grad or intern-conversion role.
- no: a stated requirement of 2+ years, or a senior-or-above title.
- unspecified: everything else.

## Ambiguity

When a field is a judgment call, label your best reading and write
`ambiguous: <field>` in the note, with a few words on why. The error
analysis counts these separately: a model that picks the other reasonable
reading of an ambiguous post is not wrong in the same way as one that
invents a salary.

## Choices still open

These affect how validation and scoring treat labels and outputs. Decide
them and record the decision here:

- Are enum values case-insensitive ("Yes" vs "yes")?
- Do true/false count as "yes"/"no" for the yes/no fields?
- How are company names compared (case, punctuation, "Inc.")?
- How are locations compared ("SF" vs "San Francisco, CA")?
- Does "Senior" in a title alone make new_grad_eligible "no"?
