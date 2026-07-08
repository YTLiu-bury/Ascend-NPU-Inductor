# PyTorch Community Issue Submission & RFC End-to-End Practical Guide

---

## Overview: A Bird's-Eye View of the Full Process

```
💡 Idea
  │
  ├── What type of change is this?
  │     ├── 🐛 Bug Fix ──────────→ Submit PR directly (jump to ④)
  │     ├── 📄 Doc / Enhancement ──→ Issue Discussion → PR (jump to ④)
  │     └── 🏗️ New Feature / New API ──→ Continue ↓
  │
  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ ① 📝 RFC Writing                                                              │
│                                                                              │
│   Core Principle: Approach from the perspective of community benefit,        │
│                   thereby achieving your own objectives                       │
│                                                                              │
│   ✎ Template: 9 elements, use as needed — the key is to clearly             │
│     articulate your logic                                                     │
│   ✎ Reference Case: torch.unravel_index (issue#185590)                       │
│   ✎ Speak through code examples + demonstrate research depth +              │
│     honestly face Drawbacks                                                   │
│                                                                  Details → Phase 1 │
└──────────────────────┬───────────────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ ② 🏠 Internal Review (Align Internally First, Then Go Out)                   │
│                                                                              │
│   ✎ Internally present the proposal → Self-review against community          │
│     standards → Revise until consensus reached                                │
│   ✎ Examine from a maintainer's perspective: Is the benefit broad enough?    │
│     Is the API concise enough? Are the Drawbacks fully considered?            │
│                                                                  Details → Phase 2 │
└──────────────────────┬───────────────────────────────────────────────────────┘
                       │ Internally Approved
                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ ② 🌐 Community Review                                                          │
│                                                                              │
│   ✎ Locate reviewers: CODEOWNERS + Persons of Interest → @ maintainer       │
│   ✎ Broadcast RFC: dev-discuss + Slack + Main Repo Issue                     │
│   ✎ No response escalation: Day2-3 ping → Day4 Slack maintainer            │
│     (Prerequisite: Provide GitHub ID + Gmail, invited to Slack workspace)     │
│     → Day5+ Attend Friday Office Hours → File DevX issue                    │
│                                                                  Details → Phase 2 │
└────────┬─────────┬───────────────────────────────────────────────────────────┘
         │         │
    ✅ Accepted    ❌ Shelved (Deferred, may be reopened)
         │
         ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ ③ 💻 Coding                                                                   │
│                                                                              │
│   ✎ Wait for RFC Acceptance before writing production code                   │
│     (Can write a prototype first to validate)                                 │
│   ✎ PR < 200 lines; one PR does one thing; break into multiple              │
│     small PRs and submit sequentially                                         │
│   ✎ Run local CI (lint + test) before submitting — all-red CI gets           │
│     no attention                                                              │
│                                                                  Details → Phase 3 │
└──────────────────────┬───────────────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ ④ 👀 Code Review                                                              │
│                                                                              │
│   Triage(~2 days) → CI + Review → 1 Approve → @pytorchbot merge             │
│                                    → Trunk Tests(~6-8h) → ✅ Merged          │
│                                                                              │
│   ✎ Draft PR means "Still under construction, please don't review";          │
│     convert to formal PR when ready                                           │
│   ✎ Reviewers focus on: PR description, backward compat, docs,              │
│     test coverage, numerics                                                   │
│   ✎ CI gate-trigger permissions are limited — find a maintainer               │
│     to trigger after passing local validation                                 │
│   ✎ No PR submissions Friday afternoons (no one reviews on weekends)         │
│                                                                  Details → Phase 4 │
└────────┬─────────┬───────────────────────────────────────────────────────────┘
         │         │
         │    ❌ Request Changes
         │         │
         │         ▼
         │    🔧 Modify Code
         │         │
         │         ▼
         │    Submit Update → Reviewer should re-review within 24h
         │
         └────┬────┘
              │
              ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ ⑤ 💬 Respond to Review Comments                                              │
│                                                                              │
│   💬 Comment (Suggestion) → Decide whether to adopt yourself;               │
│     reply "Done" or "Considered; chose X because…"                            │
│   ❌ Request Changes → Must fix; if disagree, argue with code/data           │
│   ✅ Approve → Read ALL comments before merging; don't rush                  │
│                                                                              │
│   ✎ Reply to EVERY comment (even if just a 👍), ASAP, don't force push,     │
│     don't squash commits (convenient for review)                              │
│   ✎ Maintainers may trigger AI review — AI comments also need               │
│     point-by-point replies                                                   │
│   ✎ Don't be defensive — reviewers are criticizing the code, not you        │
│   ✎ Reviewers should re-review within 24h; if >2-3 days, ping politely      │
│                                                                  Details → Phase 5 │
└──────────────────────┬───────────────────────────────────────────────────────┘
                       │
                       ▼
                     ✅ Merged
```

### Communication Channels Cheat Sheet

| Channel | Purpose | Response Time |
|---|---|---|
| **GitHub PR / Issue** | Code discussion, decision recording | 2-4 business days |
| **dev-discuss Forum** | Design discussions, RFC dissemination | 1-3 days |
| **Slack** (Invitation required: GitHub ID + Gmail) | Quick Q&A, contacting maintainers | Minutes to hours |
| **Office Hours** (Weekly Friday) | Urgent unblocking | Real-time |

---

## Phase 1: RFC Writing

### 1.1 Does Your Idea Need an RFC?

Before putting pen to paper, self-check with three questions:

| Question | Answer "Yes" → RFC Required |
|---|---|
| Does it introduce a new abstraction layer or core concept? | ✅ Requires RFC |
| Does it involve API breaking changes? | ✅ Requires RFC |
| Does it span multiple modules (e.g., simultaneously affecting autograd + nn + distributed)? | ✅ Requires RFC |
| Is it just a bug fix or doc improvement? | ❌ Submit PR directly |
| Is it a small API enhancement (e.g., adding a parameter)? | ❌ File Issue + dev-discuss discussion is sufficient |

### 1.2 RFC Writing Guide

PyTorch officially provides an RFC template (`RFC-0000-template.md`) containing 9 sections that can serve as structural reference. However, when actually writing, **strict adherence to the template is unnecessary** — the core is to clearly articulate the proposal's logic so maintainers and the community can quickly understand your intent.

> **📌 Core Principle: Approach from the perspective of community benefit, thereby achieving your own objectives.**
>
> The essence of an RFC is to convince the community that "this change brings value to the PyTorch ecosystem." Your proposal may serve your own business scenario, but the argumentative logic should be: **this pain point is not unique to you but is widespread across the community** → **your solution can systematically address this class of problems** → **community adoption will benefit more users and scenarios**. Arguing for the proposal from the community's standpoint is what truly drives it through.
>
> Reference a successful RFC case: [torch.unravel_index](https://github.com/pytorch/pytorch/issues/185590) — starting from NumPy alignment and high-frequency community user needs, concisely and clearly arguing the rationale of the API design.

Below are the intent descriptions for each section of the template, to be used as needed:

```
RFC File Structure (In Writing Order)
═══════════════════════════════════════════════════════════════

1. SUMMARY
   ✎ Use 3-5 sentences/bullets to explain what you will do
   ⚠️ Key: Let maintainers understand your proposal in 30 seconds
   Example: "This RFC proposes adding a WeightOnlyQuantizedLinear
          layer to torch.nn to support INT4/INT8 weight quantization
          inference, reducing memory consumption by 75%."

2. MOTIVATION
   ✎ Why is this proposal important? What is the cost of not doing it?
   ⚠️ Key: Speak with data and scenarios, don't just say "I think it's useful"
   Points to cover:
   - What user pain point does it solve?
   - How many users/scenarios will benefit?
   - If not done, what is the current workaround? How painful is it?
   - Is it aligned with PyTorch's strategic direction?

3. PROPOSED IMPLEMENTATION
   ✎ The core part of the RFC, needs to be detailed enough
   ⚠️ Key: "People familiar with PyTorch should understand the design;
           people familiar with implementation should be able to code directly"
   Must include:
   - API design (with complete code examples showing how users use it)
   - Internal architecture diagrams/data flow (text descriptions or ASCII art work)
   - Edge case handling (empty input, extreme values, different dtype/device combos)
   - Interaction with existing features (How will it interact with
     torch.compile, autograd, Distributed?)
   - New terminology definitions (if new concepts are introduced)

4. METRICS
   ✎ How to measure the value of this feature?
   Examples:
   - Performance: forward latency reduced by X%, memory consumed reduced by Y%
   - Adoption: expected to be adopted by Z downstream projects
   - Accuracy: accuracy on benchmark A is not lower than baseline

5. DRAWBACKS
   ✎ Honestly assess "why we should NOT do this"
   ⚠️ Key: This is highly valued by maintainers and reflects your technical maturity
   Cover at least:
   - Is it a breaking change? If so, what's the migration plan?
   - How much does code complexity increase? What's the maintenance burden?
   - Conflict with existing/planned features
   - Impact on user experience (does the API become more complex?)

6. ALTERNATIVES
   ✎ What other designs have you considered? Why didn't you choose them?
   ⚠️ Key: Demonstrates you've done thorough research, not a knee-jerk proposal
   Include:
   - Option A: xxx (pros/cons)
   - Option B: xxx (pros/cons)
   - What are the consequences of doing "nothing"?

7. PRIOR ART
   ✎ Do other frameworks/libraries have similar features? What are the
     lessons learned?
   Reference sources:
   - Corresponding features in TensorFlow / JAX / MXNet
   - Implementations in academic papers
   - Practices in other languages/ecosystems
   ⚠️ Cover both good experiences and lessons from failures

8. HOW WE TEACH THIS
   ✎ How do users learn this new feature?
   Points to cover:
   - Are naming and terminology intuitive?
   - Do you need to add new doc chapters or reorganize existing docs?
   - How to teach existing PyTorch users? (example code, tutorials, blogs?)

9. UNRESOLVED QUESTIONS
   ✎ Honestly list things you're still unsure about that need to be
     resolved during RFC discussion
   Categorize:
   - Must be resolved before RFC merge
   - Can be resolved during implementation phase
   - Explicitly outside the scope of this RFC
```

### 1.3 Golden Rules for RFC Writing

```
┌────────────────────────────────────────────────────────────┐
│  📌 Three Golden Rules for RFC Writing                     │
├────────────────────────────────────────────────────────────┤
│  1. Speak through code examples — what does the API look   │
│     like? Show with example code.                           │
│  2. Show you've done your homework — Prior Art +            │
│     Alternatives demonstrate research depth                │
│  3. Honorable > Perfect — Honestly facing issues in        │
│     Drawbacks and Unresolved Questions increases            │
│     maintainer trust                                        │
└────────────────────────────────────────────────────────────┘
```

---

## Phase 2: RFC Review Flow

RFC review is divided into two steps: **first internal review, then community submission for review after passing**. This allows most design issues to be resolved internally, reducing iteration rounds and time spent in the community.

### 2.1 Step 1: Internal Review

After the RFC draft is completed, first conduct a review within your team or organization:

```
Internal Review Flow
═══════════════════════════════════════════════════════════════

Step 1: Internally present the proposal
  ↓   Explain the RFC proposal to colleagues with PyTorch experience
  ↓   Key verification: Is the proposal logic self-consistency?
       Are there obvious omissions or vulnerabilities?

Step 2: Self-review against community standards
  ↓   Examine from a maintainer's perspective:
  ↓   - Is the benefit scope broad enough (not just your scenario)?
  ↓   - Is the API design concise and in line with PyTorch idioms?
  ↓   - Have Drawbacks and Alternatives been fully considered?

Step 3: Revise until internal consensus is reached
  ↓   Only after internal review reaches consensus,
  ↓   proceed to the community review phase
  ↓   ⚠️ If there's still internal disagreement, submitting to the
       community will only amplify the problem
```

### 2.2 Step 2: Community Review

After internal review is passed, formally submit to the PyTorch community for public review.

#### 2.2.1 Precisely Locate Reviewers (The Most Important Step)

Don't just randomly @ someone and expect them to respond. You must precisely locate them:

```
Step 1: Check CODEOWNERS file
  ↓   https://github.com/pytorch/pytorch/blob/master/CODEOWNERS
  ↓   Find the file path involved in your change → corresponding owner

Step 2: Check Persons of Interest page
  ↓   https://docs.pytorch.org/docs/2.12/community/persons_of_interest.html
  ↓   Find the module involved → corresponding maintainer and their GitHub ID

Step 3: @ corresponding maintainer in RFC PR description
  ↓   Format: cc @github_username

Step 4: Multiple notifications (Broadcast your RFC)
  ↓   ① Post under RFC Chatter category in dev-discuss forum
  ↓   ② Mention in Slack channel (if you have permission)
  ↓   ③ Create issue in main pytorch/pytorch repo linking RFC PR
```

### 2.2.2 Communication Template for Contacting Maintainers

**In the RFC PR description:**

```markdown
## RFC: [Title]

RFC PR: pytorch/rfcs#XX

### Modules Involved
- torch.nn → cc @gchanan @jbschlosser
- torch.autograd → cc @ezyang @albanD

### Summary
[One-sentence description]

### Key Areas Where Review is Requested
Particularly seeking feedback on the following aspects:
1. Is the API design reasonable?
2. Are there any omissions in interaction with existing autograd mechanisms?
3. Potential performance risks?

### Related Discussions
- dev-discuss post: [link]
- main repo issue: pytorch/pytorch#XXXX
```

### 2.3 What If No Response? (Escalation Path)

PyTorch has a clear escalation path — don't wait in silence:

```
Day 0:     Submit RFC / PR
Day 2-3:   Politely ping reviewers under the PR (@username please take a look)
Day 4:     ↓ Still no response
           Two-pronged approach:
           ① Contact the person who tagged your PR as triage on GitHub
           ② Directly contact the corresponding maintainer on Slack
              (Prerequisite: Provided GitHub ID and Gmail in Slack and
               invited to the workspace)
Day 5+:    ↓ Still no response
           Attend Dev Infra Office Hours (held every Friday),
           ask in person — this is the most efficient resolution method
           https://github.com/pytorch/pytorch/wiki/Dev-Infra-Office-Hours
           ↓ Still no response
           File an issue with the DevX team describing your blocking situation
```

> **About Slack contact with maintainers**: Compared to asynchronous review on GitHub, Slack is a higher-timeliness channel. However, two prerequisites need to be completed in advance: ① Provide your **GitHub ID** and **Gmail** in the PyTorch Slack workspace; ② Be invited by the admin of the corresponding workspace. After meeting the conditions, you can directly @ the corresponding maintainer in relevant channels to communicate, often getting a response faster than waiting for GitHub review.

---

## Phase 3: When to Start Coding?

This is where community developers most easily stumble — use this decision table:

```
┌────────────────────────────────────────────────────────────────┐
│                   When to Start Coding? Decision Tree          │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Is your proposal a bug fix?                                   │
│    ├── Yes → Write code directly and submit PR ✅              │
│    └── No ↓                                                    │
│                                                                │
│  Is your proposal a small, non-controversial improvement?       │
│  (e.g., add a parameter, fix type annotation, performance tweak)│
│    ├── Yes → Quick discussion on Issue → Write code & PR ✅    │
│    └── No ↓                                                    │
│                                                                │
│  Does your proposal involve new API / new abstraction /         │
│  architectural changes?                                        │
│    ├── Yes → ⚠️ RFC first, write code AFTER Acceptance         │
│    │       (Otherwise you might write something that gets       │
│    │        overturned and rewritten, wasting lots of time)     │
│    └── No ↓                                                    │
│                                                                │
│  Not sure?                                                     │
│    └── → Post on dev-discuss to discuss, ask maintainers       │
│           whether an RFC is needed                              │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ ⚠️ Lesson Learned: Don't start heavy coding BEFORE       │  │
│  │ RFC is accepted! Maintainers might demand a completely    │  │
│  │ different design, and your code might all be scrapped.    │  │
│  │ You can write a prototype first to validate feasibility,   │  │
│  │ but don't invest in production-grade code.               │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Standard Flow After RFC Acceptance

```
RFC Accepted → Create Tracking Issue in Main Repo → Start Implementation
                                                 │
                                         You can implement it yourself,
                                         or wait for someone to claim it
```

---

## Phase 4: Full Code Review Flow

### 4.1 PyTorch PR Complete Lifecycle

```
Submit PR
  │
  ▼
Triage (~2 business days)
  → AutoLabel Bot automatically tags
  → Triage team validates + assigns reviewers
  │
  ▼
CI Execution + Review
  → CI tests run in parallel
  → Reviewers conduct Code Review
  │
  ├──→ 💬 Comment: Non-blocking suggestion, author decides whether to adopt
  ├──→ ✅ Approve: Approval (only 1 Approve needed to enter merge flow)
  └──→ ❌ Request Changes: Blocking modification required, must resolve
        │
        ▼
      Author modifies code → pushes update → Reviewer should re-review
                              within 24 hours
        │
        ▼
Approval + CI Passing
  │
  ▼
@PyTorchBot merge (initiated by author or reviewer)
  │
  ▼
Trunk Tests (~6-8 hours, more comprehensive tests)
  │
  ▼
Merged ✅
```

> **⚠️ About CI Gate Trigger Permissions**: PyTorch's CI pipeline cannot be triggered by all contributors — typically only members with specific permissions (e.g., maintainers or triage team members) can initiate the full CI gate. Therefore, after sufficient local validation + resolving all review comments, you can **find someone with permission (e.g., the maintainer reviewing you) to help trigger the CI** — don't just wait silently.

### 4.2 What Do Reviewers Focus on in PyTorch Code Review?

According to [Code Review Values](https://github.com/pytorch/pytorch/wiki/Code-review-values), reviewers check from the following dimensions:

| Dimension | Check Points |
|---|---|
| **PR Description** | Does it clearly explain "why" this change is made? Does the description match the actual changes? |
| **Engineering Quality** | Is backward compatibility maintained? Are idiomatic patterns used instead of reinventing the wheel? Are there hidden assumptions that might cause bugs in the future? Is this feature worth its maintenance cost? |
| **Documentation** | Do new features have documentation? Do non-obvious logic have comments? |
| **Tests** | Are tests included in the same diff? Do tests cover edge cases? (contiguous/non-contiguous, different dtypes, empty inputs, etc.) For changes that are difficult to test, is manual verification method explained? |
| **Numerics** | Is the kernel bit-for-bit deterministic? Have one-time operations been moved outside critical loops? Can it handle >4GB of data? Is `cudaGetLastError()` called after kernel launch? |

---

## Phase 5: How to Respond to Review Comments

### 5.1 Three Types of Review Comments and Response Strategies

```
┌─────────────────────────────────────────────────────────────────┐
│                    Review Comment Types & Response Strategies    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ 💬 Comment (Non-blocking suggestion)                             │
│    Reviewer's attitude: "Suggest doing this, but you decide"    │
│    Your strategy:                                               │
│   - If agree → Modify, reply "Done, thanks!"                   │
│   - If disagree → Politely explain why, but not mandatory      │
│   - If unsure → Reply "Let me think about this" then evaluate  │
│                                                                 │
│ ❌ Request Changes (Blocking modification requirement)           │
│    Reviewer's attitude: "Must change this, or can't merge"     │
│ Your strategy:                                                   │
│   - Understand reviewer's concern (even if disagree,           │
│     understand first)                                            │
│   - If agree → Modify immediately + reply summary of changes   │
│   - If disagree → Argue your position with code/data/scenarios │
│   - If deadlocked → Ask a third-party reviewer to intervene    │
│                                                                 │
│ ✅ Approve (Approval)                                            │
│    Reviewer's attitude: "Ready to merge"                       │
│ Your strategy:                                                   │
│   - ⚠️ Read ALL comments first! Reviewers may attach minor    │
│     suggestions alongside the Approve                           │
│   - Handle remaining minor suggestions, then @pytorchbot merge │
│   - Don't rush to merge at first sight of Approve               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Communication Templates for Responding to Review Comments

```markdown
### For agreed-upon suggestions:
> "Done. Added boundary check for empty tensor case in the latest commit (abc1234)."

### For disagreed-upon suggestions:
> "Thanks for the suggestion! I considered this approach, but I think the
> current implementation is preferable because:
> 1. [Specific reason, preferably with code/data support]
> 2. [Cite similar existing patterns in the project as precedent]
>
> What do you think? Happy to discuss further."

### For suggestions not understood/nedding clarification:
> "I'm not sure I fully understand the concern here. Could you elaborate on
> what scenario this might break? I'd like to make sure I address the root
> issue, not just the symptom."

### When reviewers have conflicting opinions:
> "It seems like there are two different perspectives here:
> - @reviewer1 suggests approach A (reason: ...)
> - @reviewer2 suggests approach B (reason: ...)
>
> Could we get a third opinion to help resolve this?"
```

### 5.3 Golden Rules for Responding to Review Comments

```
┌────────────────────────────────────────────────────────────┐
│  📌 Four Golden Rules for Responding to Review Comments   │
├────────────────────────────────────────────────────────────┤
│  1. Don't be defensive — Reviewers are criticizing the    │
│     code, not you                                          │
│  2. Reply to EVERY comment — even if just a 👍 or "Done" │
│     This shows you've seriously considered every point     │
│  3. Push new commits after modifications; don't force     │
│     push over old commits (lets reviewers see incremental  │
│     diff for easier re-review). Don't squash commits —     │
│     keep each modification independently preserved so      │
│     reviewers can directly see what changed in the         │
│     latest commit                                           │
│  4. Reviewers should re-review within 24h — if >2-3 days  │
│     no response, ping politely                              │
└────────────────────────────────────────────────────────────┘
```
> **⏱️ Reply Timeliness**: Review comments should be **replied to as soon as possible** — don't leave reviewers waiting for a long time. Fast responses not only accelerate the merge process but also reflect your level of commitment to the contribution. If a certain issue takes longer time to research, you can first reply "Looking into this, will update soon" to let the reviewer know you haven't disappeared.

> **🤖 About AI Review Comments**: PyTorch community maintainers may use AI tools to assist in PR review, and these AI-generated comments will also appear in PR comments. **AI comments also need point-by-point serious replies** — their essence is the same as human reviewer comments, representing concerns about code quality. Ignoring AI comments is equivalent to ignoring review comments and will block your PR.

---

## Phase 6: Efficient Communication Strategies

### 6.1 Communication Channel Selection Matrix

```
┌─────────────────────────────────────────────────────────────────┐
│                     Communication Channel Selection Guide        │
├──────────────┬──────────────────┬────────────────────────────────┤
│ Channel      │ Applicable Scenarios │ Response Time              │
├──────────────┼──────────────────┼────────────────────────────────┤
│ GitHub PR    │ Code-level discussion, │ 2-4 business days           │
│ Comments     │ technical details,     │                             │
│              │ decision recording     │                             │
├──────────────┼──────────────────┼────────────────────────────────┤
│ dev-discuss  │ Design discussion,     │ 1-3 days                    │
│ Forum        │ RFC dissemination,     │                             │
│              │ opinion solicitation   │                             │
├──────────────┼──────────────────┼────────────────────────────────┤
│ Office Hours │ Urgent unblocking,     │ Real-time (every Friday)    │
│              │ face-to-face discussion │                             │
│              │ of complex issues       │                             │
├──────────────┼──────────────────┼────────────────────────────────┤
│ Slack        │ Quick Q&A,             │ Minutes to hours            │
│ (Invitation  │ informal discussion    │                             │
│  Required)   │                        │                             │
├──────────────┼──────────────────┼────────────────────────────────┤
│ GitHub Issue │ Cross-PR discussion,  │ 1-3 days                    │
│ (Main Repo)  │ scope-crept issues     │                             │
└──────────────┴──────────────────┴────────────────────────────────┘
```

### 6.2 Techniques to Make Maintainers Willing to Reply to You

```
✅ DO (Should do)                        ❌ DON'T (Should not do)
─────────────────────────────────    ─────────────────────────────────
• Keep PR small (<200 lines); break   • Submit a 3000-line mega PR
  into multiple small PRs               (Reviewers will close it and leave)

• Clearly write "why" in PR desc      • Only write "fix bug" or "add feature"
  and your design decisions and         (Maintainers don't know where to start)
  trade-offs

• Do a self-review yourself; annotate  • Throw obvious format issues,
  parts you think are controversial or   unfinished code at reviewers
  need focused review                   (Wastes both sides' time)

• Run local CI (lint + test) before    • Submit with all-red CI
  submitting                            (Maintainers won't review failed PRs)

• One PR does only one thing            • Mix refactoring + new features +
                                         bug fixes in the same PR

• Use Draft PR to mean "Still under    • Frequently @ reviewers under Draft PR
  construction, please don't review"   (Draft = Don't look at me)

• Ping only if >4 business days no     • Ping every single day
  response                              (Maintainers are volunteers; they'll resent it)

• No PR submissions Friday afternoons   • Urge during maintainer vacations/
  (Likely no one reviews until Monday)   weekends
```

### 6.3 Core Principles of Asynchronous Communication

Since PyTorch maintainers are spread across different time zones globally (Meta on US West Coast, NVIDIA worldwide, AMD in US, Huawei in China, etc.), asynchronous communication is the norm:

```
┌────────────────────────────────────────────────────────────┐
│  📌 Maximizing Async Communication Efficiency              │
├────────────────────────────────────────────────────────────┤
│ • Every message self-contained — Let reviewers understand  │
│   your question without scrolling back through 50 chat     │
│   records                                                  │
│                                                            │
│ • Anticipate questions & answer proactively — Actively    │
│   explain "why A not B" in your description to reduce     │
│   one round-trip of async communication                   │
│                                                            │
│ • Leverage time differences — You're in UTC+8, maintainer  │
│   in UTC-8. Submit code in the evening → Review comments   │
│   next morning → Modify during the day → Loop              │
│                                                            │
│ • Ideal: One round-trip communication — Strive to make    │
│   every update something the reviewer can decide on after  │
│   reading just once                                        │
└────────────────────────────────────────────────────────────┘
```

---

## Appendix: Quick Reference Sheet

```
┌──────────────────────────────────────────────────────────────────────┐
│                PyTorch Community Issue Quick Reference               │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  📋 Bug Report                                                       │
│     └→ github.com/pytorch/pytorch/issues → Bug Report Template       │
│                                                                      │
│  💡 Small Feature Request                                             │
│     └→ Issue + dev-discuss.pytorch.org Discussion                   │
│                                                                      │
│  🏗️ Large Design Proposal                                             │
│     └→ github.com/pytorch/rfcs → RFC PR                              │
│                                                                      │
│  🔍 Find Maintainers                                                  │
│     └→ CODEOWNERS + Persons of Interest Page → @ corresponding GitHub ID│
│                                                                      │
│  🆘 Blocked                                                           │
│     └→ Ping reviewers (wait 4 days) → Office Hours (Friday) → DevX issue│
│                                                                      │
│  💬 Everyday Discussion                                               │
│     └→ dev-discuss.pytorch.org (main battleground) + Slack (if authorized)│
│                                                                      │
│  📖 Start Contributing                                                │
│     └→ Good First Issue → Small PR → Build reputation → Bigger proposals│
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Reference Links

| Resource | URL |
|---|---|
| RFC Repository | https://github.com/pytorch/rfcs |
| RFC Template | https://github.com/pytorch/rfcs/blob/master/RFC-0000-template.md |
| RFC Process Guide | https://raw.githubusercontent.com/pytorch/rfcs/refs/heads/master/README.md |
| RFC Example (torch.unravel_index) | https://github.com/pytorch/pytorch/issues/185590 |
| Main Repo Contribution Guide | https://docs.pytorch.org/docs/main/community/contribution_guide.html |
| Developer Forum | https://dev-discuss.pytorch.org/ |
| RFC Chatter Category | https://dev-discuss.pytorch.org/c/rfc-chatter |
| User Forum | https://discuss.pytorch.org/ |
| Maintainers List | https://docs.pytorch.org/docs/2.12/community/persons_of_interest.html |
| CODEOWNERS | https://github.com/pytorch/pytorch/blob/master/CODEOWNERS |
| Ultimate Guide to PyTorch Contributions | https://github.com/pytorch/pytorch/wiki/The-Ultimate-Guide-to-PyTorch-Contributions |
| PR Review Etiquette | https://github.com/pytorch/pytorch/wiki/Pull-request-review-etiquette |
| Code Review Values | https://github.com/pytorch/pytorch/wiki/Code-review-values |
| Getting Help | https://github.com/pytorch/pytorch/wiki/Getting-help-as-a-contributor |
| Office Hours | https://github.com/pytorch/pytorch/wiki/Dev-Infra-Office-Hours |
| Governance Documents | https://docs.pytorch.org/docs/2.1/community/governance.html |
