\# AI Usage Log — AI Technical Interview Agent



\## Project

AI Technical Interview Agent — Problem Statement 2



\## AI Tools Used

\- Claude

\- ChatGPT

\- Antigravity / Gravity



\## Purpose of AI Assistance



AI tools were used throughout development for architecture discussion, debugging,

code review, test generation, UX improvements, and technical reasoning.



\## Major Development Tasks



\### 1. Interview Engine Architecture

Used AI assistance to reason about:

\- Adaptive interview question generation

\- Skill-module progression

\- Candidate answer evaluation

\- Cumulative skill assessment

\- Final candidate disposition



\### 2. Skill Map and Timeline Synchronization

AI assistance was used to identify and fix:

\- Skill status being overwritten between multiple probes

\- Separation between the evaluated module and the next interview module

\- Adaptive timeline progression

\- Canonical module-name normalization

\- Cumulative module evidence tracking



\### 3. Answer Quality Evaluation

AI assistance was used to improve evaluation so that:

\- "I don't know" and weak answers are not treated as strong

\- Shallow keyword-heavy answers are distinguished from deep technical answers

\- Strong answers require technical depth and relevant mechanisms

\- Historical evidence is retained across multiple questions

\- Final disposition reflects actual interview performance



\### 4. Testing and Quality Assurance

AI assistance was used to design and debug regression tests covering:

\- Answer-quality evaluation

\- Cumulative skill assessment

\- State synchronization

\- LLM client behavior

\- Interview flow

\- Final disposition



The final project verification included 89 passing automated tests.



\### 5. Session Lifecycle Debugging

AI assistance was used to diagnose an interview-session failure caused by

Uvicorn's `--reload` restarting the process and clearing the in-memory session store.



The production/demo server was configured to avoid this issue during active interviews.



\### 6. Frontend UX

AI assistance was used to improve the first-time user experience by adding:

\- An onboarding hero section

\- A clear three-step interview flow

\- A prominent Start Technical Interview action

\- Improved candidate selection page scrolling



The existing interview interface and backend logic were preserved.



\### 7. Deployment

AI assistance was used to guide:

\- Git/GitHub workflow

\- Secret management

\- Environment variables

\- Render deployment configuration

\- Production verification



The Anthropic API key is stored as a deployment environment variable and is

not committed to the repository.



\## Human Development and Decision-Making



The project team made the final implementation decisions, reviewed AI-generated

suggestions, tested the resulting application, and verified the behavior of

the final system.



AI assistance was used as a development and debugging aid rather than as a

replacement for testing and engineering judgment.

