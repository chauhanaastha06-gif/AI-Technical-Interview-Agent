import json
import re
from typing import List, Dict, Optional, Any
from app.config import settings
from app.models.domain import InterviewBrief, MissionStatus
from app.models.schemas import FeedbackResponse
from app.utils.logging import logger

try:
    import anthropic
    HAS_ANTHROPIC_PKG = True
except ImportError:
    HAS_ANTHROPIC_PKG = False


class LLMClient:
    def __init__(self):
        self.api_key = settings.ANTHROPIC_API_KEY
        self.model = settings.ANTHROPIC_MODEL
        self._client = None

        if self.api_key and HAS_ANTHROPIC_PKG:
            try:
                self._client = anthropic.Anthropic(api_key=self.api_key)
                logger.info(f"Initialized Anthropic client with model: {self.model}")
            except Exception as e:
                logger.warning(f"Failed to initialize Anthropic client: {e}. Falling back to mock mode.")
                self._client = None
        else:
            logger.info("No Anthropic API key provided or package missing. Operating in deterministic Mock LLM mode.")

    @property
    def is_mock(self) -> bool:
        return self._client is None

    def generate_interview_reply(
        self,
        system_prompt: str,
        conversation_history: List[Dict[str, str]],
        brief: InterviewBrief,
        turn_count: int = 0,
    ) -> str:
        """
        Generates the next interviewer turn reply.
        """
        if not self.is_mock and self._client:
            try:
                # Format messages for Anthropic SDK
                # Ensure messages alternate user/assistant and first message is user
                messages = []
                for msg in conversation_history:
                    role = "user" if msg["role"] == "user" else "assistant"
                    messages.append({"role": role, "content": msg["content"]})

                # If conversation history is empty (start of interview), prompt the model to begin
                if not messages:
                    messages = [{"role": "user", "content": "Hello, I am ready to begin the interview."}]

                response = self._client.messages.create(
                    model=self.model,
                    max_tokens=600,
                    temperature=0.7,
                    system=system_prompt,
                    messages=messages,
                )

                if response and response.content:
                    text_parts = [b.text for b in response.content if hasattr(b, "text")]
                    return "".join(text_parts).strip()
            except Exception as e:
                logger.error(f"Anthropic API error during interview turn: {e}. Falling back to deterministic engine.")
                # Graceful fallback to mock reply if API fails
                return self._generate_mock_reply(brief, conversation_history, turn_count)

        # Mock Mode
        return self._generate_mock_reply(brief, conversation_history, turn_count)

    def generate_feedback(
        self,
        brief: InterviewBrief,
        conversation_history: List[Dict[str, str]],
        feedback_prompt: str,
    ) -> FeedbackResponse:
        """
        Generates final structured feedback with retry and deterministic fallback.
        """
        if not self.is_mock and self._client:
            # 1. Try with Anthropic Structured Tool Calling
            try:
                feedback = self._call_anthropic_structured_feedback(feedback_prompt)
                if feedback:
                    return feedback
            except Exception as e:
                logger.warning(f"First attempt at structured feedback failed: {e}. Retrying with strict JSON instruction.")

            # 2. Retry with stricter instruction
            try:
                strict_prompt = (
                    feedback_prompt
                    + "\n\nCRITICAL: You MUST respond ONLY with a raw valid JSON object matching this exact schema: "
                    + '{"summary": "string", "strengths": ["string"], "gaps": ["string"], "next": ["string"]}. '
                    + "Do not include markdown codeblocks or any additional text."
                )
                response = self._client.messages.create(
                    model=self.model,
                    max_tokens=800,
                    temperature=0.2,
                    messages=[{"role": "user", "content": strict_prompt}],
                )
                if response and response.content:
                    text_parts = [b.text for b in response.content if hasattr(b, "text")]
                    raw_text = "".join(text_parts).strip()
                    parsed = self._extract_json_feedback(raw_text)
                    if parsed:
                        return parsed
            except Exception as e:
                logger.error(f"Retry structured feedback failed: {e}. Using deterministic fallback feedback.")

        # 3. Deterministic fallback feedback
        return self._generate_deterministic_feedback(brief, conversation_history)

    def _call_anthropic_structured_feedback(self, prompt: str) -> Optional[FeedbackResponse]:
        tools = [
            {
                "name": "submit_interview_feedback",
                "description": "Submit structured technical interview feedback for the candidate.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "Executive summary of the candidate's technical performance.",
                        },
                        "strengths": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of key technical strengths demonstrated.",
                        },
                        "gaps": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of technical knowledge gaps or struggle areas.",
                        },
                        "next": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Actionable next steps and recommendations.",
                        },
                    },
                    "required": ["summary", "strengths", "gaps", "next"],
                },
            }
        ]

        response = self._client.messages.create(
            model=self.model,
            max_tokens=800,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
            tools=tools,
            tool_choice={"type": "tool", "name": "submit_interview_feedback"},
        )

        for block in response.content:
            if getattr(block, "type", "") == "tool_use" and getattr(block, "name", "") == "submit_interview_feedback":
                input_data = getattr(block, "input", {})
                return FeedbackResponse(
                    summary=input_data.get("summary", ""),
                    strengths=input_data.get("strengths", []),
                    gaps=input_data.get("gaps", []),
                    next=input_data.get("next", []),
                )
        return None

    def _extract_json_feedback(self, text: str) -> Optional[FeedbackResponse]:
        try:
            # Look for JSON object in string
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                return FeedbackResponse(
                    summary=str(data.get("summary", "")),
                    strengths=[str(s) for s in data.get("strengths", []) if s],
                    gaps=[str(g) for g in data.get("gaps", []) if g],
                    next=[str(n) for n in data.get("next", []) if n],
                )
        except Exception as e:
            logger.debug(f"JSON extraction error: {e}")
        return None

    def _generate_mock_reply(
        self,
        brief: InterviewBrief,
        conversation_history: List[Dict[str, str]],
        turn_count: int,
    ) -> str:
        """
        High-quality deterministic mock interviewer capable of conducting complete,
        calibrated, multi-turn technical interviews tailored to candidate history.
        """
        cand_name = brief.candidate_name
        role = brief.job_role
        exp = brief.years_experience

        # Turn 0: Opening greeting and first technical question
        if not conversation_history or turn_count == 0:
            if brief.failed:
                target = brief.failed[0]
                return (
                    f"Welcome {cand_name}! It's great to speak with you today. Given your background as a {role} "
                    f"({exp} years of experience), I'd like to dive into your technical work during the cohort. "
                    f"Let's start with Day {target.day} on {target.title}. Could you explain how vector embeddings "
                    f"represent semantic similarity, and what distance metrics (like Cosine vs Euclidean) you would select for high-dimensional document retrieval?"
                )
            elif brief.struggled:
                target = brief.struggled[0]
                return (
                    f"Welcome {cand_name}! Thanks for joining us today for this technical assessment. Looking at your "
                    f"profile as a {role} ({exp} years of experience), I'd like to begin by discussing {target.title} (Day {target.day}). "
                    f"When working with this topic, what were the primary architectural or implementation challenges you encountered, and how did you resolve them?"
                )
            elif brief.skipped:
                target = brief.skipped[0]
                return (
                    f"Hello {cand_name}, welcome to your technical interview! With your experience as a {role}, "
                    f"I'd like to explore your technical approach. Let's start by discussing Day {target.day} ({target.title}). "
                    f"How would you conceptually design and implement this component within an end-to-end AI system?"
                )
            else:
                target = brief.mastered[0] if brief.mastered else None
                topic = target.title if target else "Vector Databases & Retrieval"
                return (
                    f"Welcome {cand_name}! It's a pleasure to conduct your technical interview. Based on your impressive track record as a {role}, "
                    f"let's start with {topic}. Could you walk me through the end-to-end data pipeline from document chunking to semantic vector retrieval, and how you optimize for search latency?"
                )

        # Subsequent turns: Contextually adapt based on candidate's message and question index
        last_user_msg = ""
        for m in reversed(conversation_history):
            if m["role"] == "user":
                last_user_msg = m["content"]
                break

        # Check turn index to sequence through curriculum topics
        if turn_count == 1:
            # Follow up or transition to Prompt Engineering / Structured Output / RAG
            if exp >= 6:
                return (
                    f"That makes sense. In a production environment with high throughput, how do you handle vector database index updates, "
                    f"and what strategies do you employ for hybrid search (combining sparse keyword search with dense vector embeddings)?"
                )
            else:
                return (
                    f"Good explanation. When chunking documents for vector indexing, what trade-offs exist between small fixed-size chunks "
                    f"and larger semantic paragraphs, especially regarding context window limits and retrieval precision?"
                )

        elif turn_count == 2:
            # Topic: Prompting, Function Calling & Structured Outputs (Module 4)
            struggled_prompt = next((m for m in brief.struggled if m.module_number == 4), None)
            if struggled_prompt:
                return (
                    f"Let's transition to Module 4 (LLM Core & Prompting), specifically Day {struggled_prompt.day} ({struggled_prompt.title}). "
                    f"When implementing function calling and structured outputs (e.g. JSON mode or Pydantic validation), how do you ensure the LLM strictly adheres to your schema and handles tool validation errors gracefully?"
                )
            else:
                return (
                    "Let's move to LLM integration and structured outputs. When integrating an LLM into an API backend, "
                    "what techniques do you use to enforce structured JSON outputs and guarantee type safety before passing data to downstream services?"
                )

        elif turn_count == 3:
            # Topic: Agentic AI & MCP / LangChain (Module 6)
            agent_mission = next((m for m in (brief.struggled + brief.failed + brief.mastered) if m.module_number == 6), None)
            agent_title = agent_mission.title if agent_mission else "Multi-Agent Orchestration & Model Context Protocol"
            if exp >= 6:
                return (
                    f"Now let's explore {agent_title}. When architecting multi-agent workflows, how do you manage shared state, "
                    f"prevent infinite reasoning loops, and implement context distillation when conversation histories exceed token limits?"
                )
            else:
                return (
                    f"Let's talk about {agent_title}. What is the fundamental difference between a single-step LLM chain and an autonomous agent equipped with tools, "
                    f"and how does the Model Context Protocol (MCP) standardize tool execution?"
                )

        elif turn_count == 4:
            # Topic: Security, Guardrails, Docker & Deployment (Module 7)
            sec_deploy = next((m for m in (brief.failed + brief.skipped + brief.struggled + brief.mastered) if m.module_number == 7), None)
            if sec_deploy:
                return (
                    f"Looking at Day {sec_deploy.day} ({sec_deploy.title}), how do you implement prompt injection defense, output guardrails, "
                    f"and containerize this FastAPI AI service using Docker for scalable Kubernetes deployment?"
                )
            else:
                return (
                    "Regarding security and deployment: how do you protect production LLM endpoints against indirect prompt injection, "
                    "and what metrics (e.g. time-to-first-token, token throughput) do you monitor in production?"
                )

        elif turn_count == 5:
            # Topic: Evaluation & Hallucination Benchmark
            if exp >= 6:
                return (
                    "Moving to evaluation and quality assurance: how do you measure RAG retrieval recall, "
                    "detect hallucinations programmatically, and establish automated LLM-as-a-judge evaluation suites before model deployments?"
                )
            else:
                return (
                    "Moving to testing and evaluation: how do you evaluate RAG retrieval quality and "
                    "detect hallucinations or inaccuracies in LLM-generated responses?"
                )

        elif turn_count == 6:
            # Topic: Asynchronous Token Streaming & Real-Time Moderation
            if exp >= 6:
                return (
                    "In a high-throughput production API service, how do you manage asynchronous token streaming "
                    "(via SSE or WebSockets) while enforcing real-time guardrail moderation without increasing end-to-end latency?"
                )
            else:
                return (
                    "When delivering real-time LLM responses to end users, how do streaming responses work in a web backend "
                    "and how do you handle network interruptions or partial response errors?"
                )

        elif turn_count == 7:
            # Topic: Cost Optimization, Semantic Caching & Model Routing
            if exp >= 6:
                return (
                    "API token costs and latency can scale quickly under load. What strategies do you employ for "
                    "semantic prompt caching, dynamic model routing (e.g. routing simple queries to smaller models and complex tasks to frontier models), and rate limiting?"
                )
            else:
                return (
                    "To optimize API costs and response latency, how do you implement prompt caching and "
                    "select the right model size for different types of candidate queries?"
                )

        elif turn_count == 8:
            # Topic: Model Adaptation & Fine-Tuning Trade-offs
            if exp >= 6:
                return (
                    "When deciding between RAG context injection and Parameter-Efficient Fine-Tuning (like LoRA or QLoRA) "
                    "for enterprise domain adaptation, what core architectural trade-offs and maintenance factors guide your decision?"
                )
            else:
                return (
                    "When would you choose to fine-tune an open-source LLM using techniques like LoRA versus "
                    "providing relevant context dynamically through a RAG pipeline?"
                )

        elif turn_count >= 9:
            # Topic: Capstone, Production Architecture & Enterprise Observability (Final Turn)
            return (
                f"Thank you for those insights. For our final technical topic, let's look at end-to-end production readiness. "
                f"If you were tasked with taking your Capstone Project to production serving 10,000 concurrent enterprise users, "
                f"what would be your caching, rate limiting, and end-to-end tracing/observability strategy?"
            )

        # Fallback question
        return (
            "Thank you for sharing your thoughts. Could you elaborate further on how you would test and validate this component "
            "to prevent silent regressions in an automated CI/CD pipeline?"
        )


    def _generate_deterministic_feedback(
        self,
        brief: InterviewBrief,
        conversation_history: List[Dict[str, str]],
    ) -> FeedbackResponse:
        """
        Deterministic, high-quality feedback generation aligned with candidate profile.
        """
        cand_name = brief.candidate_name
        role = brief.job_role
        exp = brief.years_experience

        # Build strengths
        strengths = []
        if brief.mastered:
            mastered_names = [f"Day {m.day} ({m.title})" for m in brief.mastered[:3]]
            strengths.append(f"Demonstrated solid mastery in core curriculum areas including {', '.join(mastered_names)}.")
        else:
            strengths.append("Demonstrated foundational familiarity with modern AI engineering concepts and workflows.")

        if brief.first_try_rate >= 0.7:
            strengths.append(f"High initial problem-solving velocity with a {brief.first_try_rate:.0%} first-try pass rate across cohort missions.")
        elif brief.commit_days >= 20:
            strengths.append(f"Consistent engagement and persistence demonstrated through {brief.commit_days} active commit days.")

        if exp >= 5:
            strengths.append(f"Applied practical engineering perspective suited for {role} role, considering system boundaries and integration.")
        else:
            strengths.append("Clear communication and structured approach to answering technical questions step-by-step.")

        # Build gaps
        gaps = []
        if brief.failed:
            failed_names = [f"Day {m.day} ({m.title})" for m in brief.failed]
            gaps.append(f"Knowledge gaps identified in failed missions: {', '.join(failed_names)}; requires deeper conceptual and hands-on reinforcement.")
        if brief.struggled:
            struggled_names = [f"Day {m.day} ({m.title}, {m.attempts} attempts)" for m in brief.struggled[:2]]
            gaps.append(f"Encountered friction during implementation in: {', '.join(struggled_names)}.")
        if brief.skipped:
            skipped_names = [f"Day {m.day} ({m.title})" for m in brief.skipped[:2]]
            gaps.append(f"Skipped missions ({', '.join(skipped_names)}) indicate unverified proficiency in these advanced topics.")
        if not gaps:
            gaps.append("Further optimization needed on edge-case error recovery and high-concurrency production latency.")

        # Build next recommendations
        next_steps = []
        if brief.failed:
            next_steps.append(f"Re-implement failed mission exercises ({', '.join([m.title for m in brief.failed])}) from scratch without starter templates.")
        if brief.skipped:
            next_steps.append(f"Complete hands-on labs for skipped topics ({', '.join([m.title for m in brief.skipped[:2]])}) to ensure comprehensive coverage.")
        next_steps.append("Build a production benchmark suite measuring RAG retrieval recall, hallucination rate, and p99 latency.")
        next_steps.append("Study advanced agent orchestration patterns and standardized Model Context Protocol (MCP) integrations.")

        # Summary
        if len(brief.failed) > 0 or len(brief.struggled) >= 4:
            perf_level = "shows foundational capability but requires targeted remediation in weak and struggled areas"
        elif brief.first_try_rate >= 0.7 and len(brief.mastered) >= 6:
            perf_level = f"demonstrated strong technical proficiency and architectural maturity consistent with a {role}"
        else:
            perf_level = f"exhibited balanced technical understanding across the AI curriculum with practical problem-solving aptitude"

        summary = (
            f"Candidate {cand_name} ({role}, {exp} years experience) {perf_level}. "
            f"During the interview, the candidate engaged thoughtfully across curriculum modules, exhibiting clear strengths "
            f"alongside specific areas identified for continued refinement before production deployment."
        )

        return FeedbackResponse(
            summary=summary,
            strengths=strengths,
            gaps=gaps,
            next=next_steps,
        )


# Singleton LLM client
llm_client = LLMClient()
