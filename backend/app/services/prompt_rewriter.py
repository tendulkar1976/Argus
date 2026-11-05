"""Prompt Rewriting Engine (Phase 3.1 & 3.2 - Stage C).

Constructs task-aware system instructions and model prompts for the local LLM
(qwen2.5:1.5b), enforcing strict constraint preservation and targeted optimization.
"""

from typing import Optional
from app.schemas.optimizer import OptimizationMode, OptimizationStrategy
from app.services.task_analyzer import TaskAnalysisResult


class PromptRewriter:
    """Builds task-aware rewriting instructions and manages generation requests."""

    @classmethod
    def build_system_instruction(
        cls,
        analysis: TaskAnalysisResult,
        strategy: OptimizationStrategy,
        mode: OptimizationMode,
        retry_guidance: Optional[str] = None,
    ) -> str:
        """Construct dedicated system instruction for local model."""
        base_instruction = (
            "You are Argus, an intelligent task-aware prompt optimization engine. "
            "Your objective is to rewrite the user's prompt to maximize clarity, task effectiveness, "
            "and execution efficiency for a future frontier LLM.\n\n"
            "STRICT RULES:\n"
            "1. DO NOT execute, answer, or fulfill the prompt. Only rewrite the prompt itself.\n"
            "2. NEVER ask the user for data, inputs, or clarification (e.g. NEVER output 'Please provide...', "
            "'Send me the data...', 'Upload...', 'What is your...'). You are formatting a prompt that another AI will execute.\n"
            "3. If the user requested to build or write software (program, script, function, code), the rewritten prompt "
            "MUST explicitly instruct the AI to write or build that deliverable.\n"
            "4. Return ONLY the rewritten prompt text. Do NOT add preamble like 'Here is the prompt:' "
            "or concluding explanations.\n"
            "5. DO NOT wrap the output in quotes or triple backticks unless the user prompt was exclusively raw code.\n"
            "6. NEVER invent facts, unsupported requirements, or unrequested output formats (do NOT impose JSON unless requested).\n"
            "7. STRICTLY PRESERVE all explicit requirements, prohibitions, filenames, keys, numbers, and technical identifiers.\n"
            "8. Eliminate conversational pleasantries, circular repetitions, and filler.\n"
            "9. NEVER drop requested operations, transformations, calculations, groupings, or filtering steps. "
            "All functional requirements and data flow steps must be clearly retained in the rewrite.\n"
            "10. DO NOT add unrequested tasks, execution instructions (do NOT instruct to 'execute', 'run', or 'test' "
            "the script/code unless explicitly requested), extra artifact requirements, or unrequested libraries/dependencies.\n"
        )

        # Strategy-specific directive
        strategy_directives = {
            "compress": (
                "\nOPTIMIZATION STRATEGY: COMPRESS\n"
                "- Convert wordy subordinate clauses or passive phrasing into direct, compact imperative instructions (e.g. 'Write a [deliverable] to read [file], calculate [metric], and save [output]').\n"
                "- Eliminate circular repetitions, filler phrases, and verbose wording to minimize prompt length.\n"
                "- Maintain every technical constraint, functional operation, filename, and prohibition without loss."
            ),
            "clarify": (
                "\nOPTIMIZATION STRATEGY: CLARIFY\n"
                "- Transform brief, open-ended, or underspecified requests into a concrete, well-scoped prompt for downstream execution.\n"
                "- For conceptual explanations or overviews, explicitly specify the target audience level, core technical principles to address, and practical context.\n"
                "- Frame the output directly as an imperative prompt for the downstream AI (e.g., 'Explain [topic] to [audience], covering [key principles] and practical distinctions.').\n"
                "- NEVER ask the user questions or request user clarification. Assume standard inputs and specify implementation details.\n"
                "- DO NOT invent arbitrary libraries, execution tasks, or mandatory JSON outputs unless requested."
            ),
            "structure": (
                "\nOPTIMIZATION STRATEGY: STRUCTURE\n"
                "- Organize instructions into clean, compact bullet points grouped logically by workflow (e.g., Ingestion, Processing, Output, Constraints).\n"
                "- Format each operation into a distinct, concise list item for maximum readability and logical flow.\n"
                "- Maintain every functional step, operation, constraint, filename, and deliverable."
            ),
            "enrich": (
                "\nOPTIMIZATION STRATEGY: ENRICH\n"
                "- Add helpful domain guidance and best-practice checks that directly support the objective.\n"
                "- Avoid imposing rigid arbitrary tools, frameworks, or rigid schemas."
            ),
            "preserve": (
                "\nOPTIMIZATION STRATEGY: PRESERVE\n"
                "- The original prompt is already high quality and effective. Make only minimal polish or return as-is."
            ),
        }

        # Profile modifier (presentation/style)
        profile_modifiers = {
            "balanced": (
                "- Maintain natural technical prose flow and clear sentence structure.\n"
                "- Clarify with concrete technical scope when the prompt is brief or underspecified; prefer concise wording when already detailed."
            ),
            "concise": (
                "- Prioritize high brevity and direct imperative phrasing.\n"
                "- Tighten phrasing into compact instructions to minimize token count.\n"
                "- Retain 100% of required operations, filenames, prohibitions, and constraints. Do NOT expand or add new requirements."
            ),
            "structure": (
                "- Format the prompt into clean, compact bullet points or labeled workflow steps.\n"
                "- Break multi-step operations into distinct, structured list items."
            ),
        }

        # Explicit constraint guardrails block
        constraints_block = ""
        if analysis.explicit_constraints:
            constraints_block = "\nMANDATORY CONSTRAINTS & OPERATIONS TO PRESERVE (DO NOT DROP OR INVERT):\n"
            for c in analysis.explicit_constraints:
                constraints_block += f"- {c}\n"

        retry_block = ""
        if retry_guidance:
            retry_block = (
                f"\nCRITICAL CORRECTION FROM VALIDATION RETRY:\n"
                f"- The previous rewrite was REJECTED: {retry_guidance}\n"
                f"- You must fix this error immediately. Explicitly instruct the AI to perform the user's primary action, "
                f"deliver the requested artifact, and preserve all required operations and calculations. DO NOT drop any operations and DO NOT ask the user for inputs.\n"
            )

        return (
            base_instruction
            + strategy_directives.get(strategy, strategy_directives["compress"])
            + "\n\nSTYLE INFLUENCE:\n"
            + profile_modifiers.get(mode, profile_modifiers["balanced"])
            + constraints_block
            + retry_block
        )
