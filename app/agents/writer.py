# Agentic RAG - WriterAgent: grounded answer generation with [n] citations.

from app.llm.client import chat_text
from app.llm.prompts import WRITER_PROMPT


class WriterAgent:
    def _build_messages(self, query, context_blocks, memory_text, feedback=None):
        if context_blocks:
            context = "\n\n".join(
                f"[{r['citation']}] ({r.get('title') or r.get('doc_id') or 'source'})\n{r['content']}"
                for r in context_blocks
            )
        else:
            context = "(no matching documents found in the collection)"
        messages = [{"role": "system", "content": WRITER_PROMPT.format(context=context)}]
        if memory_text:
            messages.append({"role": "system",
                             "content": f"Relevant conversation history:\n{memory_text}"})
        user_content = query
        if feedback:
            user_content += (
                "\n\nYour previous answer had these issues; please fix them: "
                + "; ".join(feedback)
            )
        messages.append({"role": "user", "content": user_content})
        return messages

    def run(self, query, context_blocks, memory_text="", feedback=None):
        messages = self._build_messages(query, context_blocks, memory_text, feedback)
        return chat_text(messages)

    def stream(self, query, context_blocks, memory_text="", feedback=None):
        """Generator yielding (full_answer, delta) pairs."""
        from app.llm.client import chat_stream
        messages = self._build_messages(query, context_blocks, memory_text, feedback)
        parts = []
        for delta in chat_stream(messages):
            parts.append(delta)
            yield "".join(parts), delta
        if not parts:
            yield "", ""
