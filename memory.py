# Agentic RAG - conversation memory.
# "Recent + relevant" memory backed by MongoDB. Conversations and messages
# live in MongoDB; this module re-exports the same API the agents and API layer
# already use, so nothing downstream needs to change.

from mongo import (
    create_conversation,
    conversation_owner,
    get_conversations,
    get_conversation_messages,
    delete_conversation,
    add_message,
    count_conversations,
    count_messages,
    get_recent,
    get_relevant,
    get_smart_context,
)
