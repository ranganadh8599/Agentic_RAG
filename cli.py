# Agentic RAG - CLI.
# Commands:
#   ingest <path> [--title X]      ingest a file or a whole directory
#   ask "question" [--conv N]      one-shot RAG question
#   chat [--conv N]                interactive chat with the agents
#   stats                          show collection stats
#   reset                          drop all data (re-init schema)

import argparse
import sys

import db
import ingest
import memory
from agents import OrchestratorAgent


def cmd_ingest(args):
    db.init_db()
    docs, chunks = ingest.ingest_path(args.path, title=args.title,
                                      skip_duplicates=not args.force,
                                      collection=args.collection,
                                      update_existing=args.update,
                                      user_id=args.user)
    print(f"Done: {docs} document(s) ingested into table '{args.collection}', "
          f"{chunks} chunk(s) total.")


def cmd_ask(args):
    db.init_db()
    conv = args.conversation_id or memory.create_conversation(args.query[:60])
    agent = OrchestratorAgent()
    res = agent.run(args.query, conversation_id=conv, collection=args.collection)
    print(res["answer"])
    if res["sources"]:
        print("\nSources:")
        for s in res["sources"]:
            print(f"  [{s['citation']}] {s['title']} (doc {s['doc_id']})")
    if args.conversation_id is None:
        print(f"\n(conversation id: {conv})")


def cmd_chat(args):
    db.init_db()
    conv = args.conversation_id or memory.create_conversation("interactive chat")
    print("Agentic RAG chat. Type /quit to exit.\n")
    agent = OrchestratorAgent()
    while True:
        try:
            q = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not q:
            continue
        if q.lower() in ("/quit", "/exit", "exit"):
            break
        try:
            res = agent.run(q, conversation_id=conv, collection=args.collection)
            print("Agent>", res["answer"])
        except Exception as exc:  # noqa: BLE001
            print(f"Error: {exc}")


def cmd_rebuild_sparse(args):
    db.init_db()
    import sparse
    n = sparse.rebuild()
    print(f"Rebuilt sparse vectors + term stats for {n} chunk(s).")


def cmd_admin(args):
    db.init_db()
    import mongo
    mongo.init_db()
    u = mongo.set_admin(args.username, not args.remove)
    if not u:
        print(f"user not found: {args.username}")
        sys.exit(1)
    print(f"user '{u['username']}' is_admin={u['is_admin']}")


def cmd_stats(args):
    db.init_db()
    with db.get_conn().cursor() as cur:
        for table in ("documents", "chunks", "semantic_cache"):
            cur.execute(f"SELECT count(*) AS n FROM {table}")
            print(f"  {table:16} {cur.fetchone()['n']}")
    print("  users/chat      MongoDB (users, conversations, messages)")
    print(f"  pgvector          {db.USE_PGVECTOR}")


def cmd_reset(args):
    conn = db.get_conn()
    conn.execute(
        "DROP TABLE IF EXISTS semantic_cache, messages, conversations, chunks, documents CASCADE"
    )
    print("Dropped all tables. Run `ingest` to start fresh.")


def main():
    import logging_config
    logging_config.setup_logging()
    p = argparse.ArgumentParser(prog="agentic-rag", description="Multi-agent RAG over your documents")
    sub = p.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="ingest a file or directory")
    p_ingest.add_argument("path")
    p_ingest.add_argument("--title", default=None)
    p_ingest.add_argument("--collection", default="default",
                          help="table/collection to ingest into (auto-created if missing)")
    p_ingest.add_argument("--force", action="store_true",
                          help="re-ingest even if a doc with the same name exists")
    p_ingest.add_argument("--update", action="store_true",
                          help="delta-update an existing doc: reuse unchanged chunks, "
                               "embed only changed ones (no duplicates)")
    p_ingest.add_argument("--user", default=None,
                          help="owner user_id for the doc (used by metadata filtering)")
    p_ingest.set_defaults(func=cmd_ingest)

    p_ask = sub.add_parser("ask", help="ask a one-shot question")
    p_ask.add_argument("query")
    p_ask.add_argument("--conv", dest="conversation_id", type=int, default=None)
    p_ask.add_argument("--collection", default=None,
                       help="table/collection to search (default: all)")
    p_ask.set_defaults(func=cmd_ask)

    p_chat = sub.add_parser("chat", help="interactive chat")
    p_chat.add_argument("--conv", dest="conversation_id", type=int, default=None)
    p_chat.add_argument("--collection", default=None,
                        help="table/collection to search (default: all)")
    p_chat.set_defaults(func=cmd_chat)

    p_stats = sub.add_parser("stats", help="show stats")
    p_stats.set_defaults(func=cmd_stats)

    p_sparse = sub.add_parser("rebuild-sparse",
                              help="rebuild BM25 sparse vectors + term stats for all chunks")
    p_sparse.set_defaults(func=cmd_rebuild_sparse)

    p_admin = sub.add_parser("admin", help="grant/revoke the admin role for a user")
    p_admin.add_argument("username", help="username to mark/unmark as admin")
    p_admin.add_argument("--remove", action="store_true",
                         help="revoke admin instead of granting")
    p_admin.set_defaults(func=cmd_admin)

    p_reset = sub.add_parser("reset", help="drop all data")
    p_reset.set_defaults(func=cmd_reset)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
