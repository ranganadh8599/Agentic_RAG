"""Agentic RAG — end-to-end edge-case test suite.

Covers:
  A  fact/summary/vision/general/no-answer queries + ground truth (fixtures)
  B  collection isolation & cache behaviour
  C  ingestion edge cases (dedup, unsupported, weird filenames, empty files)
  D  API robustness (empty/missing messages, huge top_k, SSE stream, lists)
  E  registration & login (dup/short/blank username, 401s, case-insensitivity)
  F  session validation (valid/invalid/missing bearer token -> 401s)
  G  auth'd chat + conversation scoping + message persistence
  H  history isolation & ownership (403 cross-user, 404, delete flows)
  I  conversation memory persistence (multi-turn order, counts, isolation)
  J  password change (wrong current, short new, revocation, logout invalidation)
  K  health / Mongo integration (conversation+message counts, cross-login)

Every factual answer is checked against the real source documents (fixtures +
resume + ML/RAG PDFs). Run with the API server up:
    & .\\.venv\\Scripts\\python.exe tests\\e2e_test.py
"""
import io
import json
import os
import re
import sys
import tempfile
import urllib.error
import urllib.request
import uuid

BASE = "http://localhost:8000"
FIX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fixtures")

PASS = FAIL = SKIP = 0
FAILURES = []


def norm(s):
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def contains(text, *needles):
    t = norm(text)
    return [n for n in needles if norm(n) in t]


def report(name, ok, detail=""):
    global PASS, FAIL
    tag = "PASS" if ok else "FAIL"
    if ok:
        PASS += 1
    else:
        FAIL += 1
        FAILURES.append((name, detail))
    print(f"[{tag}] {name}" + (f"  -- {detail}" if detail else ""))


def post_json(path, payload):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


def post_raw(path, payload, expect_2xx=True):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:400]
        if expect_2xx:
            return e.code, {"error": body}
        return e.code, {"error": body}


def upload(path, filename, collection, data=None):
    boundary = "----e2e" + uuid.uuid4().hex
    buf = io.BytesIO()
    if data is None:
        data = open(path, "rb").read()
    def part(name, value, is_file=False, fname=None, ctype="application/octet-stream"):
        buf.write(f"--{boundary}\r\n".encode())
        if is_file:
            buf.write(f'Content-Disposition: form-data; name="{name}"; filename="{fname}"\r\n'.encode())
            buf.write(f"Content-Type: {ctype}\r\n\r\n".encode())
            buf.write(value)
        else:
            buf.write(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
            buf.write(value.encode())
        buf.write(b"\r\n")
    part("file", data, True, filename, "application/octet-stream")
    part("collection", collection)
    buf.write(f"--{boundary}--\r\n".encode())
    req = urllib.request.Request(BASE + "/ingest", data=buf.getvalue(), method="POST", headers={
        "Content-Type": f"multipart/form-data; boundary={boundary}"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode()[:300]}


def chat(query, collection=None, stream=False, conversation_id=None):
    payload = {"messages": [{"role": "user", "content": query}]}
    if collection is not None:
        payload["collection"] = collection
    if conversation_id is not None:
        payload["conversation_id"] = conversation_id
    if stream:
        payload["stream"] = True
        return post_json("/v1/chat/completions", payload)
    return post_json("/v1/chat/completions", payload)


def answer_of(resp):
    return resp["choices"][0]["message"]["content"]


def cited_numbers(answer):
    return sorted({int(x) for x in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", answer or "")
                   for x in re.findall(r"\d+", x)})


# ---------------------------------------------------------------------------
# Auth / users / history helpers (SECTIONS E+)
# ---------------------------------------------------------------------------

def _req(method, path, payload=None, token=None, timeout=120):
    """HTTP request with optional JSON body + Bearer token. Returns (status, body)."""
    headers = {}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(BASE + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode()
            return r.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            body = json.loads(raw)
        except Exception:
            body = {"error": raw[:300]}
        return e.code, body


def api_get(path, token=None):
    return _req("GET", path, token=token)


def api_post(path, payload, token=None):
    return _req("POST", path, payload, token=token)


def api_delete(path, token=None):
    return _req("DELETE", path, token=token)


def register(username, password, display_name=None):
    return api_post("/api/register", {"username": username, "password": password,
                                      "display_name": display_name or username})


def login(username, password):
    return api_post("/api/login", {"username": username, "password": password})


def auth_chat(query, token, conversation_id=None, collection=None):
    """Chat with an optional Bearer token (returns (status, body))."""
    payload = {"messages": [{"role": "user", "content": query}]}
    if conversation_id is not None:
        payload["conversation_id"] = conversation_id
    if collection is not None:
        payload["collection"] = collection
    return api_post("/v1/chat/completions", payload, token=token)


# =====================================================================
print("\n" + "=" * 78)
print("SECTION A — QUERY / ANSWER + GROUND-TRUTH VERIFICATION (default table)")
print("=" * 78)

# A1 fact: Acme revenue
r = chat("What total revenue did Acme Analytics report in 2024?")
a = answer_of(r)
got = contains(a, "2,400,000", "2.4 million", "$2,400,000")
ok = len(got) > 0 and any("report.pdf" in (s["title"] or "") for s in r["sources"])
report("A1 fact: Acme 2024 revenue=$2.4M (report.pdf)", ok, f"answer={a[:120]!r}")

# A2 breakdown
r = chat("What is the revenue breakdown by product in the annual report?")
a = answer_of(r)
got = contains(a, "1,600,000", "1.6 million", "600,000", "200,000")
report("A2 fact: revenue breakdown 1.6M/600K/200K", len(got) >= 2, f"answer={a[:120]!r}")

# A3 xlsx
r = chat("What are the Q1 sales for the North region in sales.xlsx?")
a = answer_of(r)
report("A3 xlsx: North Q1 = 100", contains(a, "100") and any("sales.xlsx" in (s["title"] or "") for s in r["sources"]), f"answer={a[:120]!r}")

# A4 pptx
r = chat("What phases are in the product roadmap?")
a = answer_of(r)
ok = contains(a, "phase 1", "phase 2", "ai", "mobile") and any("roadmap.pptx" in (s["title"] or "") for s in r["sources"])
report("A4 pptx: roadmap phases", ok, f"answer={a[:120]!r}")

# A5 summary
r = chat("Summarize the Acme Analytics annual report.")
a = answer_of(r)
report("A5 summary: covers report.pdf", any("report.pdf" in (s["title"] or "") for s in r["sources"]) and contains(a, "acme"), f"answer={a[:120]!r}")

# A6 vision: querying by filename must retrieve the image doc
r = chat("Describe what is shown in chart.png.")
a = answer_of(r)
ok = any("chart.png" in (s["title"] or "") for s in r["sources"])
report("A6 vision: chart.png retrieved by filename", ok, f"answer={a[:120]!r} src={[s['title'] for s in r['sources']]}")

# A7 no-answer in docs (should NOT fabricate a doc citation)
r = chat("What color is the CEO's car according to the documents?")
a = answer_of(r)
mentions_unknown = any(x in a.lower() for x in ["not ", "no ", "does not", "information", "unable"])
report("A7 no-answer: refuses without hallucination", mentions_unknown, f"answer={a[:140]!r}")

# A8 general knowledge (router=general). Must NOT fabricate a doc citation.
r = chat("What is the capital of France?")
a = answer_of(r)
ok = contains(a, "paris") and not (r["sources"])
report("A8 general: answers Paris, no fabricated doc sources", ok, f"answer={a[:120]!r} src={[s['title'] for s in r['sources']]}")

# A9 ambiguous pronoun (no antecedent) — must not fabricate
r = chat("Where does he work now?")
a = answer_of(r)
mentions_unknown = any(x in a.lower() for x in ["not ", "no ", "does not", "information", "which person", "who", "unclear"])
report("A9 ambiguous 'he': no fabrication", mentions_unknown, f"answer={a[:140]!r}")

# A10 citation integrity: every [n] maps to a returned source, and sources == distinct [n]
r = chat("What is AcmeInsight and what did Acme achieve in sustainability?")
a = answer_of(r)
nums = cited_numbers(a)
src_nums = sorted({int(s["citation"]) for s in r["sources"]})
ok = (nums == src_nums) and len(nums) >= 1
report("A10 citation integrity: [n] <=> source cards", ok, f"cited={nums} src={src_nums}")

# A11 every cited source's snippet should overlap the answer (spot check)
r = chat("What is AcmeInsight?")
a = answer_of(r)
bad_snippets = []
for s in r["sources"]:
    overlap = sum(1 for w in re.findall(r"[a-zA-Z]{3,}", norm(s["snippet"]))
                  if w in {x for x in re.findall(r"[a-zA-Z]{3,}", norm(a))})
    if overlap == 0:
        bad_snippets.append(s["title"])
report("A11 snippet overlaps answer", not bad_snippets, f"no-overlap={bad_snippets}")

# A12 greeting short-circuit: replies from the LLM WITHOUT RAG (no sources)
for g in ("hello", "hey", "whatsup"):
    rg = chat(g)
    ok = rg.get("type") == "greeting" and not rg.get("sources") and answer_of(rg) != ""
    report(f"A12 greeting '{g}' -> greeting, no RAG", ok, f"type={rg.get('type')} src={len(rg.get('sources') or [])}")

# A13 greeting + a real question must STILL do RAG (not treated as pure greeting)
rg = chat("hi, what is in the report?")
report("A13 greeting+question still does RAG", rg.get("type") != "greeting" and len(rg.get("sources") or []) >= 1,
       f"type={rg.get('type')} src={len(rg.get('sources') or [])}")


# =====================================================================
print("\n" + "=" * 78)
print("SECTION B — COLLECTIONS / ISOLATION")
print("=" * 78)

# B1 isolation: same Q, different tables
q = "What revenue figure is mentioned in the business review document?"
r_hr = chat(q, collection="hr")
a_hr = answer_of(r_hr)
r_res = chat(q, collection="resume")
a_res = answer_of(r_res)
ok_hr = contains(a_hr, "1.2") and all("business.docx" in (s["title"] or "") for s in r_hr["sources"])
ok_res = all("business" not in (s["title"] or "") for s in r_res["sources"])
report("B1 hr: business.docx only, $1.2M", ok_hr, f"src={[s['title'] for s in r_hr['sources']]}")
report("B1 resume: no business.docx leak", ok_res, f"src={[s['title'] for s in r_res['sources']]}")

# B2 nonexistent collection — must not 500 (falls back to All)
status, body = post_raw("/v1/chat/completions",
                        {"messages": [{"role": "user", "content": "what is supervised learning?"}],
                         "collection": "zz_no_such_table_xyz"})
report("B2 nonexistent collection: no 500", status == 200, f"status={status} {body.get('error','')[:120]}")

# B3 All collections (collection=null)
r = chat("what is supervised learning?")
report("B3 All collections works", len(r["sources"]) >= 1, f"src={[s['title'] for s in r['sources'][:3]]}")

# B4 cache isolation across collections
r1 = chat("What revenue is in the business review?", collection="hr")
r2 = chat("What revenue is in the business review?", collection="resume")
a1, a2 = answer_of(r1), answer_of(r2)
different = norm(a1) != norm(a2)
report("B4 cross-table cache isolation (hr vs resume differ)", different, f"hr={a1[:60]!r} resume={a2[:60]!r}")

# B5 repeat query in same table → served (should be fast / identical)
import time
t0 = time.time()
r3 = chat("What revenue is in the business review?", collection="hr")
t1 = time.time()
report("B5 repeat query same table works", answer_of(r3) != "" and (t1 - t0) < 30, f"time={t1-t0:.1f}s")


# =====================================================================
print("\n" + "=" * 78)
print("SECTION C — INGESTION EDGE CASES")
print("=" * 78)

tmp = tempfile.mkdtemp(prefix="e2e_ingest_")
tbl_dup = "e2e_dup_" + uuid.uuid4().hex[:6]

# C1 duplicate file same table -> skipped
st, j1 = upload(os.path.join(FIX, "notes.txt"), "notes.txt", tbl_dup)
st, j2 = upload(os.path.join(FIX, "notes.txt"), "notes.txt", tbl_dup)
report("C1 duplicate same table skipped", st == 200 and j1.get("skipped") is False and j2.get("skipped") is True,
       f"first={j1} second={j2}")

# C2 same file different table -> NOT skipped
tbl2 = "e2e_dup2_" + uuid.uuid4().hex[:6]
st, j3 = upload(os.path.join(FIX, "notes.txt"), "notes.txt", tbl2)
report("C2 same file different table not skipped", st == 200 and j3.get("skipped") is False, f"resp={j3}")

# C3 unsupported extension -> graceful (no crash, error message)
zip_path = os.path.join(tmp, "evil.zip")
with open(zip_path, "wb") as f:
    f.write(b"PK\x05\x06" + b"\x00" * 18)
st, j4 = upload(zip_path, "evil.zip", tbl_dup)
report("C3 unsupported .zip handled gracefully", st in (200, 400, 415, 500) and j4.get("error") is not None,
       f"status={st} resp={j4}")

# C4 filename with spaces + unicode
weird = os.path.join(tmp, "my notes ü § .txt")
with open(weird, "w", encoding="utf-8") as f:
    f.write("Edge test: this file has a weird name with spaces and unicode.")
st, j5 = upload(weird, os.path.basename(weird), tbl_dup)
report("C4 filename spaces+unicode", st == 200 and j5.get("chunks", 0) >= 1, f"status={st} resp={j5}")

# C5 empty file -> graceful, no doc created, clear note (NOT a misleading "skipped")
empty = os.path.join(tmp, "empty.txt")
open(empty, "w").close()
st, j6 = upload(empty, "empty.txt", tbl_dup)
ok = (st == 200 and j6.get("chunks") == 0 and j6.get("skipped") is False
      and j6.get("note") is not None and j6.get("document_id") == 0)
report("C5 empty file graceful (no doc, note)", ok, f"status={st} resp={j6}")

# C6 malformed pdf -> graceful
bad_pdf = os.path.join(tmp, "bad.pdf")
with open(bad_pdf, "wb") as f:
    f.write(b"%PDF-1.4\nthis is not a real pdf at all" + b"\x00" * 50)
st, j7 = upload(bad_pdf, "bad.pdf", tbl_dup)
report("C6 malformed pdf graceful", st in (200, 400, 415, 500), f"status={st} resp={j7}")


# =====================================================================
print("\n" + "=" * 78)
print("SECTION D — API ROBUSTNESS")
print("=" * 78)

# D1 empty messages -> 4xx, not 500
status, body = post_raw("/v1/chat/completions", {"messages": []}, expect_2xx=False)
report("D1 empty messages -> 4xx not 500", 400 <= status < 500, f"status={status}")

# D2 no messages key
status, body = post_raw("/v1/chat/completions", {}, expect_2xx=False)
report("D2 missing messages -> 4xx", 400 <= status < 500, f"status={status}")

# D3 huge top_k
status, body = post_raw("/v1/chat/completions",
                        {"messages": [{"role": "user", "content": "what is in the report?"}],
                         "top_k": 9999})
report("D3 top_k=9999 no crash", status == 200, f"status={status} {body.get('error','')[:100]}")

# D4 stream returns SSE with sources + [DONE]
req = urllib.request.Request(BASE + "/v1/chat/completions",
                             data=json.dumps({"messages": [{"role": "user", "content": "what revenue is in the report?"}],
                                              "collection": "default", "stream": True}).encode(),
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=120) as r:
    raw = r.read().decode()
ok = "data: [DONE]" in raw and '"sources"' in raw
report("D4 stream: SSE + sources + [DONE]", ok, f"len={len(raw)}")

# D5 /collections returns names + counts (GET)
import urllib.request as _ur
with _ur.urlopen(BASE + "/collections", timeout=30) as _r:
    cols = json.loads(_r.read().decode())
names = [c["name"] for c in cols]
report("D5 /collections lists tables", tbl_dup in names and "default" in names, f"count={len(names)}")

# D6 /documents?collection= filter correctness (GET)
with _ur.urlopen(BASE + f"/documents?collection={tbl_dup}", timeout=30) as _r:
    docs = json.loads(_r.read().decode())
ok = all((d.get("collection") == tbl_dup) for d in docs) and len(docs) >= 1
report("D6 /documents?collection filters", ok, f"docs={[(d['title'], d.get('collection')) for d in docs]}")

# D7 health
with urllib.request.urlopen(BASE + "/health", timeout=10) as r:
    ok = r.status == 200
report("D7 /health", ok)


# =====================================================================
print("\n" + "=" * 78)
print("SECTION E — REGISTRATION & LOGIN")
print("=" * 78)

uid_e = "e_" + uuid.uuid4().hex[:6]

# E1 register new user
st, u = register(uid_e, "pass1234", "E2E User")
ok = st == 200 and u.get("username") == uid_e and u.get("display_name") == "E2E User" and bool(u.get("id"))
report("E1 register returns user", ok, f"st={st} u={u}")

# E2 duplicate username -> 409
st, u2 = register(uid_e, "pass9999")
report("E2 duplicate username 409", st == 409, f"st={st} body={u2}")

# E3 too-short password -> 409
st, u3 = register("e3_" + uuid.uuid4().hex[:6], "abc")
report("E3 short password 409", st == 409, f"st={st} body={u3}")

# E4 blank username -> 409 (not 500)
st, u4 = register("   ", "pass1234")
report("E4 blank username 409", st == 409, f"st={st}")

# E5 missing fields -> 422
st, b = _req("POST", "/api/register", {"username": "e5_" + uuid.uuid4().hex[:6]})
report("E5 register missing fields 422", st == 422, f"st={st}")

# E6 login wrong password -> 401
st, b = login(uid_e, "wrongpass")
report("E6 wrong password 401", st == 401, f"st={st}")

# E7 login unknown user -> 401
st, b = login("nobody_" + uuid.uuid4().hex[:6], "whatever1")
report("E7 unknown user 401", st == 401, f"st={st}")

# E8 login missing fields -> 422
st, b = _req("POST", "/api/login", {"username": uid_e})
report("E8 login missing fields 422", st == 422, f"st={st}")

# E9 login ok -> {user, token}
st, lg = login(uid_e, "pass1234")
tok_e = (lg or {}).get("token")
ok = st == 200 and bool(tok_e) and lg.get("user", {}).get("username") == uid_e
report("E9 login returns user+token", ok, f"st={st} user={lg.get('user')}")

# E10 login is case-insensitive on username
st, lg2 = login(uid_e.upper(), "pass1234")
report("E10 login case-insensitive", st == 200, f"st={st}")


# =====================================================================
print("\n" + "=" * 78)
print("SECTION F — SESSION VALIDATION (401s)")
print("=" * 78)

# F1 /api/me with valid token
st, me = api_get("/api/me", token=tok_e)
report("F1 /api/me valid token", st == 200 and me.get("username") == uid_e, f"st={st}")

# F2 /api/me with no token -> 401
st, _ = api_get("/api/me")
report("F2 /api/me no token 401", st == 401, f"st={st}")

# F3 /api/me with garbage token -> 401
st, _ = api_get("/api/me", token="garbage.token.here")
report("F3 /api/me invalid token 401", st == 401, f"st={st}")

# F4 /conversations with no token -> 401
st, _ = api_get("/conversations")
report("F4 /conversations no token 401", st == 401, f"st={st}")

# F5 /conversations with garbage token -> 401
st, _ = api_get("/conversations", token="nope")
report("F5 /conversations bad token 401", st == 401, f"st={st}")

# F6 anonymous chat (no token) still works
st, r = auth_chat("What is the capital of France?", None)
report("F6 anonymous chat works (200)", st == 200 and answer_of(r) != "", f"st={st}")


# =====================================================================
print("\n" + "=" * 78)
print("SECTION G — AUTH'D CHAT + CONVERSATION SCOPING")
print("=" * 78)

# G1 chat with token -> returns conversation_id
st, r = auth_chat("What is AcmeInsight?", tok_e)
cid_e = (r or {}).get("conversation_id")
report("G1 auth chat returns conversation_id", st == 200 and bool(cid_e), f"st={st} cid={cid_e}")

# G2 conversation appears in the user's history
st, convs = api_get("/conversations", token=tok_e)
ids = [c["id"] for c in convs]
report("G2 history contains the conversation", st == 200 and cid_e in ids, f"st={st} n={len(ids)}")

# G3 title = first user message (truncated to 60 chars)
c = next((c for c in convs if c["id"] == cid_e), None)
report("G3 title from first user message", c is not None and c.get("title", "").startswith("What is AcmeInsight?"),
       f"title={c and c.get('title')}")

# G4 history message count = 2 after one turn (user + assistant)
report("G4 history message count = 2", c is not None and c.get("messages") == 2, f"msgs={c and c.get('messages')}")

# G5 GET /conversations/{id} returns user + assistant messages
st, detail = api_get(f"/conversations/{cid_e}", token=tok_e)
roles = [m["role"] for m in (detail or {}).get("messages", [])]
report("G5 conversation messages user+assistant", st == 200 and roles == ["user", "assistant"], f"roles={roles}")

# G6 assistant answer is persisted (non-empty)
content = [m["content"] for m in (detail or {}).get("messages", [])]
report("G6 assistant answer persisted", st == 200 and bool(content[1].strip()), f"len={len(content[1])}")

# G7 grounded sources are persisted with the assistant message (so citations and
# source cards survive switching chats / reloading the page).
asst = next((m for m in detail["messages"] if m["role"] == "assistant"), None)
srcs = (asst or {}).get("sources") or []
src_nums = sorted({int(s["citation"]) for s in srcs if s.get("citation") is not None})
cited = cited_numbers(asst["content"]) if asst else []
report("G7 sources persisted with assistant message", st == 200 and len(srcs) >= 1 and src_nums == cited,
       f"src={src_nums} cited={cited}")


# =====================================================================
print("\n" + "=" * 78)
print("SECTION H — HISTORY ISOLATION & OWNERSHIP (403/404/delete)")
print("=" * 78)

uid_h = "h_" + uuid.uuid4().hex[:6]
register(uid_h, "pass5678")
_, lg_h = login(uid_h, "pass5678")
tok_h = lg_h["token"]

# H1 user B cannot READ A's conversation -> 403
st, _ = api_get(f"/conversations/{cid_e}", token=tok_h)
report("H1 cross-user read 403", st == 403, f"st={st}")

# H2 user B cannot DELETE A's conversation -> 403
st, _ = api_delete(f"/conversations/{cid_e}", token=tok_h)
report("H2 cross-user delete 403", st == 403, f"st={st}")

# H3 user B's history does not contain A's conversation
st, convs_h = api_get("/conversations", token=tok_h)
report("H3 user B sees only own history", st == 200 and all(c["id"] != cid_e for c in convs_h), f"st={st}")

# H4 user B chats -> own conversation, fully isolated from A
st, rb = auth_chat("What is the capital of France?", tok_h)
cid_h = rb["conversation_id"]
st, convs_h = api_get("/conversations", token=tok_h)
ids_h = [c["id"] for c in convs_h]
report("H4 user B own conversation isolated", st == 200 and cid_h in ids_h and cid_e not in ids_h, f"st={st}")

# H5 user B resumes own conversation -> same conversation_id (no new conv)
st, rb2 = auth_chat("What is the capital of Germany?", tok_h, conversation_id=cid_h)
report("H5 resume own conversation", st == 200 and rb2.get("conversation_id") == cid_h, f"st={st}")

# H6 user B tries to resume A's conversation -> 403
st, _ = auth_chat("hi there", tok_h, conversation_id=cid_e)
report("H6 resume other's conversation 403", st == 403, f"st={st}")

# H7 resume non-existent conversation -> 404
st, _ = auth_chat("hi there", tok_e, conversation_id="000000000000000000000000")
report("H7 resume nonexistent conversation 404", st == 404, f"st={st}")

# H8-H11 delete tests use a dedicated conversation so later memory tests can reuse cid_h
st, rd = auth_chat("What is 7 times 8?", tok_h)
cid_del = rd["conversation_id"]
st, b = api_delete(f"/conversations/{cid_del}", token=tok_h)
report("H8 delete own conversation ok", st == 200 and b.get("deleted") == cid_del, f"st={st} b={b}")

st, _ = api_get(f"/conversations/{cid_del}", token=tok_h)
report("H9 deleted conversation 404 on read", st == 404, f"st={st}")

st, convs_h = api_get("/conversations", token=tok_h)
report("H10 deleted conversation gone from history", st == 200 and cid_del not in [c["id"] for c in convs_h], f"st={st}")

st, _ = api_delete("/conversations/000000000000000000000000", token=tok_h)
report("H11 delete nonexistent 404", st == 404, f"st={st}")


# =====================================================================
print("\n" + "=" * 78)
print("SECTION I — CONVERSATION MEMORY PERSISTENCE (multi-turn)")
print("=" * 78)

# I1 two more turns in the SAME conversation (G1 made 1 turn) -> 6 messages total, correct role order
st, _ = auth_chat("What is 2+2?", tok_e, conversation_id=cid_e)
st, _ = auth_chat("What is the capital of Germany?", tok_e, conversation_id=cid_e)
st, detail = api_get(f"/conversations/{cid_e}", token=tok_e)
msgs = [m["content"] for m in detail["messages"]]
roles = [m["role"] for m in detail["messages"]]
EXP_ROLES = ["user", "assistant"] * 3
report("I1 multi-turn persists 6 messages (3 turns)", st == 200 and len(msgs) == 6 and roles == EXP_ROLES,
       f"n={len(msgs)} roles={roles}")

# I2 all three user messages preserved in order
report("I2 user messages preserved in order",
       msgs[0].startswith("What is AcmeInsight?") and msgs[2].startswith("What is 2+2?")
       and msgs[4].startswith("What is the capital of Germany?"),
       f"m0={msgs[0][:24]!r} m2={msgs[2][:24]!r} m4={msgs[4][:24]!r}")

# I3 every assistant answer is non-empty
report("I3 assistant answers non-empty all turns",
       all(bool(msgs[i].strip()) for i in (1, 3, 5)), "")

# I4 history message count reflects 6 messages
st, convs = api_get("/conversations", token=tok_e)
c = next((c for c in convs if c["id"] == cid_e), None)
report("I4 history shows 6 msgs", c is not None and c.get("messages") == 6, f"msgs={c and c.get('messages')}")

# I5 preview = first user message
report("I5 preview = first user message", c is not None and c.get("preview", "").startswith("What is AcmeInsight?"),
       f"preview={c and c.get('preview')[:30]!r}")

# I6 per-conversation isolation: B's conversation has exactly its own 2 turns
st, db_ = api_get(f"/conversations/{cid_h}", token=tok_h)
roles_h = [m["role"] for m in db_["messages"]]
report("I6 per-conversation isolation (B own messages only)", roles_h == ["user", "assistant", "user", "assistant"],
       f"roles_h={roles_h}")

# I7 no anonymous conversations leak into a user's history: E created exactly ONE
# conversation (cid_e); if anonymous chats (sections A-F) leaked, E's list would
# contain far more entries.
st, convs = api_get("/conversations", token=tok_e)
ids_e = [c["id"] for c in convs]
report("I7 no anonymous convs leak into user history", st == 200 and ids_e == [cid_e], f"ids={ids_e}")


# =====================================================================
print("\n" + "=" * 78)
print("SECTION J — PASSWORD CHANGE & LOGOUT")
print("=" * 78)

uid_j = "j_" + uuid.uuid4().hex[:6]
register(uid_j, "origpass1")
_, lg_j = login(uid_j, "origpass1")
tok_j = lg_j["token"]
# second session created BEFORE the password change (for revocation check)
_, lg_j2 = login(uid_j, "origpass1")
tok_j2 = lg_j2["token"]

# J1 password change without token -> 401
st, _ = api_post("/api/password", {"current_password": "origpass1", "new_password": "newpass99"})
report("J1 change password no token 401", st == 401, f"st={st}")

# J2 wrong current password -> 400
st, b = api_post("/api/password", {"current_password": "wrong1", "new_password": "newpass99"}, token=tok_j)
report("J2 wrong current password 400", st == 400, f"st={st} body={b}")

# J3 too-short new password -> 400
st, b = api_post("/api/password", {"current_password": "origpass1", "new_password": "ab"}, token=tok_j)
report("J3 short new password 400", st == 400, f"st={st} body={b}")

# J4 successful change -> 200
st, b = api_post("/api/password", {"current_password": "origpass1", "new_password": "newpass99"}, token=tok_j)
report("J4 password change ok", st == 200 and b.get("ok") is True, f"st={st} body={b}")

# J5 old password rejected after change
st, _ = login(uid_j, "origpass1")
report("J5 old password rejected", st == 401, f"st={st}")

# J6 new password works
st, lg_j3 = login(uid_j, "newpass99")
report("J6 new password works", st == 200 and bool(lg_j3.get("token")), f"st={st}")

# J7 password change revokes OTHER sessions (created before the change)
st, _ = api_get("/api/me", token=tok_j2)
report("J7 other session revoked after change", st == 401, f"st={st}")

# J8 the session used to change the password survives
st, _ = api_get("/api/me", token=tok_j)
report("J8 current session survives change", st == 200, f"st={st}")

# J9 logout invalidates the token
_, lg_j4 = login(uid_j, "newpass99")
tok_j4 = lg_j4["token"]
st, _ = api_post("/api/logout", {}, token=tok_j4)
st2, _ = api_get("/api/me", token=tok_j4)
report("J9 logout invalidates token", st == 200 and st2 == 401, f"logout={st} me_after={st2}")

# J10 logout without a token is harmless
st, _ = api_post("/api/logout", {})
report("J10 logout no token harmless", st == 200, f"st={st}")


# =====================================================================
print("\n" + "=" * 78)
print("SECTION K — HEALTH / MONGO INTEGRATION")
print("=" * 78)

# K1 /health reports Mongo conversation/message counts (not the dropped Postgres tables)
with urllib.request.urlopen(BASE + "/health", timeout=10) as r:
    h = json.loads(r.read().decode())
report("K1 /health includes mongo counts", "conversations" in h and "messages" in h and h.get("status") == "ok",
       f"keys={list(h.keys())}")

# K2 messages were actually persisted for our auth'd conversations (mongo-level integrity)
_, lg_e = login(uid_e, "pass1234")
tok_e2 = lg_e["token"]
st, convs = api_get("/conversations", token=tok_e2)
total_msgs = sum(c.get("messages", 0) for c in convs if c["id"] == cid_e)
report("K2 persisted messages survive a new login", st == 200 and total_msgs == 6, f"st={st} msgs={total_msgs}")


# =====================================================================
print("\n" + "=" * 78)
print("SUMMARY")
print("=" * 78)
print(f"PASS={PASS}  FAIL={FAIL}")
if FAILURES:
    print("\nFAILED CASES:")
    for name, detail in FAILURES:
        print(f"  - {name}: {detail}")
sys.exit(0 if FAIL == 0 else 1)
