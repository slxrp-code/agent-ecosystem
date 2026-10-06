import os, json, uuid
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import google.generativeai as genai
import psycopg2
import psycopg2.extras

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

genai.configure(api_key=os.environ.get("GOOGLE_API_KEY", ""))

# Ordered newest→oldest free Gemini models. App auto-falls back if one is deprecated.
GEMINI_MODELS = [
    "gemini-3.8-flash",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
]
_working_model = None  # cached so every call does not retry

def call_gemini(system: str, history: list, message: str) -> str:
    global _working_model
    # Try cached model first, then fall through the list
    ordered = ([_working_model] + [m for m in GEMINI_MODELS if m != _working_model]) if _working_model else GEMINI_MODELS
    last_err = None
    for model_name in ordered:
        try:
            model = genai.GenerativeModel(model_name=model_name, system_instruction=system)
            session = model.start_chat(history=history)
            reply = session.send_message(message).text
            _working_model = model_name  # lock in the one that worked
            return reply
        except Exception as e:
            err = str(e)
            if any(k in err.lower() for k in ("404", "not found", "deprecated", "no longer available", "does not exist")):
                last_err = err
                continue  # try next model
            raise  # non-model error (bad API key, quota, etc) — surface immediately
    raise HTTPException(500, f"No available Gemini model found. Last error: {last_err}")

def get_db():
    url = os.environ.get("DATABASE_URL", "")
    # psycopg2 requires postgresql://, Render provides postgres://
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    conn = psycopg2.connect(url)
    conn.autocommit = False
    return conn

def init_db():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS agents (
        id TEXT PRIMARY KEY, name TEXT NOT NULL, role TEXT NOT NULL,
        color TEXT DEFAULT '#3b82f6', system_prompt TEXT NOT NULL,
        capabilities TEXT DEFAULT '[]', created_at TEXT NOT NULL)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS messages (
        id TEXT PRIMARY KEY, agent_id TEXT NOT NULL, role TEXT NOT NULL,
        content TEXT NOT NULL, created_at TEXT NOT NULL)""")
    conn.commit(); cur.close(); conn.close()

init_db()

class AgentCreate(BaseModel):
    name: str; role: str
    color: Optional[str] = "#3b82f6"
    system_prompt: str
    capabilities: Optional[List[str]] = []

class AgentUpdate(BaseModel):
    name: Optional[str] = None; role: Optional[str] = None
    color: Optional[str] = None; system_prompt: Optional[str] = None
    capabilities: Optional[List[str]] = None

class ChatMessage(BaseModel):
    message: str

def get_all_agents():
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT * FROM agents ORDER BY created_at")
    rows = cur.fetchall()
    cur.close(); conn.close()
    result = []
    for r in rows:
        a = dict(r); a['capabilities'] = json.loads(a['capabilities']); result.append(a)
    return result

def build_orchestrator_prompt():
    agents = get_all_agents()
    agent_list = "\n".join([
        f"  • {a['name']} ({a['role']}): {', '.join(a['capabilities']) if a['capabilities'] else 'capabilities TBD'}"
        for a in agents
    ]) if agents else "  No agents deployed yet."
    return f"""You are the Main Orchestrator of a business ecosystem. You have an unwavering business-first mindset — every decision, recommendation, and action is evaluated through the lens of business value, ROI, and growth.

Your ecosystem currently contains:
{agent_list}

Your responsibilities:
- Understand each agent's capabilities as thoroughly as they do themselves
- Coordinate agents for tasks requiring collaboration
- Identify ecosystem gaps and recommend new agents when needed
- Monitor overall ecosystem performance and efficiency
- Provide strategic business direction and step in wherever needed

Think in terms of: execution, profitability, efficiency, and scalability. No fluff — only actionable intelligence."""

@app.get("/")
def root():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/agents")
def list_agents(): return get_all_agents()

@app.post("/agents")
def create_agent(data: AgentCreate):
    conn = get_db(); cur = conn.cursor(); aid = str(uuid.uuid4())
    cur.execute("INSERT INTO agents VALUES (%s,%s,%s,%s,%s,%s,%s)",
        (aid, data.name, data.role, data.color, data.system_prompt,
         json.dumps(data.capabilities), datetime.utcnow().isoformat()))
    conn.commit(); cur.close(); conn.close()
    return {"id": aid}

@app.put("/agents/{aid}")
def update_agent(aid: str, data: AgentUpdate):
    conn = get_db(); cur = conn.cursor(); fields, vals = [], []
    if data.name is not None: fields.append("name=%s"); vals.append(data.name)
    if data.role is not None: fields.append("role=%s"); vals.append(data.role)
    if data.color is not None: fields.append("color=%s"); vals.append(data.color)
    if data.system_prompt is not None: fields.append("system_prompt=%s"); vals.append(data.system_prompt)
    if data.capabilities is not None: fields.append("capabilities=%s"); vals.append(json.dumps(data.capabilities))
    if fields:
        vals.append(aid)
        cur.execute(f"UPDATE agents SET {','.join(fields)} WHERE id=%s", vals)
        conn.commit()
    cur.close(); conn.close(); return {"status": "updated"}

@app.delete("/agents/{aid}")
def delete_agent(aid: str):
    conn = get_db(); cur = conn.cursor()
    cur.execute("DELETE FROM agents WHERE id=%s", (aid,))
    cur.execute("DELETE FROM messages WHERE agent_id=%s", (aid,))
    conn.commit(); cur.close(); conn.close(); return {"status": "deleted"}

@app.get("/agents/{aid}/messages")
def get_messages(aid: str):
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT role, content, created_at FROM messages WHERE agent_id=%s ORDER BY created_at", (aid,))
    rows = cur.fetchall(); cur.close(); conn.close()
    return [dict(r) for r in rows]

@app.post("/agents/{aid}/chat")
def chat(aid: str, msg: ChatMessage):
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    if aid == "orchestrator":
        system = build_orchestrator_prompt()
    else:
        cur.execute("SELECT * FROM agents WHERE id=%s", (aid,))
        row = cur.fetchone()
        if not row: cur.close(); conn.close(); raise HTTPException(404, "Agent not found")
        a = dict(row); caps = json.loads(a['capabilities'])
        system = f"""You are {a['name']}, a specialized {a['role']} agent in a business ecosystem.

Your defined capabilities: {', '.join(caps) if caps else 'to be defined'}

Your instructions:
{a['system_prompt']}

Core directive: Business-first mindset at all times. Every response is practical, actionable, and oriented toward measurable outcomes. Be direct and results-focused."""

    cur.execute("SELECT role, content FROM messages WHERE agent_id=%s ORDER BY created_at", (aid,))
    history = cur.fetchall()
    gemini_history = [
        {"role": "model" if r["role"] == "assistant" else "user", "parts": [r["content"]]}
        for r in history
    ]

    try:
        reply = call_gemini(system, gemini_history, msg.message)
    except Exception as e:
        cur.close(); conn.close(); raise HTTPException(500, str(e))

    now = datetime.utcnow().isoformat()
    cur.execute("INSERT INTO messages VALUES (%s,%s,'user',%s,%s)", (str(uuid.uuid4()), aid, msg.message, now))
    cur.execute("INSERT INTO messages VALUES (%s,%s,'assistant',%s,%s)", (str(uuid.uuid4()), aid, reply, now))
    conn.commit(); cur.close(); conn.close()
    return {"response": reply}

@app.delete("/agents/{aid}/messages")
def clear_messages(aid: str):
    conn = get_db(); cur = conn.cursor()
    cur.execute("DELETE FROM messages WHERE agent_id=%s", (aid,))
    conn.commit(); cur.close(); conn.close(); return {"status": "cleared"}
