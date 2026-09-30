import os, json, uuid
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import anthropic
import sqlite3

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DB_PATH = "ecosystem.db"

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute("""CREATE TABLE IF NOT EXISTS agents (
        id TEXT PRIMARY KEY, name TEXT NOT NULL, role TEXT NOT NULL,
        color TEXT DEFAULT '#3b82f6', system_prompt TEXT NOT NULL,
        capabilities TEXT DEFAULT '[]', created_at TEXT NOT NULL)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS messages (
        id TEXT PRIMARY KEY, agent_id TEXT NOT NULL, role TEXT NOT NULL,
        content TEXT NOT NULL, created_at TEXT NOT NULL)""")
    conn.commit(); conn.close()

init_db()
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

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
    rows = conn.execute("SELECT * FROM agents ORDER BY created_at").fetchall()
    conn.close()
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
    conn = get_db(); aid = str(uuid.uuid4())
    conn.execute("INSERT INTO agents VALUES (?,?,?,?,?,?,?)",
        (aid, data.name, data.role, data.color, data.system_prompt,
         json.dumps(data.capabilities), datetime.utcnow().isoformat()))
    conn.commit(); conn.close()
    return {"id": aid}

@app.put("/agents/{aid}")
def update_agent(aid: str, data: AgentUpdate):
    conn = get_db(); fields, vals = [], []
    if data.name is not None: fields.append("name=?"); vals.append(data.name)
    if data.role is not None: fields.append("role=?"); vals.append(data.role)
    if data.color is not None: fields.append("color=?"); vals.append(data.color)
    if data.system_prompt is not None: fields.append("system_prompt=?"); vals.append(data.system_prompt)
    if data.capabilities is not None: fields.append("capabilities=?"); vals.append(json.dumps(data.capabilities))
    if fields:
        vals.append(aid)
        conn.execute(f"UPDATE agents SET {','.join(fields)} WHERE id=?", vals)
        conn.commit()
    conn.close(); return {"status": "updated"}

@app.delete("/agents/{aid}")
def delete_agent(aid: str):
    conn = get_db()
    conn.execute("DELETE FROM agents WHERE id=?", (aid,))
    conn.execute("DELETE FROM messages WHERE agent_id=?", (aid,))
    conn.commit(); conn.close(); return {"status": "deleted"}

@app.get("/agents/{aid}/messages")
def get_messages(aid: str):
    conn = get_db()
    rows = conn.execute("SELECT role, content, created_at FROM messages WHERE agent_id=? ORDER BY created_at", (aid,)).fetchall()
    conn.close(); return [dict(r) for r in rows]

@app.post("/agents/{aid}/chat")
def chat(aid: str, msg: ChatMessage):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    conn = get_db()
    if aid == "orchestrator":
        system = build_orchestrator_prompt()
    else:
        row = conn.execute("SELECT * FROM agents WHERE id=?", (aid,)).fetchone()
        if not row: conn.close(); raise HTTPException(404, "Agent not found")
        a = dict(row); caps = json.loads(a['capabilities'])
        system = f"""You are {a['name']}, a specialized {a['role']} agent in a business ecosystem.

Your defined capabilities: {', '.join(caps) if caps else 'to be defined'}

Your instructions:
{a['system_prompt']}

Core directive: Business-first mindset at all times. Every response is practical, actionable, and oriented toward measurable outcomes. Be direct and results-focused."""

    history = conn.execute("SELECT role, content FROM messages WHERE agent_id=? ORDER BY created_at", (aid,)).fetchall()
    messages = [{"role": r["role"], "content": r["content"]} for r in history]
    messages.append({"role": "user", "content": msg.message})

    try:
        resp = client.messages.create(
            model="claude-opus-4-5", max_tokens=2048, system=system, messages=messages)
        reply = resp.content[0].text
    except Exception as e:
        conn.close(); raise HTTPException(500, str(e))

    now = datetime.utcnow().isoformat()
    conn.execute("INSERT INTO messages VALUES (?,?,'user',?,?)", (str(uuid.uuid4()), aid, msg.message, now))
    conn.execute("INSERT INTO messages VALUES (?,?,'assistant',?,?)", (str(uuid.uuid4()), aid, reply, now))
    conn.commit(); conn.close()
    return {"response": reply}

@app.delete("/agents/{aid}/messages")
def clear_messages(aid: str):
    conn = get_db()
    conn.execute("DELETE FROM messages WHERE agent_id=?", (aid,))
    conn.commit(); conn.close(); return {"status": "cleared"}
