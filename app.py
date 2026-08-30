"""
Lyceum AI - A Personal AI Tutor (v5)
Features: web search tool, persistent memory (SQLite), spaced repetition,
photo/image understanding, voice input/output, multilingual, math rendering.

Built by Khursheed - Computer Science undergraduate, Iqra University, Karachi.
"""

import os
import json
import uuid
import base64
from datetime import datetime, timedelta
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from groq import Groq
from ddgs import DDGS

# ---------------------------------------------------------
# 1. SETUP
# ---------------------------------------------------------

API_KEY = os.environ.get("GROQ_API_KEY", "")
client = Groq(api_key=API_KEY)
MODEL = "openai/gpt-oss-120b"
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

app = FastAPI(title="Lyceum AI - Study Buddy")

REVIEW_INTERVAL_DAYS = 3  # spaced repetition: recap a topic 3 days after learning it
MAX_HISTORY_MESSAGES = 12


# ---------------------------------------------------------
# 2. IN-MEMORY STORAGE (simple, no database setup needed for deployment)
# NOTE: this resets whenever the server restarts. For a class assignment
# this is fine. For real production use, swap this for a real database.
# ---------------------------------------------------------

SESSIONS: dict[str, list] = {}          # {session_id: [ {role, content}, ... ]}
LEARNED_TOPICS: dict[str, list] = {}    # {session_id: [ {id, topic, learned_at, last_recap_at}, ... ]}
_topic_id_counter = 0


def load_history(session_id: str) -> list:
    return SESSIONS.get(session_id, [])


def save_message(session_id: str, role: str, content: str):
    SESSIONS.setdefault(session_id, []).append({"role": role, "content": content})


def save_learned_topic(session_id: str, topic: str):
    global _topic_id_counter
    _topic_id_counter += 1
    LEARNED_TOPICS.setdefault(session_id, []).append({
        "id": _topic_id_counter,
        "topic": topic[:120],
        "learned_at": datetime.utcnow(),
        "last_recap_at": None,
    })


def get_due_recaps(session_id: str) -> list:
    """Topics learned >= REVIEW_INTERVAL_DAYS ago that haven't been recapped since."""
    cutoff = datetime.utcnow() - timedelta(days=REVIEW_INTERVAL_DAYS)
    due = []
    for t in LEARNED_TOPICS.get(session_id, []):
        if t["learned_at"] <= cutoff and (t["last_recap_at"] is None or t["last_recap_at"] <= cutoff):
            due.append({"id": t["id"], "topic": t["topic"], "learned_at": t["learned_at"].isoformat()})
    return due[:5]


def mark_recapped(topic_id: int):
    for topics in LEARNED_TOPICS.values():
        for t in topics:
            if t["id"] == topic_id:
                t["last_recap_at"] = datetime.utcnow()
                return


# ---------------------------------------------------------
# 3. THE TOOL: web search
# ---------------------------------------------------------

def web_search(query: str, max_results: int = 4) -> str:
    try:
        results = DDGS().text(query, max_results=max_results)
        if not results:
            return "No search results found."
        combined = ""
        for r in results:
            combined += f"- {r.get('title', '')}: {r.get('body', '')}\n"
        return combined
    except Exception as e:
        return f"Search failed: {e}"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the internet for current information on a topic. Use this "
                "when the student asks about a NEW topic you need facts for, wants "
                "a quiz, or wants a summary. Do NOT use this for simple follow-ups "
                "like 'explain more', 'give another example', or 'I don't "
                "understand' -- for those, use the conversation history you already have."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "The search query"}},
                "required": ["query"],
            },
        },
    }
]


# ---------------------------------------------------------
# 4. SYSTEM PROMPT
# ---------------------------------------------------------

SYSTEM_PROMPT = """You are "Lyceum AI," a compassionate, brilliant, and endlessly patient
personal tutor. You can teach ANY subject a student brings to you -- math, science,
languages, history, literature, computer science, or anything else -- at whatever level
the student needs, from a curious beginner to an advanced learner.

CORE BEHAVIOR RULES:

1. LANGUAGE SENSITIVITY:
   - Detect the student's language automatically: English, Roman Urdu, Urdu script,
     Sindhi script, or any other language the student writes in.
   - ALWAYS reply in the exact language or mix the student used. Keep vocabulary simple
     and accessible -- avoid heavy or archaic words.

2. NATURAL TEACHING FLOW (default for concept questions):
   - Explain the concept in 2-3 very simple sentences (Explain Like I'm 5) -- write this
     as normal flowing text, never labeled as "Step 1" or similar.
   - Weave in a relatable real-world example the student can picture, as part of the
     natural explanation, not as a separately labeled section.
   - End with ONE short, engaging follow-up question to check understanding.
   - NEVER write "Step 1", "Step 2", numbered headers, or any meta-labels describing
     your own structure. Just teach naturally, like a real teacher speaking.

3. MATHEMATICS MODE (a key strength):
   - When solving or explaining a math problem, ALWAYS show FULL step-by-step working,
     never skip steps, and write all formulas and equations using LaTeX notation
     wrapped in $ for inline math or $$ for block/display equations (e.g. $x^2 + 2x + 1$
     or $$\frac{-b \pm \sqrt{b^2-4ac}}{2a}$$). The webpage renders LaTeX automatically,
     so always use it for any equation, fraction, exponent, or mathematical symbol
     instead of plain text approximations.
   - After solving, briefly explain the REASONING behind each step, not just the
     mechanical calculation, so the student learns the method, not just the answer.

4. LANGUAGE-LEARNING MODE (when practicing a language, asking for grammar help, or
   saying "correct my English/Urdu/etc."):
   - Gently correct mistakes without being discouraging -- show the corrected sentence,
     then briefly explain WHY in simple words.
   - Introduce 1-2 new useful vocabulary words naturally, with a simple example sentence.

5. ABSOLUTE RESTRAINTS:
   - NEVER give direct homework answers immediately -- guide step-by-step instead
     (Socratic method: hints, not full answers) UNLESS the student is asking you to
     check/solve a math problem for learning purposes, where full worked solutions ARE
     appropriate since seeing the full method is how math is learned.
   - If the student says "I don't understand," switch to a COMPLETELY different analogy.
     Do not search the web again for this -- use conversation context you already have.

6. SPECIAL MODES (detect intent from the message):
   - QUIZ requested: generate 3-5 multiple choice questions with the correct answer
     marked at the end.
   - SUMMARY requested: condense into simple bullet points.
   - Otherwise: use the natural teaching flow.

7. CONVERSATION CONTINUITY:
   - You have the recent conversation history -- use it for follow-ups like "explain
     more" or "give another example" instead of starting over.

8. TOOL USE:
   - You have a web_search tool. Use it for NEW topics needing current facts. Skip it
     for simple follow-ups answerable from conversation history.

9. ABOUT YOURSELF:
   - If asked who made you, who developed you, or similar questions, answer warmly:
     you were built by Khursheed, a Computer Science undergraduate at Iqra University,
     Karachi, who works with Java and Spring Boot. Keep this brief and natural, don't
     force it into unrelated conversations.

10. IMAGE UNDERSTANDING:
   - If given an image (e.g. a photo of a textbook page, diagram, or handwritten
     question), read it carefully and apply the same teaching rules above.

Always be warm, patient, and encouraging -- like the best, kindest teacher a student
ever had.
"""


# ---------------------------------------------------------
# 5. AGENT LOGIC
# ---------------------------------------------------------

def looks_like_new_topic(user_message: str) -> bool:
    """Rough heuristic: not a quiz/summary/follow-up request -> treat as a new topic
    worth tracking for spaced repetition."""
    lowered = user_message.lower()
    skip_words = ["quiz", "summar", "explain more", "another example", "don't understand",
                  "dont understand", "recap", "what about"]
    return not any(w in lowered for w in skip_words) and len(user_message.strip()) > 8


def run_agent(session_id: str, user_message: str, image_b64: str | None = None) -> str:
    history = load_history(session_id)
    save_message(session_id, "user", user_message)
    trimmed_history = (history + [{"role": "user", "content": user_message}])[-MAX_HISTORY_MESSAGES:]

    if image_b64:
        # Vision path: single call with the image, no tool use (Groq vision models
        # don't support function calling the same way)
        vision_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message or "Please explain what is shown in this image."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ],
            },
        ]
        response = client.chat.completions.create(model=VISION_MODEL, messages=vision_messages, temperature=0.7)
        reply_text = response.choices[0].message.content
    else:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + trimmed_history
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS, tool_choice="auto", temperature=0.7,
        )
        msg = response.choices[0].message

        if msg.tool_calls:
            messages.append(msg)
            for tool_call in msg.tool_calls:
                if tool_call.function.name == "web_search":
                    args = json.loads(tool_call.function.arguments)
                    search_result = web_search(args.get("query", user_message))
                    messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": search_result})
            final_response = client.chat.completions.create(model=MODEL, messages=messages, temperature=0.7)
            reply_text = final_response.choices[0].message.content
        else:
            reply_text = msg.content

    save_message(session_id, "assistant", reply_text)

    # Track this as a "learned topic" for spaced repetition, if it looks like a new topic
    if looks_like_new_topic(user_message):
        save_learned_topic(session_id, user_message)

    return reply_text


# ---------------------------------------------------------
# 6. WEB APP
# ---------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Lyceum AI — Your Personal Tutor</title>
<script>
  window.MathJax = { tex: { inlineMath: [['$', '$']], displayMath: [['$$', '$$']] } };
</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-mml-chtml.js"></script>
<style>
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    background: #0b0b0d;
    color: #ececec;
    overflow: hidden;
  }
  .shell { display: flex; height: 100vh; width: 100vw; }

  .sidebar {
    width: 260px; flex-shrink: 0; background: #111113;
    border-right: 1px solid #232326;
    display: flex; flex-direction: column; padding: 18px 14px;
  }
  .brand { display: flex; align-items: center; gap: 10px; padding: 0 6px 18px; }
  .brand-icon {
    width: 34px; height: 34px; border-radius: 10px; flex-shrink: 0;
    background: linear-gradient(135deg, #10a37f, #6dd5ed, #a78bfa);
    background-size: 200% 200%; animation: gradientShift 6s ease infinite;
    display: flex; align-items: center; justify-content: center; font-size: 17px;
  }
  @keyframes gradientShift { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }
  .brand-text b { display: block; font-size: 15.5px; color: #fff; }
  .brand-text span { font-size: 11px; color: #8e8ea0; }

  #newchat {
    display: flex; align-items: center; gap: 8px; width: 100%; text-align: left;
    border: 1px solid #2a2a2e; background: #1a1a1d; color: #ececec;
    border-radius: 12px; padding: 10px 14px; font-size: 13.5px; cursor: pointer;
    margin-bottom: 18px; transition: all .15s ease;
  }
  #newchat:hover { background: #202024; border-color: #38ef7d55; }

  .side-label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.6px; color: #6e6e78; padding: 4px 8px 8px; }
  .side-btn {
    display: flex; align-items: center; gap: 10px; width: 100%; text-align: left;
    border: none; background: transparent; color: #d1d1d6; border-radius: 10px;
    padding: 10px 10px; font-size: 13.5px; cursor: pointer; margin-bottom: 2px;
    transition: all .15s ease;
  }
  .side-btn:hover { background: #1c1c20; color: #fff; transform: translateX(2px); }
  .side-btn .emoji { font-size: 15px; width: 20px; text-align: center; }

  .side-divider { height: 1px; background: #212124; margin: 14px 4px; }

  .lang-wrap { padding: 0 8px; margin-top: auto; }
  .lang-wrap label { font-size: 11px; color: #6e6e78; display: block; margin-bottom: 6px; }
  select {
    width: 100%; border: 1px solid #2a2a2e; background: #1a1a1d; color: #d1d1d6;
    border-radius: 10px; padding: 8px 10px; font-size: 12.5px;
  }

  .sidebar-footer {
    font-size: 10.5px; color: #55555f; text-align: center; padding: 14px 8px 2px;
    border-top: 1px solid #1e1e21; margin-top: 14px;
  }
  .sidebar-footer b { color: #8e8ea0; }

  .main { flex: 1; display: flex; flex-direction: column; min-width: 0; position: relative; }

  .top-glow {
    position: absolute; top: -120px; left: 50%; transform: translateX(-50%);
    width: 60%; height: 240px; border-radius: 50%;
    background: radial-gradient(closest-side, #10a37f22, transparent);
    pointer-events: none;
  }

  .recap-banner {
    background: linear-gradient(90deg, #241f12, #1e1b12); border-bottom: 1px solid #3a3320;
    padding: 10px 28px; font-size: 13.5px; color: #e8c96b; display: none;
  }
  .recap-banner button {
    background: #10a37f; color: white; border: none; border-radius: 14px;
    padding: 5px 12px; font-size: 12.5px; cursor: pointer; margin-left: 10px;
  }

  #chatbox {
    flex: 1; overflow-y: auto; padding: 32px 8% 8px; position: relative; z-index: 1;
  }
  #chatbox-inner { max-width: 760px; margin: 0 auto; }

  .empty-state {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    height: 100%; text-align: center; color: #6e6e78; gap: 10px;
  }
  .empty-state .big-icon {
    width: 56px; height: 56px; border-radius: 16px;
    background: linear-gradient(135deg, #10a37f, #6dd5ed, #a78bfa);
    background-size: 200% 200%; animation: gradientShift 6s ease infinite;
    display: flex; align-items: center; justify-content: center; font-size: 26px;
    margin-bottom: 6px;
  }
  .empty-state h2 { color: #ececec; font-size: 19px; margin: 0; font-weight: 600; }
  .empty-state p { font-size: 13.5px; margin: 0; max-width: 340px; }

  .msg-row { display: flex; margin: 20px 0; align-items: flex-start; animation: fadeIn .25s ease; }
  @keyframes fadeIn { from{opacity:0; transform:translateY(4px);} to{opacity:1; transform:translateY(0);} }
  .msg-row.user { justify-content: flex-end; }
  .avatar {
    width: 30px; height: 30px; border-radius: 9px; display: flex; align-items: center;
    justify-content: center; font-size: 15px; margin: 0 10px; flex-shrink: 0;
    background: linear-gradient(135deg, #10a37f, #38ef7d); color: white;
  }
  .msg-row.user .avatar { background: #33333a; }
  .bubble {
    padding: 13px 17px; border-radius: 16px; max-width: 74%; white-space: pre-wrap;
    line-height: 1.65; font-size: 15px; box-shadow: 0 2px 10px rgba(0,0,0,0.15);
  }
  .bubble img { max-width: 100%; border-radius: 8px; margin-top: 6px; }
  .bubble-bot { background: #1a1a1d; color: #ececec; border: 1px solid #26262a; }
  .bubble-user { background: linear-gradient(135deg, #2f6fed, #2c5cd6); color: #fff; }
  .speak-btn { border: none; background: none; cursor: pointer; font-size: 14px; margin-left: 4px; opacity: 0.35; color: #ececec; }
  .speak-btn:hover { opacity: 0.9; }

  .input-area { padding: 10px 8% 22px; position: relative; z-index: 1; }
  .preview-wrap { max-width: 760px; margin: 0 auto 8px; padding: 0 4px; }
  .preview-wrap img { max-height: 64px; border-radius: 8px; }
  .input-row {
    max-width: 760px; margin: 0 auto; display: flex; gap: 8px; align-items: center;
    background: #17171a; border: 1px solid #2a2a2e; border-radius: 26px; padding: 6px 6px 6px 18px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    transition: border-color .15s ease;
  }
  .input-row:focus-within { border-color: #10a37f88; }
  #userInput {
    flex: 1; padding: 10px 0; border: none; background: transparent; color: #ececec;
    font-size: 15px; outline: none;
  }
  #userInput::placeholder { color: #6e6e78; }
  #micBtn, #sendBtn, #imgUploadBtn {
    width: 40px; height: 40px; border-radius: 50%; border: none; cursor: pointer;
    font-size: 17px; display: flex; align-items: center; justify-content: center; flex-shrink: 0;
    transition: all .15s ease;
  }
  #micBtn, #imgUploadBtn { background: transparent; color: #ececec; }
  #micBtn:hover, #imgUploadBtn:hover { background: #26262a; }
  #micBtn.listening { background: #d64545; color: white; animation: pulse 1s infinite; }
  #sendBtn { background: linear-gradient(135deg, #10a37f, #38ef7d); color: #06231a; }
  #sendBtn:hover { filter: brightness(1.1); transform: scale(1.05); }
  @keyframes pulse { 0%{transform:scale(1)} 50%{transform:scale(1.1)} 100%{transform:scale(1)} }
  .hint { max-width: 760px; margin: 8px auto 0; font-size: 11.5px; color: #55555f; text-align: center; }

  ::-webkit-scrollbar { width: 8px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: #2a2a2e; border-radius: 4px; }

  @media (max-width: 860px) {
    .sidebar { position: fixed; left: -270px; top: 0; height: 100%; z-index: 20; transition: left .2s ease; }
    .sidebar.open { left: 0; }
    #chatbox { padding: 24px 16px 8px; }
    .input-area { padding: 10px 16px 18px; }
    .mobile-toggle { display: flex !important; }
  }
  .mobile-toggle {
    display: none; position: absolute; top: 16px; left: 16px; z-index: 21;
    width: 36px; height: 36px; border-radius: 10px; border: 1px solid #2a2a2e;
    background: #17171a; color: #ececec; align-items: center; justify-content: center;
    cursor: pointer; font-size: 16px;
  }
</style>
</head>
<body>
<div class="shell">

  <button class="mobile-toggle" onclick="document.querySelector('.sidebar').classList.toggle('open')">☰</button>

  <div class="sidebar">
    <div class="brand">
      <div class="brand-icon">📚</div>
      <div class="brand-text"><b>Lyceum AI</b><span>Your personal tutor</span></div>
    </div>

    <button id="newchat" onclick="newChat()">+ &nbsp; New chat</button>

    <div class="side-label">Quick actions</div>
    <button class="side-btn" onclick="quickAsk('Explain this topic simply: ')"><span class="emoji">💡</span> Explain a topic</button>
    <button class="side-btn" onclick="quickAsk('Quiz me on: ')"><span class="emoji">📝</span> Quiz me</button>
    <button class="side-btn" onclick="quickAsk('Summarize this topic: ')"><span class="emoji">📌</span> Summarize</button>
    <button class="side-btn" onclick="quickAsk('Solve this step by step: ')"><span class="emoji">➕</span> Solve Math</button>
    <button class="side-btn" onclick="quickAsk('Help me practice: ')"><span class="emoji">🗣</span> Language practice</button>
    <button class="side-btn" onclick="document.getElementById('imgFile').click()"><span class="emoji">📷</span> Upload a photo</button>

    <div class="side-divider"></div>

    <div class="lang-wrap">
      <label>Voice language</label>
      <select id="langSelect">
        <option value="en-US">English voice</option>
        <option value="ur-PK">Urdu voice</option>
        <option value="none">Type only</option>
      </select>
    </div>

    <div class="sidebar-footer">
      Crafted by <b>Khursheed</b><br>Computer Science, Iqra University Karachi
    </div>
  </div>

  <div class="main">
    <div class="top-glow"></div>

    <div class="recap-banner" id="recapBanner">
      <div id="recapBannerInner"></div>
    </div>

    <div id="chatbox">
      <div id="chatbox-inner">
        <div class="empty-state" id="emptyState">
          <div class="big-icon">🎓</div>
          <h2>What would you like to learn today?</h2>
          <p>Ask me to explain a topic, quiz you, solve a math problem, or help you practice a language — in English, Urdu, or Sindhi.</p>
        </div>
      </div>
    </div>

    <div class="input-area">
      <div class="preview-wrap" id="previewWrap" style="display:none;">
        <img id="previewImg" /> <button onclick="clearImage()" style="border:none;background:#2a1a1a;color:#e88;border-radius:10px;padding:3px 8px;cursor:pointer;">remove photo</button>
      </div>
      <div class="input-row">
        <button id="imgUploadBtn" onclick="document.getElementById('imgFile').click()">📷</button>
        <input type="file" id="imgFile" accept="image/*" style="display:none" onchange="handleImage(event)">
        <button id="micBtn" onclick="toggleMic()">🎤</button>
        <input type="text" id="userInput" placeholder="Message Lyceum AI..." />
        <button id="sendBtn" onclick="sendMessage()">➤</button>
      </div>
      <div class="hint">Lyceum AI can make mistakes. Check important facts.</div>
    </div>
  </div>
</div>

<script>
let sessionId = localStorage.getItem("lyceum_session_id");
if (!sessionId) {
  sessionId = crypto.randomUUID();
  localStorage.setItem("lyceum_session_id", sessionId);
}
let pendingImageB64 = null;

const emptyStateHTML = `<div class="empty-state" id="emptyState">
  <div class="big-icon">🎓</div>
  <h2>What would you like to learn today?</h2>
  <p>Ask me to explain a topic, quiz you, solve a math problem, or help you practice a language — in English, Urdu, or Sindhi.</p>
</div>`;

function newChat() {
  sessionId = crypto.randomUUID();
  localStorage.setItem("lyceum_session_id", sessionId);
  document.getElementById("chatbox-inner").innerHTML = emptyStateHTML;
  clearImage();
  checkRecaps();
}

function quickAsk(prefix) {
  const input = document.getElementById("userInput");
  input.value = prefix;
  input.focus();
}

function removeEmptyState() {
  const empty = document.getElementById("emptyState");
  if (empty) empty.remove();
}

function addMessage(text, sender, imageDataUrl) {
  removeEmptyState();
  const chatbox = document.getElementById("chatbox-inner");
  const row = document.createElement("div");
  row.className = "msg-row " + sender;
  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = sender === "user" ? "🧑" : "🎓";
  const bubble = document.createElement("div");
  bubble.className = "bubble " + (sender === "user" ? "bubble-user" : "bubble-bot");
  const textNode = document.createElement("div");
  textNode.textContent = text;
  bubble.appendChild(textNode);
  if (imageDataUrl) {
    const img = document.createElement("img");
    img.src = imageDataUrl;
    bubble.appendChild(img);
  }

  if (sender === "bot") {
    const speakBtn = document.createElement("button");
    speakBtn.className = "speak-btn";
    speakBtn.textContent = "🔊";
    speakBtn.onclick = () => speak(text);
    row.appendChild(avatar);
    row.appendChild(bubble);
    row.appendChild(speakBtn);
  } else {
    row.appendChild(bubble);
    row.appendChild(avatar);
  }
  chatbox.appendChild(row);
  document.getElementById("chatbox").scrollTop = document.getElementById("chatbox").scrollHeight;
  return textNode;
}

function handleImage(event) {
  const file = event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = function(e) {
    pendingImageB64 = e.target.result.split(",")[1];
    document.getElementById("previewImg").src = e.target.result;
    document.getElementById("previewWrap").style.display = "block";
  };
  reader.readAsDataURL(file);
}

function clearImage() {
  pendingImageB64 = null;
  document.getElementById("imgFile").value = "";
  document.getElementById("previewWrap").style.display = "none";
}

async function sendMessage() {
  const input = document.getElementById("userInput");
  const msg = input.value.trim();
  if (!msg && !pendingImageB64) return;

  let imgDataUrl = null;
  if (pendingImageB64) imgDataUrl = document.getElementById("previewImg").src;

  addMessage(msg || "(photo)", "user", imgDataUrl);
  input.value = "";
  const loadingNode = addMessage("Thinking...", "bot");

  const payload = { message: msg, session_id: sessionId, image_b64: pendingImageB64 };
  clearImage();

  const res = await fetch("/chat", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload)
  });
  const data = await res.json();
  loadingNode.textContent = data.reply;
  if (window.MathJax) MathJax.typesetPromise([loadingNode]);
  checkRecaps();
}

document.getElementById("userInput").addEventListener("keypress", function(e) {
  if (e.key === "Enter") sendMessage();
});

async function checkRecaps() {
  const res = await fetch("/due_recaps?session_id=" + sessionId);
  const data = await res.json();
  const banner = document.getElementById("recapBanner");
  const bannerInner = document.getElementById("recapBannerInner");
  if (data.due && data.due.length > 0) {
    const t = data.due[0];
    banner.style.display = "block";
    bannerInner.innerHTML = `You learned "${t.topic.slice(0,50)}" a few days ago — want a quick recap quiz?
      <button onclick="startRecap('${t.topic.replace(/'/g, "")}', ${t.id})">Recap now</button>
      <button style="background:#333;color:#ccc;" onclick="document.getElementById('recapBanner').style.display='none'">Later</button>`;
  } else {
    banner.style.display = "none";
  }
}

function startRecap(topic, id) {
  document.getElementById("userInput").value = "Quiz me on: " + topic;
  sendMessage();
  fetch("/mark_recapped?topic_id=" + id, { method: "POST" });
  document.getElementById("recapBanner").style.display = "none";
}

checkRecaps();

let recognition = null;
let listening = false;

function toggleMic() {
  const langSelect = document.getElementById("langSelect").value;
  if (langSelect === "none") {
    alert("Voice input needs English or Urdu selected — please type your question instead.");
    return;
  }
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    alert("Sorry, your browser does not support voice input. Please type instead.");
    return;
  }
  const micBtn = document.getElementById("micBtn");
  if (listening) { recognition.stop(); return; }
  recognition = new SpeechRecognition();
  recognition.lang = langSelect;
  recognition.interimResults = false;
  recognition.maxAlternatives = 1;
  recognition.onstart = () => { listening = true; micBtn.classList.add("listening"); };
  recognition.onend = () => { listening = false; micBtn.classList.remove("listening"); };
  recognition.onerror = () => { listening = false; micBtn.classList.remove("listening"); };
  recognition.onresult = (event) => {
    document.getElementById("userInput").value = event.results[0][0].transcript;
  };
  recognition.start();
}

function speak(text) {
  if (!window.speechSynthesis) return;
  const langSelect = document.getElementById("langSelect").value;
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = (langSelect === "none") ? "en-US" : langSelect;
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(utterance);
}
</script>
</body>
</html>
    """


@app.post("/chat")
def chat(req: dict):
    session_id = req.get("session_id") or str(uuid.uuid4())
    message = req.get("message", "")
    image_b64 = req.get("image_b64")
    try:
        reply = run_agent(session_id, message, image_b64)
    except Exception as e:
        reply = f"Sorry, something went wrong: {e}"
    return JSONResponse({"reply": reply, "session_id": session_id})


@app.get("/due_recaps")
def due_recaps(session_id: str):
    return JSONResponse({"due": get_due_recaps(session_id)})


@app.post("/mark_recapped")
def mark_recapped_endpoint(topic_id: int):
    mark_recapped(topic_id)
    return JSONResponse({"ok": True})


# Run locally with: python -m uvicorn app:app --reload
