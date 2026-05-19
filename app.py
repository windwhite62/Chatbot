#!/usr/bin/env python3
"""
Chatbot Lambersart — Backend Flask Production
mistralai==1.2.5 (version fixée)
"""

import os, re, json, time, logging
from pathlib import Path

# ── Chargement .env ───────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# ── Import Mistral (v1.x) ─────────────────────────────────────────────────────
try:
    from mistralai import Mistral
    MISTRAL_V1 = True
except ImportError:
    from mistralai.client import MistralClient as Mistral
    MISTRAL_V1 = False

import requests as req_lib
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)

ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
CORS(app, origins=[o.strip() for o in ALLOWED_ORIGINS])
limiter = Limiter(get_remote_address, app=app, default_limits=["60/hour","15/minute"])

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
MODEL           = os.environ.get("MISTRAL_MODEL", "mistral-small-latest")
ADMIN_TOKEN     = os.environ.get("ADMIN_TOKEN", "changeme")
MAX_HISTORY     = 10
INDEX_FILE      = Path("lambersart_index.json")
INDEX_TTL       = 3600 * 12

PAGES_TO_INDEX = [
    "https://lambersart.fr/",
    "https://lambersart.fr/agenda",
    "https://lambersart.fr/actualites",
    "https://lambersart.fr/mairie",
    "https://lambersart.fr/le-ccas-de-lambersart",
    "https://lambersart.fr/education",
    "https://lambersart.fr/associations",
    "https://lambersart.fr/urbanisme",
    "https://lambersart.fr/se-deplacer",
    "https://lambersart.fr/cadre-de-vie",
    "https://lambersart.fr/demarches-administratives",
    "https://lambersart.fr/jeunesse",
    "https://lambersart.fr/seniors",
]

SYSTEM_PROMPT = """\
Tu es l'assistant numérique officiel de la ville de Lambersart (59130, Nord).
Tu aides les habitants à trouver des informations sur les services municipaux.
RÈGLES :
1. Réponds en français, ton professionnel et bienveillant.
2. Base-toi UNIQUEMENT sur le contexte ci-dessous. N'invente rien.
3. Si l'info est absente du contexte, propose d'appeler la mairie.
4. Contact mairie : 19 av. Georges-Clemenceau · Tél. 03 20 08 44 44
   Horaires : lun-jeu 8h30-17h30 · ven 8h30-12h30
CONTEXTE :
{context}
"""

_index, _vectorizer, _tfidf_matrix = {}, None, None
OFF_TOPIC = re.compile(r"\b(bitcoin|crypto|pornograph|terroris|hacker|<script)\b", re.I)

def fetch_page(url):
    try:
        r = req_lib.get(url, timeout=10, headers={"User-Agent": "LambersartBot/1.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        title = soup.title.get_text(strip=True) if soup.title else url
        for tag in soup(["script","style","nav","footer","header","form","noscript"]):
            tag.decompose()
        text = re.sub(r"\n{3,}", "\n\n", soup.get_text(separator="\n", strip=True))
        return {"url": url, "title": title, "text": text[:4000]}
    except Exception as e:
        log.warning(f"Fetch failed {url}: {e}")
        return {"url": url, "title": url, "text": ""}

def _fit_tfidf():
    global _vectorizer, _tfidf_matrix
    docs = list(_index.values())
    if not docs: return
    _vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=1,
                                   token_pattern=r"[a-zA-ZA-y]{2,}")
    _tfidf_matrix = _vectorizer.fit_transform([d["text"] for d in docs])

def build_index(force=False):
    global _index
    if not force and INDEX_FILE.exists():
        if time.time() - INDEX_FILE.stat().st_mtime < INDEX_TTL:
            log.info("Index depuis cache.")
            _index = {d["url"]: d for d in json.loads(INDEX_FILE.read_text(encoding="utf-8"))}
            _fit_tfidf(); return
    log.info("Crawl des pages lambersart.fr...")
    docs = []
    for url in PAGES_TO_INDEX:
        doc = fetch_page(url)
        if doc["text"]: docs.append(doc)
        time.sleep(0.3)
    _index = {d["url"]: d for d in docs}
    INDEX_FILE.write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")
    _fit_tfidf()
    log.info(f"Index pret : {len(_index)} pages.")

def retrieve_context(query, top_k=3):
    if _vectorizer is None: return ""
    sims = cosine_similarity(_vectorizer.transform([query]), _tfidf_matrix).flatten()
    chunks = []
    for i in np.argsort(sims)[::-1][:top_k]:
        if sims[i] < 0.01: continue
        doc = list(_index.values())[i]
        chunks.append(f"[{doc['url']}]\n{doc['text'][:800]}")
    return "\n\n---\n\n".join(chunks) if chunks else ""

sessions = {}

def call_mistral(messages):
    """Appel Mistral compatible v1+ et fallback v0."""
    client = Mistral(api_key=MISTRAL_API_KEY)
    if MISTRAL_V1:
        resp = client.chat.complete(model=MODEL, messages=messages,
                                    temperature=0.2, max_tokens=600)
    else:
        from mistralai.models.chat_completion import ChatMessage
        resp = client.chat(
            model=MODEL,
            messages=[ChatMessage(role=m["role"], content=m["content"]) for m in messages],
            temperature=0.2, max_tokens=600
        )
    return resp.choices[0].message.content

@app.route("/chat", methods=["POST"])
@limiter.limit("20/minute")
def chat():
    data = request.get_json(force=True)
    sid  = data.get("session_id", "default")
    msg  = data.get("message", "").strip()[:400]
    if not msg:
        return jsonify({"error": "Message vide"}), 400
    if OFF_TOPIC.search(msg):
        return jsonify({"answer": "Je réponds uniquement aux questions sur Lambersart.", "sources": []}), 200

    context = retrieve_context(msg)
    system  = SYSTEM_PROMPT.format(context=context)
    history = sessions.setdefault(sid, [])
    history.append({"role": "user", "content": msg})

    try:
        messages = [{"role": "system", "content": system}] + history[-MAX_HISTORY:]
        answer   = call_mistral(messages)
        history.append({"role": "assistant", "content": answer})
        urls    = re.findall(r"https?://lambersart\.fr[^\s\]\)\"']*", answer)
        sources = [{"url": u, "title": _index.get(u,{}).get("title", u)} for u in dict.fromkeys(urls)]
        return jsonify({"answer": answer, "sources": sources})
    except Exception as e:
        log.error(f"Mistral error: {e}")
        return jsonify({"error": f"Erreur API : {str(e)[:100]}"}), 500

@app.route("/reindex", methods=["POST"])
def reindex():
    if request.headers.get("X-Admin-Token","") != ADMIN_TOKEN:
        return jsonify({"error": "Unauthorized"}), 401
    build_index(force=True)
    return jsonify({"status": "ok", "pages": len(_index)})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "mistral_v1": MISTRAL_V1,
                    "model": MODEL, "indexed_pages": len(_index)})

# Démarrage
build_index()
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5050)), debug=False)
