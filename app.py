import os, re, json, time, logging
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

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
CORS(app)
limiter = Limiter(get_remote_address, app=app, default_limits=["60/hour","15/minute"])

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
MODEL           = os.environ.get("MISTRAL_MODEL", "mistral-small-latest")
ADMIN_TOKEN     = os.environ.get("ADMIN_TOKEN", "changeme")
INDEX_FILE      = Path("lambersart_index.json")
INDEX_TTL       = 3600 * 12
MAX_HISTORY     = 10

PAGES = [
    "https://lambersart.fr/",
    "https://lambersart.fr/agenda",
    "https://lambersart.fr/actualites",
    "https://lambersart.fr/mairie",
    "https://lambersart.fr/le-ccas-de-lambersart",
    "https://lambersart.fr/education",
    "https://lambersart.fr/associations",
    "https://lambersart.fr/urbanisme",
    "https://lambersart.fr/se-divertir",
    "https://lambersart.fr/cadre-de-vie",
    "https://lambersart.fr/jeunesse",
    "https://lambersart.fr/seniors",
]

PROMPT = (
    "Tu es l'assistant numerique officiel de la ville de Lambersart (59130, Nord).\n"
    "Tu aides les habitants a trouver des informations sur les services municipaux.\n"
    "Reponds toujours en francais, sois professionnel et bienveillant.\n"
    "Base-toi UNIQUEMENT sur le contexte ci-dessous. N'invente rien.\n"
    "Si l'info est absente, propose d'appeler la mairie : 03 20 08 44 44\n"
    "Horaires mairie : lun-jeu 8h30-17h30, ven 8h30-12h30\n"
    "CONTEXTE :\n{context}"
)

URL_PATTERN = re.compile(r"https?://lambersart\.fr[^\s\]\)\"']*")

_index, _vec, _mat = {}, None, None

def fetch(url):
    try:
        r = req_lib.get(url, timeout=10, headers={"User-Agent": "LambersartBot/1.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        title = soup.title.get_text(strip=True) if soup.title else url
        for t in soup(["script","style","nav","footer","header","form","noscript"]):
            t.decompose()
        text = re.sub(r"\n{3,}", "\n\n", soup.get_text(separator="\n", strip=True))
        return {"url": url, "title": title, "text": text[:4000]}
    except Exception as e:
        log.warning("Fetch failed %s: %s", url, e)
        return {"url": url, "title": url, "text": ""}

def fit():
    global _vec, _mat
    docs = list(_index.values())
    if not docs:
        return
    _vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1,
                           token_pattern=r"[a-zA-Z]{2,}")
    _mat = _vec.fit_transform([d["text"] for d in docs])

def build(force=False):
    global _index
    if not force and INDEX_FILE.exists():
        if time.time() - INDEX_FILE.stat().st_mtime < INDEX_TTL:
            log.info("Index from cache")
            _index = {d["url"]: d for d in json.loads(INDEX_FILE.read_text(encoding="utf-8"))}
            fit()
            return
    log.info("Crawling lambersart.fr...")
    docs = []
    for url in PAGES:
        d = fetch(url)
        if d["text"]:
            docs.append(d)
        time.sleep(0.3)
    _index = {d["url"]: d for d in docs}
    INDEX_FILE.write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")
    fit()
    log.info("Index ready: %d pages", len(_index))

def get_context(query, k=3):
    if _vec is None:
        return ""
    sims = cosine_similarity(_vec.transform([query]), _mat).flatten()
    chunks = []
    for i in np.argsort(sims)[::-1][:k]:
        if sims[i] < 0.01:
            continue
        d = list(_index.values())[i]
        chunks.append("[" + d["url"] + "]\n" + d["text"][:800])
    return "\n\n---\n\n".join(chunks)

def mistral_call(messages):
    client = Mistral(api_key=MISTRAL_API_KEY)
    if MISTRAL_V1:
        r = client.chat.complete(model=MODEL, messages=messages,
                                 temperature=0.2, max_tokens=600)
    else:
        from mistralai.models.chat_completion import ChatMessage
        r = client.chat(
            model=MODEL,
            messages=[ChatMessage(role=m["role"], content=m["content"]) for m in messages],
            temperature=0.2,
            max_tokens=600
        )
    return r.choices[0].message.content

sessions = {}

@app.route("/")
def index():
    return jsonify({"status": "ok", "service": "Assistant Lambersart",
                    "pages_indexed": len(_index)})

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": MODEL,
                    "indexed": len(_index), "mistral_v1": MISTRAL_V1})

@app.route("/chat", methods=["POST"])
@limiter.limit("20/minute")
def chat():
    data = request.get_json(force=True)
    sid  = data.get("session_id", "default")
    msg  = data.get("message", "").strip()[:400]
    if not msg:
        return jsonify({"error": "Message vide"}), 400

    ctx  = get_context(msg)
    hist = sessions.setdefault(sid, [])
    hist.append({"role": "user", "content": msg})

    try:
        msgs   = [{"role": "system", "content": PROMPT.format(context=ctx)}] + hist[-MAX_HISTORY:]
        answer = mistral_call(msgs)
        hist.append({"role": "assistant", "content": answer})
        urls    = URL_PATTERN.findall(answer)
        sources = [{"url": u, "title": _index.get(u, {}).get("title", u)}
                   for u in dict.fromkeys(urls)]
        return jsonify({"answer": answer, "sources": sources})
    except Exception as e:
        log.error("Mistral error: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route("/reindex", methods=["POST"])
def reindex():
    if request.headers.get("X-Admin-Token", "") != ADMIN_TOKEN:
        return jsonify({"error": "Unauthorized"}), 401
    build(force=True)
    return jsonify({"status": "ok", "pages": len(_index)})

build()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5050)), debug=False)
