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
import io
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

try:
    import pdfplumber
    PDF_ENGINE = "pdfplumber"
except ImportError:
    PDF_ENGINE = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
limiter = Limiter(get_remote_address, app=app, default_limits=["60/hour","15/minute"])

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
MODEL           = os.environ.get("MISTRAL_MODEL", "mistral-small-latest")
ADMIN_TOKEN     = os.environ.get("ADMIN_TOKEN", "changeme")
INDEX_FILE      = Path("lambersart_index.json")
PDF_INDEX_FILE  = Path("lambersart_pdf_index.json")
KNOWLEDGE_FILE  = Path("knowledge.json")
INDEX_TTL       = 3600 * 12
MAX_HISTORY     = 10

KNOWLEDGE = (
    "=== VILLE DE LAMBERSART ===\n"
    "Ville : Lambersart (59130) - Nord, Hauts-de-France\n"
    "Population : environ 27 400 habitants\n"
    "Site officiel : https://lambersart.fr\n\n"
    "=== MAIRE ET CONSEIL MUNICIPAL ===\n"
    "Maire : Nicolas Bouche (reelu aux elections municipales de mars 2026)\n"
    "Conseil municipal : 35 elus\n"
    "Page conseil : https://lambersart.fr/le-conseil-municipal\n\n"
    "=== MAIRIE ===\n"
    "Adresse : 19 avenue Georges-Clemenceau, 59130 Lambersart\n"
    "Telephone : 03 20 08 44 44\n"
    "Email : mairie@lambersart.fr\n"
    "Horaires : Lundi-Jeudi 8h30-17h30 / Vendredi 8h30-12h30 / Samedi-Dimanche ferme\n"
    "Contact : https://lambersart.fr/nous-contacter\n\n"
    "=== ETAT CIVIL ===\n"
    "Naissances, mariages, deces, PACS : service etat civil mairie\n"
    "Cartes identite et passeports : RDV obligatoire au 03 20 08 44 44\n"
    "Page : https://lambersart.fr/etat-civil\n\n"
    "=== POLICE MUNICIPALE ===\n"
    "Telephone : 03 20 08 44 60\n"
    "Page : https://lambersart.fr/prevention-et-securite-publique\n\n"
    "=== CCAS ===\n"
    "Services : aides financieres, portage repas, aide a domicile, epicerie sociale\n"
    "Navette CCAS : transport gratuit pour +70 ans ou retraites a mobilite reduite\n"
    "Contact : 03 20 08 44 44\n"
    "Page : https://lambersart.fr/le-ccas-de-lambersart\n\n"
    "=== SENIORS ===\n"
    "Page : https://lambersart.fr/seniors\n\n"
    "=== DEPLACEMENTS ===\n"
    "Reseau Ilevia (bus, metro, tramway)\n"
    "Velos V'Lille en libre-service\n"
    "Page : https://lambersart.fr/deplacements\n\n"
    "=== EDUCATION ===\n"
    "Inscription scolaire : 03 20 08 44 44\n"
    "Page : https://lambersart.fr/education\n\n"
    "=== JEUNESSE ===\n"
    "Page : https://lambersart.fr/jeunesse\n\n"
    "=== CONTACTS UTILES ===\n"
    "Mairie            : 03 20 08 44 44\n"
    "Police municipale : 03 20 08 44 60\n"
    "Site officiel     : https://lambersart.fr\n"
    "Demarches en ligne: https://lambersart.fr/mes-demarches\n"
)

PROMPT = (
    "Tu es l'assistant numerique officiel de la ville de Lambersart (59130, Nord).\n"
    "Tu aides les habitants a trouver des informations sur les services municipaux.\n\n"
    "REGLES ABSOLUES - A RESPECTER IMPERATIVEMENT :\n"
    "1. Tu ne reponds QUE sur la base des informations ci-dessous (BASE DE CONNAISSANCE + CONTEXTE DU SITE).\n"
    "2. Si une information n'est PAS dans la base ou le contexte : reponds 'Je n'ai pas cette information. Contactez la mairie au 03 20 08 44 44 ou sur lambersart.fr'\n"
    "3. N'INVENTE JAMAIS de date, d'horaire, de nom, de tarif ou de procedure.\n"
    "4. N'INVENTE JAMAIS de lien ou d'adresse email.\n"
    "5. Si tu n'es pas certain a 100%, dis-le clairement et oriente vers la mairie.\n\n"
    "STYLE DE REPONSE :\n"
    "- Reponds en francais, de facon claire et structuree\n"
    "- Utilise des emojis pour aerer : 📍 adresse, 🕐 horaires, 📞 telephone, 📅 date\n"
    "- Mets les infos cles en gras avec **texte**\n"
    "- Termine par une invitation a contacter la mairie si besoin\n\n"
    "BASE DE CONNAISSANCE LAMBERSART :\n"
    + KNOWLEDGE +
    "\n\nCONTEXTE EXTRAIT DU SITE lambersart.fr :\n{context}\n\n"
    "Si le contexte ci-dessus ne contient pas la reponse, dis-le honnetement."
)

_index, _vec, _mat = {}, None, None


def load_knowledge_index():
    if KNOWLEDGE_FILE.exists():
        try:
            data = json.loads(KNOWLEDGE_FILE.read_text(encoding="utf-8"))
            log.info("knowledge.json charge : %d pages", len(data))
            return {d["url"]: d for d in data}
        except Exception as e:
            log.warning("knowledge.json illisible: %s", e)
    return {}


def fetch(url):
    try:
        r = req_lib.get(url, timeout=12, headers={"User-Agent": "LambersartBot/2.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        title = soup.title.get_text(strip=True) if soup.title else url
        # Collecter liens internes
        links = set()
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if href.startswith("/") and not href.startswith("//"):
                href = "https://lambersart.fr" + href
            if (href.startswith("https://lambersart.fr")
                    and "?" not in href and "#" not in href):
                links.add(href.rstrip("/"))
        for t in soup(["script","style","nav","footer","header","form","noscript"]):
            t.decompose()
        text = re.sub(r"\n{3,}", "\n\n", soup.get_text(separator="\n", strip=True))
        return {
            "url": url, "title": title,
            "text": text[:5000],
            "raw_html": r.text[:40000],
            "links": list(links)
        }
    except Exception as e:
        log.warning("Fetch failed %s: %s", url, e)
        return {"url": url, "title": url, "text": "", "raw_html": "", "links": []}


def fetch_pdf(url):
    """Telecharge et extrait TOUT le texte d un PDF, avec plusieurs fallbacks."""
    if PDF_ENGINE is None:
        return None
    try:
        r = req_lib.get(
            url, timeout=30,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; LambersartBot/2.0)",
                "Accept": "application/pdf,*/*"
            },
            allow_redirects=True
        )
        r.raise_for_status()

        content = r.content
        if len(content) < 100:
            log.warning("PDF vide ou trop petit : %s (%d bytes)", url, len(content))
            return None

        text_pages = []
        try:
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                for i, page in enumerate(pdf.pages):
                    # Methode 1 : extraction normale
                    t = page.extract_text(x_tolerance=3, y_tolerance=3)
                    if not t or len(t.strip()) < 10:
                        # Methode 2 : extraction par mots
                        words = page.extract_words()
                        if words:
                            t = " ".join(w["text"] for w in words)
                    if t and t.strip():
                        text_pages.append(f"[Page {i+1}]\n{t.strip()}")
        except Exception as e:
            log.warning("pdfplumber failed pour %s : %s", url, e)
            return None

        if not text_pages:
            log.warning("PDF sans texte extractible (scan ?) : %s", url)
            return None

        full_text = "\n\n".join(text_pages)
        full_text = re.sub(r"\n{3,}", "\n\n", full_text.strip())

        # Titre propre depuis l URL
        filename = url.split("/")[-1]
        title = re.sub(r"[_\-]+", " ", filename.replace(".pdf", "")).strip().title()
        if not title:
            title = url

        log.info("PDF OK : %s | %d pages | %d chars", title, len(text_pages), len(full_text))
        return {
            "url": url,
            "title": title,
            "text": full_text[:8000],
            "type": "pdf",
            "pages": len(text_pages)
        }

    except req_lib.exceptions.SSLError:
        # Retry sans verification SSL pour certains PDFs de mairie
        try:
            r = req_lib.get(url, timeout=30, verify=False,
                            headers={"User-Agent": "Mozilla/5.0 (compatible; LambersartBot/2.0)"})
            r.raise_for_status()
            text_pages = []
            with pdfplumber.open(io.BytesIO(r.content)) as pdf:
                for i, page in enumerate(pdf.pages):
                    t = page.extract_text(x_tolerance=3, y_tolerance=3)
                    if t and t.strip():
                        text_pages.append(f"[Page {i+1}]\n{t.strip()}")
            if not text_pages:
                return None
            full_text = "\n\n".join(text_pages)
            title = re.sub(r"[_\-]+", " ", url.split("/")[-1].replace(".pdf","")).title()
            log.info("PDF OK (no-SSL) : %s | %d pages", title, len(text_pages))
            return {"url": url, "title": title, "text": full_text[:8000],
                    "type": "pdf", "pages": len(text_pages)}
        except Exception as e2:
            log.warning("PDF SSL fallback failed %s : %s", url, e2)
            return None
    except Exception as e:
        log.warning("PDF failed %s : %s", url, e)
        return None


def discover_pdf_urls(pages_index):
    """Collecte toutes les URLs PDF depuis les pages HTML indexees."""
    pdf_re   = re.compile(r'https?://[^\s<>"\']+\.pdf(?:[?#][^\s<>"\']*)?', re.I)
    href_re  = re.compile(r'href=["\']([^"\']+\.pdf[^"\']*)["\']', re.I)
    pdf_urls = set()

    for doc in pages_index.values():
        raw = doc.get("raw_html", "")
        txt = doc.get("text", "")
        base_url = doc.get("url", "")

        # Methode 1 : regex sur le HTML brut (liens absolus)
        for u in pdf_re.findall(raw):
            if "lambersart.fr" in u:
                pdf_urls.add(u.split("?")[0].split("#")[0])

        # Methode 2 : href relatifs
        for href in href_re.findall(raw):
            if href.startswith("/"):
                pdf_urls.add("https://lambersart.fr" + href.split("?")[0])
            elif href.startswith("http") and "lambersart.fr" in href:
                pdf_urls.add(href.split("?")[0])

        # Methode 3 : texte brut (liens copies)
        for u in pdf_re.findall(txt):
            if "lambersart.fr" in u:
                pdf_urls.add(u.split("?")[0])

    # Nettoyer
    pdf_urls = {u for u in pdf_urls
                if "lambersart.fr" in u
                and not any(u.lower().endswith(e) for e in (".jpg",".png",".gif"))}

    log.info("URLs PDF decouverts : %d", len(pdf_urls))
    return pdf_urls


def crawl_pdfs_list(pdf_urls):
    """Indexe tous les PDFs. Force le re-crawl si le cache est vide ou expire."""
    if PDF_ENGINE is None:
        log.warning("pdfplumber non installe - PDFs ignores. Faites : pip install pdfplumber")
        return {}

    # Cache valide ?
    if PDF_INDEX_FILE.exists():
        cached_data = json.loads(PDF_INDEX_FILE.read_text(encoding="utf-8"))
        cache_age   = time.time() - PDF_INDEX_FILE.stat().st_mtime
        if cache_age < INDEX_TTL and len(cached_data) > 0:
            log.info("PDF cache valide : %d docs (age: %dmin)", len(cached_data), int(cache_age/60))
            return {d["url"]: d for d in cached_data}

    pdf_urls = {u for u in pdf_urls if "lambersart.fr" in u}
    if not pdf_urls:
        log.info("Aucun PDF lambersart.fr detecte")
        return {}

    log.info("Indexation PDFs : %d fichiers...", len(pdf_urls))
    docs    = []
    errors  = []

    for i, url in enumerate(sorted(pdf_urls), 1):
        log.info("PDF [%d/%d] : %s", i, len(pdf_urls), url)
        d = fetch_pdf(url)
        if d:
            docs.append(d)
        else:
            errors.append(url)
        time.sleep(0.5)  # Respecter le serveur

    log.info("PDFs : %d OK / %d erreurs / %d total", len(docs), len(errors), len(pdf_urls))
    if errors:
        log.info("PDFs en erreur : %s", errors[:5])

    PDF_INDEX_FILE.write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")
    return {d["url"]: d for d in docs}


def fit():
    global _vec, _mat
    docs = [d for d in _index.values() if d.get("text","").strip()]
    if not docs:
        return
    _vec = TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=1,
                           token_pattern=r"[a-zA-Z]{2,}")
    _mat = _vec.fit_transform([d["text"] for d in docs])


def build(force=False):
    global _index
    kn = load_knowledge_index()

    if not force and INDEX_FILE.exists():
        if time.time() - INDEX_FILE.stat().st_mtime < INDEX_TTL:
            log.info("Index from cache")
            cached = {d["url"]: d for d in json.loads(INDEX_FILE.read_text(encoding="utf-8"))}
            pdf_cached = {}
            if PDF_INDEX_FILE.exists():
                age_pdf = time.time() - PDF_INDEX_FILE.stat().st_mtime
                if age_pdf < INDEX_TTL:
                    pdf_cached = {d["url"]: d for d in json.loads(PDF_INDEX_FILE.read_text(encoding="utf-8"))}
                else:
                    log.info("Cache PDF expire, re-indexation PDFs...")
                    pdf_urls_c = discover_pdf_urls(cached)
                    pdf_cached = crawl_pdfs_list(pdf_urls_c)
            else:
                log.info("Pas de cache PDF, indexation initiale...")
                pdf_urls_c = discover_pdf_urls(cached)
                pdf_cached = crawl_pdfs_list(pdf_urls_c)
            _index = {**kn, **cached, **pdf_cached}
            fit()
            log.info("Cache : %d pages + %d PDFs", len(cached), len(pdf_cached))
            return

    log.info("Crawl recursif complet lambersart.fr...")
    visited  = set()
    queue    = ["https://lambersart.fr/"]
    docs     = []
    pdf_urls = set()
    pdf_re   = re.compile(r"https?://[^\s<>\"']+[.]pdf", re.I)

    while queue:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)
        skip_exts = (".jpg",".jpeg",".png",".gif",".zip",".doc",".xls",".css",".js",".svg",".ico",".webp")
        if any(url.lower().endswith(e) for e in skip_exts):
            continue
        if url.lower().endswith(".pdf"):
            pdf_urls.add(url)
            continue
        d = fetch(url)
        time.sleep(0.25)
        if d.get("text"):
            docs.append(d)
            log.info("[%d] %s", len(docs), url)
        for link in d.get("links", []):
            if link not in visited:
                queue.append(link)
        for pu in pdf_re.findall(d.get("raw_html", "")):
            if "lambersart.fr" in pu:
                pdf_urls.add(pu)

    crawled = {d["url"]: d for d in docs}
    INDEX_FILE.write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Crawl HTML : %d pages", len(docs))

    # Decouvrir les PDFs depuis les pages HTML + liens collectes
    pdf_urls_from_pages = discover_pdf_urls(crawled)
    all_pdf_urls = pdf_urls | pdf_urls_from_pages
    log.info("Total PDFs a indexer : %d", len(all_pdf_urls))

    pdf_idx = crawl_pdfs_list(all_pdf_urls)
    _index = {**kn, **crawled, **pdf_idx}
    fit()
    log.info("Index pret : %d pages + %d PDFs", len(crawled)+len(kn), len(pdf_idx))


def get_context(query, k=4):
    if _vec is None or _mat is None:
        return ""
    sims = cosine_similarity(_vec.transform([query]), _mat).flatten()
    chunks = []
    docs = [d for d in _index.values() if d.get("text","").strip()]
    for i in np.argsort(sims)[::-1][:k]:
        if sims[i] < 0.01:
            continue
        d = docs[i]
        chunks.append("[" + d["url"] + "]\n" + d["text"][:1000])
    return "\n\n---\n\n".join(chunks)


def mistral_call(messages, retries=3):
    for attempt in range(retries):
        try:
            client = Mistral(api_key=MISTRAL_API_KEY)
            if MISTRAL_V1:
                r = client.chat.complete(model=MODEL, messages=messages,
                                         temperature=0.1, max_tokens=600)
            else:
                from mistralai.models.chat_completion import ChatMessage
                r = client.chat(
                    model=MODEL,
                    messages=[ChatMessage(role=m["role"], content=m["content"]) for m in messages],
                    temperature=0.1, max_tokens=600
                )
            return r.choices[0].message.content
        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower() or "capacity" in err.lower():
                wait = 2 ** attempt
                log.warning("Rate limit, retry %d/%d dans %ds", attempt+1, retries, wait)
                time.sleep(wait)
            else:
                raise
    return "Service momentanement indisponible. Appelez le 03 20 08 44 44."


sessions = {}


@app.route("/")
def index():
    return jsonify({"status": "ok", "service": "Assistant Lambersart",
                    "pages_indexed": len(_index)})


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": MODEL, "indexed": len(_index)})


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
        msgs   = [{"role": "system", "content": PROMPT.format(context=ctx or "Aucun contexte trouve.")}] + hist[-MAX_HISTORY:]
        answer = mistral_call(msgs)
        hist.append({"role": "assistant", "content": answer})
        url_re  = re.compile(r"https?://lambersart\.fr[^\s\])'\"]*")
        urls    = url_re.findall(answer)
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
