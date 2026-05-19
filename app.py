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
KNOWLEDGE_FILE  = Path("knowledge.json")
INDEX_TTL       = 3600 * 12
MAX_HISTORY     = 10

PAGES = [
    "https://lambersart.fr/",
    "https://lambersart.fr/agenda",
    "https://lambersart.fr/actualites",
    "https://lambersart.fr/le-ccas-de-lambersart",
    "https://lambersart.fr/education",
    "https://lambersart.fr/associations",
    "https://lambersart.fr/urbanisme",
    "https://lambersart.fr/se-deplacer",
    "https://lambersart.fr/jeunesse",
    "https://lambersart.fr/seniors",
    "https://lambersart.fr/etat-civil",
    "https://lambersart.fr/arena",
]

KNOWLEDGE = (
    "=== VILLE DE LAMBERSART ===\n"
    "Ville : Lambersart (59130) - Nord, Hauts-de-France\n"
    "Population : ~27 400 habitants\n"
    "Site officiel : https://lambersart.fr\n\n"
    "=== MAIRE ET CONSEIL MUNICIPAL ===\n"
    "Maire : Nicolas Bouche (reelu aux elections municipales de mars 2026)\n"
    "Conseil municipal : 35 elus\n"
    "Page conseil : https://lambersart.fr/le-conseil-municipal\n\n"
    "=== MAIRIE ===\n"
    "Adresse : 19 avenue Georges-Clemenceau, 59130 Lambersart\n"
    "Telephone : 03 20 08 44 44\n"
    "Email : mairie@lambersart.fr\n"
    "Horaires : Lun-Jeu 8h30-17h30 / Ven 8h30-12h30 / Sam-Dim ferme\n"
    "Contact : https://lambersart.fr/nous-contacter\n"
    "Formulaire : https://lambersart.fr/la-mairie-vous-repond\n\n"
    "=== ETAT CIVIL ===\n"
    "Naissances, mariages, deces, PACS : service etat civil mairie\n"
    "Cartes d'identite et passeports : RDV obligatoire au 03 20 08 44 44\n"
    "Page : https://lambersart.fr/etat-civil\n"
    "Titres identite : https://lambersart.fr/titres-didentite\n\n"
    "=== POLICE MUNICIPALE ===\n"
    "Telephone : 03 20 08 44 60\n"
    "Disponible lundi au vendredi\n"
    "Signalement : https://lambersart.fr/signalements\n\n"
    "=== CCAS ===\n"
    "Centre Communal d'Action Sociale - accompagnement habitants en difficulte\n"
    "Services : aides financieres, portage repas, aide a domicile, epicerie sociale\n"
    "Navette CCAS : transport gratuit pour +70 ans ou retraites a mobilite reduite\n"
    "Apres-midi convivial seniors CCAS : inscriptions avant le 5 juin 2026\n"
    "Contact : 03 20 08 44 44\n"
    "Page : https://lambersart.fr/le-ccas-de-lambersart\n\n"
    "=== SENIORS ===\n"
    "Activites, animations, ateliers bien-etre\n"
    "Ateliers anti-escroqueries +60 ans (mai 2026)\n"
    "Allocation solidarite personnes agees : https://lambersart.fr/demander-une-allocation-de-solidarite-aux-personnes-agees-en-ligne\n"
    "Page : https://lambersart.fr/seniors\n\n"
    "=== DEPLACEMENTS ===\n"
    "Reseau Ilevia (bus, metro, tramway)\n"
    "Velos V'Lille en libre-service\n"
    "Box velos securises : https://lambersart.fr/demander-une-place-dans-un-box-velos\n"
    "Page : https://lambersart.fr/deplacements\n\n"
    "=== EDUCATION ===\n"
    "Projet educatif et social 2024-2029\n"
    "Inscription scolaire : 03 20 08 44 44\n"
    "Restauration scolaire : https://lambersart.fr/la-restauration-scolaire\n"
    "Page : https://lambersart.fr/education\n\n"
    "=== JEUNESSE ===\n"
    "Point Information Jeunesse : 12-25 ans\n"
    "Job Day 2026 : mercredi 21 mai 2026 - emploi, alternance, stages\n"
    "Conseil des Jeunes : https://lambersart.fr/le-conseil-des-jeunes\n"
    "Page : https://lambersart.fr/jeunesse\n\n"
    "=== SPORTS ET LOISIRS ===\n"
    "Arena Lambersart : sports de sable, bords de Deule (nouvelle gestion avril 2026)\n"
    "Cinema : https://lambersart.fr/cinema\n"
    "Bibliotheques et ludotheques : https://lambersart.fr/bibliotheques-et-ludotheques\n"
    "Salle Malraux : salle de spectacle\n\n"
    "=== URBANISME ===\n"
    "Permis de construire, declarations prealables, PLU\n"
    "Page : https://lambersart.fr/urbanisme-0\n\n"
    "=== AGENDA MAI 2026 ===\n"
    "18 mai - 5 juin : Inscription gouter CCAS seniors\n"
    "21 mai : Job Day emploi et alternance\n"
    "21 mai : Assemblee de quartier Canteleu\n"
    "21 mai : Cine-debat salle Malraux\n"
    "23 mai : Braderie Briqueterie (rue Jean Moulin)\n"
    "23 mai : Marche nocturne (berges de la Deule)\n"
    "Agenda complet : https://lambersart.fr/agenda\n\n"
    "=== CONTACTS UTILES ===\n"
    "Mairie           : 03 20 08 44 44\n"
    "Police municipale: 03 20 08 44 60\n"
    "Site officiel    : https://lambersart.fr\n"
    "Demarches ligne  : https://lambersart.fr/mes-demarches\n"
    "Newsletter       : https://lambersart.fr/sinscrire-la-newsletter\n"
)

PROMPT = (
    "Tu es LAMI, l'assistant municipal intelligent de Lambersart (59130, Nord).\n"
    "Tu es chaleureux, precis et utile. Tu reponds comme un agent d'accueil expert.\n\n"

    "=== STYLE DE REPONSE ===\n"
    "1. Commence toujours par une accroche courte et chaleureuse (1 ligne max)\n"
    "2. Presente les infos avec des blocs visuels clairs separes par des lignes vides\n"
    "3. Utilise ces emojis selon le contexte :\n"
    "   📍 adresse   🕐 horaires   📞 telephone   📧 email\n"
    "   ✅ info cle  📅 date/agenda  🎓 ecole  👴 seniors  🏊 sport\n"
    "   🔗 lien utile  💡 conseil  ⚠️ attention  🎉 evenement\n"
    "4. Pour les horaires, les numeros de tel et adresses : une info par ligne\n"
    "5. Termine par une ligne de cloture aidante avec un emoji 😊 ou 👋\n"
    "6. Reponds en francais courant, sans jargon administratif\n"
    "7. Maximum 180 mots. Si besoin de plus, utilise des sections courtes\n"
    "8. NE JAMAIS utiliser de tirets (-) pour les listes, uniquement des emojis\n"
    "9. Gras **texte** pour mettre en valeur les infos critiques\n\n"

    "=== EXEMPLES DE BONNES REPONSES ===\n\n"

    "Question : Horaires mairie ?\n"
    "Reponse :\n"
    "Bonjour ! La mairie de Lambersart vous accueille :\n\n"
    "🕐 **Lundi – Jeudi :** 8h30 → 17h30\n"
    "🕐 **Vendredi :** 8h30 → 12h30\n\n"
    "📍 19 avenue Georges-Clemenceau, 59130 Lambersart\n"
    "📞 **03 20 08 44 44**\n"
    "📧 mairie@lambersart.fr\n\n"
    "💡 Pour un RDV carte d'identite ou passeport, appelez d'abord pour reserver votre creneau 😊\n\n"

    "Question : Job Day c'est quand ?\n"
    "Reponse :\n"
    "🎉 Le **Job Day 2026** approche !\n\n"
    "📅 **Mercredi 21 mai 2026**\n"
    "📍 Espace Jeunesse Honvault, Lambersart\n\n"
    "✅ Emploi, alternance, stages : tous les secteurs representes\n"
    "✅ Ouvert aux jeunes et aux demandeurs d'emploi\n\n"
    "👋 Venez avec votre CV ! Plus d'infos sur lambersart.fr/job-day-2\n\n"

    "=== BASE DE CONNAISSANCE ===\n"
    + KNOWLEDGE +
    "\n\n=== CONTEXTE DU SITE ===\n{context}"
)

URL_PATTERN = re.compile(r"https?://lambersart\.fr[^\s\])'\"]*")

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
        r = req_lib.get(url, timeout=10, headers={"User-Agent": "LambersartBot/2.0"})
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
    _mat = _vec.fit_transform([d["text"] for d in docs if d.get("text")])


def build(force=False):
    global _index
    # Charger knowledge.json en priorite
    kn = load_knowledge_index()

    if not force and INDEX_FILE.exists():
        if time.time() - INDEX_FILE.stat().st_mtime < INDEX_TTL:
            log.info("Index from cache")
            cached = {d["url"]: d for d in json.loads(INDEX_FILE.read_text(encoding="utf-8"))}
            _index = {**kn, **cached}
            fit()
            return

    log.info("Crawling lambersart.fr...")
    docs = []
    for url in PAGES:
        d = fetch(url)
        if d["text"]:
            docs.append(d)
        time.sleep(0.3)

    crawled = {d["url"]: d for d in docs}
    _index = {**kn, **crawled}
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
        if not d.get("text"):
            continue
        chunks.append("[" + d["url"] + "]\n" + d["text"][:800])
    return "\n\n---\n\n".join(chunks)


def mistral_call(messages, retries=3):
    for attempt in range(retries):
        try:
            client = Mistral(api_key=MISTRAL_API_KEY)
            if MISTRAL_V1:
                r = client.chat.complete(model=MODEL, messages=messages,
                                         temperature=0.2, max_tokens=600)
            else:
                from mistralai.models.chat_completion import ChatMessage
                r = client.chat(
                    model=MODEL,
                    messages=[ChatMessage(role=m["role"], content=m["content"]) for m in messages],
                    temperature=0.2, max_tokens=600
                )
            return r.choices[0].message.content
        except Exception as e:
            err = str(e)
            if "429" in err or "capacity" in err.lower() or "rate" in err.lower():
                wait = 2 ** attempt
                log.warning("Rate limit Mistral, retry %d/%d dans %ds", attempt+1, retries, wait)
                time.sleep(wait)
            else:
                raise
    return "Le service est momentanement surcharge. Reessayez dans quelques secondes ou appelez le 03 20 08 44 44."


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
