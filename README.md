# 🏛️ Chatbot Lambersart × Mistral AI

Assistant numérique de recherche pour [lambersart.fr](https://lambersart.fr), propulsé par **Mistral AI** et une architecture RAG légère (crawl + TF-IDF).

---

## 🗂️ Structure du dépôt

```
lambersart-chatbot/
├── backend/
│   ├── app.py                  # Serveur Flask (API REST)
│   ├── requirements.txt        # Dépendances Python
│   └── tests/
│       └── test_app.py         # Tests unitaires pytest
├── frontend/
│   └── chatbot.html            # Widget flottant (standalone)
├── docs/
│   └── architecture.md         # Schéma d'architecture
├── .env.example                # Variables d'environnement à copier
├── .gitignore
├── Procfile                    # Déploiement Heroku / Render / Railway
└── README.md
```

---

## 🚀 Installation locale

### 1. Cloner le dépôt
```bash
git clone https://github.com/VilleLambersart/lambersart-chatbot.git
cd lambersart-chatbot
```

### 2. Créer l'environnement Python
```bash
python3 -m venv .venv
source .venv/bin/activate         # Windows : .venv\Scripts\activate
pip install -r backend/requirements.txt
```

### 3. Configurer les secrets
```bash
cp .env.example .env
nano .env   # Renseigner MISTRAL_API_KEY et ADMIN_TOKEN
```

### 4. Lancer le backend
```bash
cd backend
python app.py
# → http://localhost:5050
```

### 5. Ouvrir le widget
Ouvrez `frontend/chatbot.html` dans votre navigateur.  
Changez `const API = "http://localhost:5050/chat";` si besoin.

---

## 🌐 Déploiement en production

### Option A — Render.com (recommandé, gratuit)
1. Créer un compte sur [render.com](https://render.com)
2. Nouveau **Web Service** → connecter ce dépôt GitHub
3. Build command : `pip install -r backend/requirements.txt`
4. Start command : `gunicorn backend.app:app --bind 0.0.0.0:$PORT`
5. Variables d'environnement : `MISTRAL_API_KEY`, `ADMIN_TOKEN`, `ALLOWED_ORIGINS`

### Option B — Railway
```bash
railway init
railway up
railway variables set MISTRAL_API_KEY=sk-...
```

### Option C — VPS (Nginx + Gunicorn)
```bash
gunicorn backend.app:app --bind 127.0.0.1:5050 --workers 2 --daemon
# Puis configurer Nginx en reverse proxy vers 127.0.0.1:5050
```

---

## 🔌 Intégration sur lambersart.fr

Collez juste avant `</body>` dans votre template CMS :

```html
<script>
  window.LAMBERSART_CHAT_API = "https://assistant.lambersart.fr/chat";
</script>
<script src="https://votre-cdn/lambersart-chatbot.js" async defer></script>
```

---

## 🔒 Sécurité

| Mesure | Détail |
|--------|--------|
| Clé API hors code | Variable d'environnement `.env` (jamais committé) |
| CORS restreint | Seuls les domaines listés dans `ALLOWED_ORIGINS` |
| Rate limiting | 20 req/min par IP (Flask-Limiter) |
| Modération | Filtre regex des questions hors-sujet |
| Admin token | Route `/reindex` protégée par header `X-Admin-Token` |

---

## 🔄 Re-indexation automatique

Déclencher un recrawl via cron ou webhook :
```bash
curl -X POST https://assistant.lambersart.fr/reindex \
     -H "X-Admin-Token: VOTRE_TOKEN"
```

---

## 📄 Licence

MIT — Ville de Lambersart · Développé par Pierre Ciemniejewski
