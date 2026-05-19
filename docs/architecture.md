# Architecture du chatbot Lambersart

## Schéma global

```
Habitant (navigateur)
      │
      │  HTTPS
      ▼
Widget HTML flottant (frontend/chatbot.html)
      │
      │  POST /chat  { session_id, message }
      ▼
Backend Flask (backend/app.py)
      │
      ├─► Index TF-IDF local (lambersart_index.json)
      │     ↑ crawlé depuis lambersart.fr au démarrage
      │
      ├─► Récupération du contexte le plus pertinent (cosine similarity)
      │
      └─► API Mistral AI  (mistral-small-latest)
              │
              └─► Réponse avec citations + sources
                      │
                      ▼
            Widget → bulle de réponse + pills sources cliquables
```

## Flux d'une requête

1. L'habitant tape sa question dans le widget
2. Le widget envoie `POST /chat` au backend Flask
3. Le backend vectorise la question (TF-IDF)
4. Recherche cosine similarity dans l'index des pages lambersart.fr
5. Les 3 passages les plus proches sont injectés dans le prompt système
6. Mistral génère une réponse ancrée sur ce contexte
7. Le backend extrait les URLs citées et les renvoie comme `sources`
8. Le widget affiche la réponse + des pills cliquables vers les pages

## Technologies

| Composant | Technologie |
|-----------|-------------|
| LLM | Mistral AI (mistral-small-latest) |
| Backend | Flask 3 + Gunicorn |
| RAG | scikit-learn TF-IDF + cosine similarity |
| Scraping | requests + BeautifulSoup4 |
| CORS | flask-cors |
| Rate limit | flask-limiter |
| Frontend | HTML/CSS/JS vanilla (zéro dépendance) |
| CI/CD | GitHub Actions |
