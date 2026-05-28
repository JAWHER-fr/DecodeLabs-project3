# Project 3: AI Recommendation System Engine

## Domain: Artificial Intelligence Track
**Developed by:** Dridi Jawher  
**Framework Layer:** Content-Based Filtering Engine  

## System Architectural Design
Ce système implémente un pipeline de recommandation basé sur le contenu (Content-Based Filtering). Contrairement au simple comptage binaire de mots-clés, l'algorithme normalise les entrées textuelles brutes et les projette dans un espace vectoriel multidimensionnel à l'aide de **TF-IDF (Term Frequency-Inverse Document Frequency)**, puis calcule l'alignement directionnel via la **Similarité Cosinus**.

### Pipeline de l'Architecture :
1. **Input Layer :** Capture et nettoie les intérêts ou compétences saisis par l'utilisateur (`.lower().strip()`).
2. **TF-IDF Processing :** Extrait les poids vectoriels en pénalisant les termes génériques fréquents et en valorisant les tags techniques spécifiques.
3. **Similarity Engine :** Calcule le score de similarité cosinus entre le profil utilisateur et la base de données des cours.
4. **Fallback Strategy :** Gère les cas de recherche sans correspondance en proposant le cours le plus populaire par défaut pour garantir la continuité du service.

## Structure des Fichiers
* `project3.py` - Le code source principal de l'application.
* `README.md` - Documentation du projet.

## Instructions de Déploiement
Installez les bibliothèques requises :
```bash
pip install scikit-learn pandas
