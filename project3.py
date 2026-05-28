# ====================================================================
# DecodeLabs AI Internship - Project 3: Content-Based Recommender
# Student: Dridi Jawher
# ====================================================================

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("--- Engine Recommendation System v1.0 ---")

# Base de donnees des cours disponibles
les_cours = [
    {"id": 101, "nom": "Advanced Python Development", "keywords": "python backend code automation scripting development"},
    {"id": 102, "nom": "Deep Learning and Neural Networks", "tags": "ai artificial intelligence neural networks tensors deep learning optimization machine learning"},
    {"id": 103, "nom": "Frontend Web Design & UI/UX", "keywords": "web design frontend html css javascript UI UX application layout"},
    {"id": 104, "nom": "Introduction to Machine Learning", "keywords": "ai machine learning supervised learning algorithms python statistics data"},
    {"id": 105, "nom": "Data Analytics & Cloud Pipelines", "keywords": "data analytics cloud pipelines sql databases automation infrastructure"}
]

# Creation du DataFrame pandas
data_cours = pd.DataFrame(les_cours)

# Capture des interets de l'etudiant
print("\n[!] Entrez vos competences ou interets (ex: python, web design...)")
mon_choix = input(">> Vos interets: ").lower().strip()

# Securite si l'utilisateur ne tape rien
if not mon_choix:
    print("[!] Input vide! Utilisation du choix par defaut: 'ai python'")
    mon_choix = "ai python"

# Fusion du choix de l'utilisateur avec la base globale
liste_totale = list(data_cours['keywords']) + [mon_choix]

# Vectorisation avec TF-IDF pour calculer les poids des mots
mon_vectorizer = TfidfVectorizer()
matrice_tfidf = mon_vectorizer.fit_transform(liste_totale)

# Calcul du score de similarite via Cosine Similarity
calcul_similarite = cosine_similarity(matrice_tfidf[-1], matrice_tfidf[:-1]).flatten()

# Ajout des scores dans notre tableau
data_cours['score'] = calcul_similarite

# Tri des cours du plus proche au moins proche
resultats_tri = data_cours.sort_values(by='score', ascending=False)

print("\n==================================================")
print("🎯 COURS RECOMMANDES POUR VOUS :")
print("==================================================")

trouve = False
for index, ligne in resultats_tri.iterrows():
    # Affichage des cours qui ont une correspondance (> 0)
    if ligne['score'] > 0:
        print(f"⭐ {ligne['nom']}")
        print(f"   ↳ Taux de correspondance: {ligne['score']*100:.1f}%")
        print(f"   ↳ Tags: {ligne['keywords']}\n")
        trouve = True

# Strategie de secours (Fallback) si aucun mot ne match
if not trouve:
    print("🤖 Aucun resultat exact trouve.")
    print("   Voici le cours le plus populaire actuellement:")
    print(f"   🔥 {data_cours.iloc[0]['nom']} (Recommandation generale)")
print("==================================================")