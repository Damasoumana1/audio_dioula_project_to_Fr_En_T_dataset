# Projet Audio Dioula - Assistant IA Éducatif 🇧🇫

Ce projet vise à préparer le corpus de parole Dioula pour l'entraînement de modèles de reconnaissance vocale (Whisper) et de synthèse vocale, dans le cadre d'un assistant IA multilingue (Français, Moore, Dioula) pour le système éducatif au Burkina Faso.

## 📚 Source du Dataset
Le dataset utilisé est **Koumankan4Dyula**, fourni par l'**UVCI (Université Virtuelle de Côte d'Ivoire)**.
- **Lien Hugging Face** : [uvci/koumankan4dyula](https://huggingface.co/datasets/uvci/koumankan4dyula)
- **Contenu** : ~15 heures de parole, 10 929 enregistrements.
- **Langues** : Dioula (parlé), aligné avec Français et Anglais (écrit).

## 📁 Structure du Projet
Le projet suit une architecture stricte pour la gestion des données :
- `data/audio/wav/` : Fichiers audio extraits et convertis.
- `data/transcripts/` : Transcriptions organisées par langue (Dioula, Français, Anglais).
- `data/processed/` : Fichiers CSV contenant toutes les métadonnées (ID, genre, âge, pays).
- `src/` : Scripts de traitement et d'extraction.

## 🚀 Utilisation
1. **Installation** : `pip install -r requirements.txt`
2. **Extraction** : `python main.py`
3. **Statistiques** : `python src/statistics.py`

## 🎯 Objectifs du Projet
- Réduire la barrière linguistique dans l'éducation au Burkina Faso.
- Permettre aux enfants (6-15 ans) d'apprendre dans leur langue maternelle.
- Développer un assistant capable de fonctionner hors-ligne.

---
*Projet réalisé dans le cadre de mon Master sur le cours Intelligence Artificielle Appliquée au Contexte Africain (- Section Education).*

## 👨‍💻 Auteur
**Soumana Dama** (Ingénieur d'extraction du projet)
- 💼 LinkedIn : [Soumana Dama](https://www.linkedin.com/in/soumana-dama-445096253/)
- 🌐 Portfolio : [soumanadama.netlify.app](https://soumanadama.netlify.app/)

## 📜 Licence
Ce projet est sous licence **MIT** (Open Source), dans la continuité de l'aspect open source du dataset d'origine. Voir le fichier `LICENSE` pour plus de détails.
