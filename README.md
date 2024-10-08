# LLM_Mistral_connected_internet

Ce projet est un système de recherche d'information basé sur la méthode du **Retrieval-Augmented Generation (RAG)**, conçu pour interroger des données directement trouvées sur Internet et fournir des réponses précises à l'aide des modèles **Mistral**. Le processus consiste à sélectionner les informations les plus pertinentes à l'aide de la cos-similarité et d'un modèle d'embedding, puis de les faire passer dans un LLM pour générer des réponses complètes, accompagnées des sources et des URL des sites d'où proviennent les données.


### Fonctionnalités principales

  Recherche d'information : Le système récupère les données en ligne à partir de requêtes spécifiques.
  
  Cos-similarité et embeddings : Les chunks de texte les plus pertinents sont sélectionnés en fonction de leur similarité avec la requête.
  
  Modèle LLM (Mistral) : Utilisation du modèle Mistral pour générer des réponses en langage naturel à partir des données récupérées.
  
  Retour des sources : Chaque réponse est accompagnée des sources pertinentes, y compris les URL des sites consultés.

  
  
  Deux versions :
      mistral_internet.py : Version basique en ligne de commande.
      interface_ia.py : Version avec interface graphique via Streamlit.

### Prérequis

Avant d'utiliser ce projet, assurez-vous d'avoir installé les dépendances suivantes :

  Python 3.x
  Packages requis (disponibles dans requirements.txt)

Pour installer les dépendances :


```
pip install -r requirements.txt
```

Utilisation
Version en ligne de commande (mistral_internet.py)

  Exécutez le script directement dans votre terminal :

    python mistral_internet.py

  Entrez votre requête. Le système récupérera les informations pertinentes, générera une réponse et vous fournira les sources.

Version avec interface graphique (interface_ia.py)

  Lancez l'interface Streamlit :

    python -m streamlit run interface_ia.py

  Une interface graphique s'ouvrira dans votre navigateur, où vous pourrez entrer vos requêtes et visualiser les réponses avec les sources.

### Clés d'API requises

Pour utiliser ce projet, vous aurez besoin de deux clés d'API :

  Clé Google : Une clé gratuite est nécessaire pour effectuer les 100 premières requêtes. Vous pouvez obtenir cette clé via la Console Google Cloud.

  Clé Mistral : Une clé d'API pour Mistral est également requise. Veuillez suivre les instructions fournies par Mistral pour obtenir cette clé.

### Structure du projet

  mistral_internet.py : Script de la version en ligne de commande.
  interface_ia.py : Script de la version avec interface Streamlit.
  requirements.txt : Liste des dépendances Python nécessaires.
  README.md : Ce fichier.

### Contribuer

Les contributions sont les bienvenues ! N'hésitez pas à soumettre des pull requests ou à signaler des problèmes via les issues.


Ce projet est sous licence [MIT](https://github.com/mathlesage/Mistral_connected_internet/blob/main/LICENSE).
