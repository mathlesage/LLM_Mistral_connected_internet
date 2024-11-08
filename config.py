import json
from pathlib import Path
import sys
chemin_script = Path(sys.argv[0]).resolve()
CHEMIN_COURANT = chemin_script.parent / ".config" /"config.json"
def get_api_config(file_path=CHEMIN_COURANT):
    """
    Lit le fichier de configuration JSON et renvoie les informations de l'API.

    :param file_path: Chemin du fichier de configuration JSON.
    :return: Dictionnaire contenant la clé API et l'endpoint de l'API.
    """
    try:
        with open(file_path, 'r') as f:
            config = json.load(f)
        return {
            'cle_google': config['cle_google'],
            'cle_mistral': config['cle_mistral'],
        }
    except FileNotFoundError:
        raise FileNotFoundError(f"Le fichier de configuration '{file_path}' est introuvable.")
    except json.JSONDecodeError:
        raise ValueError(f"Le fichier de configuration '{file_path}' est mal formé.")

# Exemple d'utilisation
if __name__ == "__main__":
    api_config = get_api_config()
