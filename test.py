
import requests


def get_google_search_results(mot_cle: str,num_results: int, cle_google=cle_google) -> list[str]:
    """Fonction qui renvoie des url à partir de mot clée en utilisant l'API google. Attention 100 requete par jour maximum.

    Args:
        mot_cle (str): _description_
        num_results (int): Nombre d'url voulu

    Returns:
        list[str]: liste d'url
    """   
    api_key = cle_google
    cx = "40a2133a6472a4f88"
    
    mot_cle = mot_cle.replace(" ", "+")  # Gérer les espaces dans la requête
    search_url = f"https://www.googleapis.com/customsearch/v1?q={mot_cle}&num={num_results}&key={api_key}&cx={cx}"
    response = requests.get(search_url)
    response.raise_for_status()
    results = response.json()
    links = []
    if 'items' in results:
        for item in results['items']:
            links.append(item['link'])
            if len(links) >= num_results:
                break
    return links
