import os
import csv
import requests
import pandas as pd
# Calea catre folderul unde se vor salva imaginile
save_folder = r'C:\Users\Ruxi\Desktop\Recomandare_filme\static\images'
csv_path = r'C:\Users\Ruxi\Desktop\Recomandare_filme\ml-latest-small\links.csv'
df = pd.read_csv(csv_path)
tmdb_to_movie_dict = dict(zip(df['tmdbId'], df['movieId']))
tmdb_id_cautat = 862
movie_id = tmdb_to_movie_dict.get(tmdb_id_cautat, "MovieID nu a fost găsit")
print("Movieid", movie_id, tmdb_id_cautat)

def download_movie_image(tmdb_id):
    # Obtinem movieId asociat cu tmdbId
    movie_id = tmdb_to_movie_dict.get(int(tmdb_id), "MovieID nu a fost găsit")
    if not movie_id:
        print(f"Nu există movieId pentru tmdbId: {tmdb_id}")
        return
    if movie_id>2763:

        print(" muvie, tmdb_id", movie_id, int(tmdb_id))
        # Construim URL-ul TMDb pentru imagine
        url = f" "
        response = requests.get(url)
        # Continuam doar daca raspunsul este valid
        if response.status_code != 200:
            print(f"Eroare la accesarea paginii pentru TMDb ID {tmdb_id}: {response.status_code}")
            return
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        image_tag = soup.find('img', {'class': 'poster'})
        image_url = None
        if not image_tag:
            image_tag = soup.find('div', {'class': 'blurred'})
            if image_tag:
                # Extragem URL-ul imaginii din stilul CSS (background-image)
                style = image_tag['style']
                image_url = style.split('url(')[1].split(')')[0].strip("'")
        else:
            # Extragem URL-ul imaginii din atributul 'src' al tag-ului <img>
            image_url = image_tag['src']
        img_name = f"{movie_id}.jpg"
        img_path = os.path.join(save_folder, img_name)
        if os.path.exists(img_path):
            print(f"Imaginea pentru movieId {movie_id} există deja. Se trece peste descărcare.")
            return
        # Descarcam imaginea
        if image_url:
            print(f"Descarc imaginea pentru movieId: {movie_id} de la {image_url}")
            img_response = requests.get(image_url)

            # Numele fisierului este movieId.jpg
            img_name = f"{movie_id}.jpg"
            img_path = os.path.join(save_folder, img_name)

            os.makedirs(save_folder, exist_ok=True)
            with open(img_path, 'wb') as file:
                file.write(img_response.content)
            print(f"Imaginea pentru movieId {movie_id} a fost salvată ca {img_path}")
        else:
            print(f"Nu am găsit imaginea pentru filmul TMDb ID: {tmdb_id}")
import time
with open(r'C:\Users\Ruxi\Desktop\Recomandare_filme\ml-latest-small\links.csv', mode='r') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        tmdb_id = row['tmdbId']  # Extrage tmdbId din CSV
        print(tmdb_id)
        if tmdb_id:
            download_movie_image(tmdb_id)
        else:
            print("tmdbId este gol, trecem peste această linie.")
