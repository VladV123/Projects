from flask import Flask, render_template, redirect, url_for, session
import numpy as np
from flask import request, jsonify
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data.dataset import Dataset
import pandas as pd
import os
import torch.nn as nn
import time
app = Flask(__name__)
app.secret_key = "your_secret_key"

ratings = pd.read_csv(r"C:\Users\Ruxi\Desktop\Recomandare_filme\ml-latest-small\ratings.csv")
numar_utilizatori = pd.read_csv(r"C:\Users\Ruxi\Desktop\Recomandare_filme\utilizatori_parole.csv")
n_users = numar_utilizatori["userId"].nunique()
print("Numar utilizatori", n_users)
n_items = ratings["movieId"].nunique()
movies = pd.read_csv(r"C:\Users\Ruxi\Desktop\Recomandare_filme\ml-latest-small\movies.csv", low_memory=False)
toate_filmele = movies['title'].unique()
movieid2info = {}
for index, row in movies.iterrows():
    movieid2info[row['movieId']] = (row['title'], row['genres'].split('|'))
utilizatori = pd.read_csv('utilizatori_parole.csv')  # Baza de date de utilizatori

class MatrixFactorizationWithFC(nn.Module):
    def __init__(self, n_users, n_items, n_factors=64, fc_hidden=64):
        super().__init__()
        # Matrix Factorization
        self.user_factors = nn.Embedding(n_users, n_factors, sparse=False)
        self.item_factors = nn.Embedding(n_items, n_factors, sparse=False)

        self.user_factors.weight.data.uniform_(0, 0.05)
        self.item_factors.weight.data.uniform_(0, 0.05)

        # Fully Connected Layers with Dropout
        self.fc = nn.Sequential(
            nn.Linear(n_factors * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, user, item):
        user_emb = self.user_factors(user)
        item_emb = self.item_factors(item)

        combined_emb = torch.cat([user_emb, item_emb], dim=1)
        output = self.fc(combined_emb).squeeze(1)
        return output

    def update_user_embedding(self, user_idx, item_idx, rating, optimizer, criterion):
        """
        Actualizam embedding-urile unui utilizator si ale unui item pe baza unei singure interactiuni.
        :param user_idx: Indexul utilizatorului
        :param item_idx: Indexul item-ului
        :param rating: Rating-ul real acordat de utilizator item-ului (float).
        :param optimizer: Optimizator pentru a actualiza parametrii modelului.
        :param criterion: Functie de pierdere MSELoss
        """
        # Convertim indicii in tensori
        user_idx_tensor = torch.tensor([user_idx], dtype=torch.long)
        item_idx_tensor = torch.tensor([item_idx], dtype=torch.long)

        for param in self.fc.parameters():
            param.requires_grad = False
        # Se seteaza la zero gradientele
        optimizer.zero_grad()
        # Forward pass
        predicted_rating = self.forward(user_idx_tensor, item_idx_tensor)
        print("Predicted rating grad_fn:", predicted_rating.grad_fn)
        loss = criterion(predicted_rating, torch.tensor([rating], dtype=torch.float32))
        print("Loss grad_fn:", loss.grad_fn)
        loss.backward()
        optimizer.step()
        for param in self.fc.parameters():
          param.requires_grad = True
        return loss.item()

    def add_new_user(self, n_users, n_factors):
        """
        Adaugam un utilizator nou în model si extindem matricea embedding-urilor
        :param n_users: Numarul curent de utilizatori
        :param n_factors: Numarul de factori latenti
        :return: Model actualizat, n_users incrementat
        """
        # Noul embedding extins pentru utilizatori
        new_user_factors = nn.Embedding(n_users + 1, n_factors)
        with torch.no_grad():
            new_user_factors.weight[:-1] = self.user_factors.weight
            new_user_factors.weight[-1].uniform_(0, 0.05)  # Initializam noul embedding aleatoriu

        # Inlocuim embedding-ul vechi
        self.user_factors = new_user_factors

        n_users += 1
        return self, n_users

class Loader(Dataset):
    def __init__(self, ratings):
        self.ratings = ratings
        users = ratings.userId.unique()
        movies = ratings.movieId.unique()
        # Maparea utilizatorilor si filmelor la indici
        self.userid2idx = {o: i for i, o in enumerate(users)}
        self.movieid2idx = {o: i for i, o in enumerate(movies)}
        self.idx2userid = {i: o for o, i in self.userid2idx.items()}
        self.idx2movieid = {i: o for o, i in self.movieid2idx.items()}

        self.x = torch.tensor(self.ratings[['userId', 'movieId']].values)
        self.y = torch.tensor(self.ratings['rating'].values)

    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.ratings)

model = MatrixFactorizationWithFC(n_users, n_items, n_factors=64)
model.load_state_dict(torch.load("MODEL_13ian_modificare_embedding_doar_utilizator_cu_update_gradient.pth", map_location=torch.device('cpu')))
train_set = Loader(ratings)

def recommend_movies_for_new_user(new_user_vector, model, train_set, top_n=5):
    existing_user_factors = model.user_factors.weight.detach().cpu().numpy()
    similarities = cosine_similarity([new_user_vector], existing_user_factors)[0]
    most_similar_user_idx = np.argsort(similarities)[::-1][0]
    similar_user_movies = train_set.ratings[train_set.ratings.userId == most_similar_user_idx].movieId.values
    similar_user_movies = similar_user_movies.tolist()
    recommendations = []
    for movie_idx in similar_user_movies:
        movie_id = train_set.idx2movieid[movie_idx]
        if movie_id in movieid2info:
            movie_title, genres = movieid2info[movie_id]
            recommendations.append((movie_id, movie_title, genres))
        if len(recommendations) == top_n:
            break
    return [rec[0] for rec in recommendations]
@app.route("/preferences/<username>", methods=["GET", "POST"])
def preferences(username):
    # Calculul filmelor cu rating bayesian
    # https://github.com/topspinj/tmls-2020-recommender-workshop/blob/master/tutorial.ipynb
    C = ratings['rating'].mean()
    m = 10
    movie_stats = ratings.groupby('movieId').agg(count=('rating', 'size'), mean=('rating', 'mean')).reset_index()
    movie_stats['bayesian_avg'] = (movie_stats['count'] * movie_stats['mean'] + m * C) / (movie_stats['count'] + m)
    movie_stats = movie_stats.merge(movies[['movieId', 'title', 'genres']], on='movieId')
    top_movies_details = movie_stats.sort_values('bayesian_avg', ascending=False).head(15)
    movies_list = [
        {
            'id': row['movieId'],
            'title': row['title'],
            'genres': row['genres'],
            'rating': round(row['bayesian_avg'], 2),
            'image': f"images/{row['movieId']}.jpg"
        }
        for _, row in top_movies_details.iterrows()
    ]
    if request.method == "POST":
        # Salvam filmele selectate ca fiind vazute
        selected_movies = request.form.getlist("movies")
        session["selected_movies"] = selected_movies
        new_user_ratings = [(int(movie_id), 5.0) for movie_id in selected_movies]
        session["new_user_ratings"] = new_user_ratings
        utilizatori = pd.read_csv('utilizatori_parole.csv')  # Baza de date de utilizatori
        user_row = utilizatori[utilizatori['Utilizator'] == username]
        print("user_row",user_row)
        if not user_row.empty:
            user_id = int(user_row['userId'].values[0])
            for movie_id, rating in new_user_ratings:
                save_rating(user_id, movie_id, 5)
                print("S-a salvat utilizatorul in csv ratings",user_id)
        return redirect(url_for("recommendations", username=username))
    return render_template("preferences.html", movies=movies_list, username=username)

def recommend_movies_for_user(model, user_id, seen_movies, train_set, top_n=5):
    """
    Recomandam filme pentru un utilizator pe baza modelului de Matrix Factorization
    :param model: Modelul antrenat de Matrix FactorizationWithFC
    :param user_id: ID ul utilizatorului pentru care facem recomandarile
    :param seen_items: Filmele pe care utilizatorul le a vazut deja (ID uri filme)
    :param train_set: Setul de date folosit pentru antrenarea modelului
    :param top_n: Numarul de filme recomandate
    :return: Lista cu ID-urile filmelor recomandate
    """
    ratings = pd.read_csv(r"C:\Users\Ruxi\Desktop\Recomandare_filme\ml-latest-small\ratings.csv")
    train_set = Loader(ratings)
    # print("Maparea filmelor",train_set.movieid2idx)
    print("Lungime train",len(train_set))
    # Obtinem indexul utilizatorului in functie de ID ul acestuia
    user_idx = train_set.userid2idx[user_id]
    print('user_idx in reco ', user_idx, user_id)
    # Extrage toate filmele disponibile
    all_movies = list(train_set.movieid2idx.values())  # Extragem indicii (movie_idx) ale filmelor
    # Obtinem filmele pe care utilizatorul le a vazut deja
    seen_item_indices = [train_set.movieid2idx[movie_id] for movie_id in seen_movies]
    # Filmele nevazute de utilizator (sunt toate filmele care nu sunt în `seen_item_indices`)
    unseen_item_indices = list(set(all_movies) - set(seen_item_indices))
    # Calculam ratingurile pentru filmele nevazute
    scores = []
    with torch.no_grad():
        for movie_idx in unseen_item_indices:
            try:
                # Validare `movie_idx` pentru embedding
                if not (0 <= movie_idx < model.item_factors.num_embeddings):
                    print(f"Movie index {movie_idx} out of range for embedding.")
                    continue  # Daca indicele este invalid se sare
                # Prezicem ratingul folosind modelul (validam ca movie_idx e valid)
                user_tensor = torch.tensor([user_idx], dtype=torch.long)
                movie_tensor = torch.tensor([movie_idx], dtype=torch.long)
                predicted_rating = model(user_tensor, movie_tensor).item()  #ratingul prezis
                scores.append((movie_idx, predicted_rating))
            except Exception as e:
                print(f"Error while predicting rating for movie {movie_idx}: {e}")
    # Sortam filmele pe baza scorurilor (ratingurilor prezise)
    scores.sort(key=lambda x: x[1], reverse=True)
    # Extragem top_n filmele recomandate
    recommended_item_indices = [score[0] for score in scores[:top_n]]
    recommended_ratings = [score[1] for score in scores[:top_n]]
    # Convertim indicii recomandarilor in ID uri de filme
    recommended_movie_ids = [train_set.idx2movieid[idx] for idx in recommended_item_indices]
    # Detaliile filmelor recomandate din DataFrame ul de filme
    recommended_movies = movies[movies['movieId'].isin(recommended_movie_ids)]
    recommended_movies.loc[:, 'predicted_rating'] = recommended_ratings
    # titlurile genurile și ratingurile prezise ale filmelor recomandate
    print(recommended_movies[['movieId', 'title', 'genres', 'predicted_rating']])
    return recommended_movie_ids

@app.route('/')
def home():
    return render_template('home.html')  # Noua pagina principala

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if username and password:
            if os.path.exists(USERS_FILE):
                df = pd.read_csv(USERS_FILE)
                user_row = df[df['Utilizator'] == username]

                if not user_row.empty and user_row['Parola'].values[0] == password:
                    return redirect(url_for('recommendations', username=username))
                else:
                    return render_template('error.html', error="Utilizator sau parolă incorectă!")
            else:
                return render_template('error.html', error="Nu există utilizatori înregistrați!")
        else:
            return render_template('error.html', error="Toate câmpurile sunt obligatorii!")
    return render_template('login.html')

@app.route('/recommendations/<username>', methods=['GET', 'POST'])
def recommendations(username):
    if request.method == 'POST':
        # reincarcam modelul
        model.load_state_dict(torch.load("MODEL_13ian_modificare_embedding_doar_utilizator_cu_update_gradient.pth",
                                         map_location=torch.device('cpu')))

        model.eval()
        utilizatori = pd.read_csv('utilizatori_parole.csv')  # Baza de date de utilizatori
        ratings = pd.read_csv(r"C:\Users\Ruxi\Desktop\Recomandare_filme\ml-latest-small\ratings.csv")
        user_id = utilizatori.loc[utilizatori['Utilizator'] == username, 'userId'].values[0]

        train_set = Loader(ratings)
        print(train_set.userid2idx)
        print("Id-ul utilizatorului",user_id)
        user_idx = train_set.userid2idx[user_id]
        print("Indexul utilizatorului",user_idx)
        print("Primul factor latent al unui utilizator", model.user_factors.weight[user_idx])

        # Obtinem filmele vazute de utilizator si ratingurile lor
        seen_movies = ratings[ratings['userId'] == user_id]
        seen_movie_details = pd.merge(
            seen_movies,
            movies,
            on='movieId',
            how='inner'
        )[['movieId', 'title', 'genres', 'rating']]
        print("Detalii despre filmele vizualizate",seen_movie_details)
        print("filme vazute",len(seen_movie_details))
        # Selectam primele 5 filme vizualizate, sortate dupa rating descrescator
        top_seen_movie_details = (
            seen_movie_details
            .sort_values(by='rating', ascending=False)
            .head(5)
            .to_dict(orient='records')
        )
        print("qqq", seen_movies['movieId'])
        # Generam recomandari (obtinem ID-urile filmelor recomandate)
        recommended_movie_ids = recommend_movies_for_user(model, user_id, seen_movies['movieId'].tolist(), train_set)

        # Obtinem detaliile despre filmele recomandate
        recommended_movies = movies[movies['movieId'].isin(recommended_movie_ids)]
        # Obține detaliile despre filmele recomandate
        recommended_movie_details = recommended_movies[['movieId', 'title', 'genres']].to_dict(orient='records')
        # Trimite datele catre sablon
        return render_template(
            'recommendations.html',
            username=username,
            movies=recommended_movie_details,
            seen_movies=top_seen_movie_details
        )

    return render_template('recommendations.html', username=username)

def find_movie_by_id(movie_id):
    for movie in toate_filmele:
        if movie['movieId'] == toate_filmele[movie_id]:
            return movie
    return None

RATINGS_FILE = r"C:\Users\Ruxi\Desktop\Recomandare_filme\ml-latest-small\ratings.csv"
def save_rating(user_name, movie_id, rating_value):
    if os.path.exists(RATINGS_FILE):
        try:
            df = pd.read_csv(RATINGS_FILE)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame(columns=['userId', 'movieId', 'rating', 'timestamp'])
    else:
        df = pd.DataFrame(columns=['userId', 'movieId', 'rating', 'timestamp'])
    timestamp = pd.to_datetime('now').strftime('%Y-%m-%d %H:%M:%S')
    new_row = {
        'userId': user_name,
        'movieId': movie_id,
        'rating': rating_value,
        'timestamp': timestamp
    }
    df = df.append(new_row, ignore_index=True)
    df.to_csv(RATINGS_FILE, index=False)

@app.route('/rate_movie', methods=['POST'])
def rate_movie():
    data = request.json
    user_name = data.get('user_name')
    movie_id = data.get('movie_id')
    rating_value = data.get('rating')

    if user_name and movie_id and rating_value:
        utilizatori = pd.read_csv('utilizatori_parole.csv')
        ratings = pd.read_csv(r"C:\Users\Ruxi\Desktop\Recomandare_filme\ml-latest-small\ratings.csv")
        train_set = Loader(ratings)
        # Baza de date de utilizatori
        user_row = utilizatori[utilizatori['Utilizator'] == user_name]
        if not user_row.empty:
            user_id = int(user_row['userId'].values[0])
        save_rating(user_id, movie_id, rating_value)
        print("Date csv", user_id, movie_id, rating_value)

        user_idx = train_set.userid2idx[int(user_id)]
        rating = torch.tensor([float(rating_value)])
        criterion = nn.MSELoss()
        params_to_optimize = [
            {'params': model.user_factors.parameters()},
            {'params': model.item_factors.parameters()}
        ]
        optimizer = torch.optim.SGD(params_to_optimize, lr=0.001)
        indexul_filmului = train_set.movieid2idx[int(movie_id)]
        if indexul_filmului >= 0 or indexul_filmului <= model.item_factors.num_embeddings:
            print(f"{indexul_filmului} Indexul Filmului {movie_id}")

        loss = model.update_user_embedding(user_idx=user_idx, item_idx=indexul_filmului, rating=rating, optimizer=optimizer, criterion=criterion)
        print(f"Loss inainte: {loss}")

        while loss>0.5:
            loss = model.update_user_embedding(user_idx=user_idx, item_idx=indexul_filmului, rating=rating,
                                                      optimizer=optimizer, criterion=criterion)
            print(f"Loss: {loss}")
        torch.save(model.state_dict(), f'MODEL_13ian_modificare_embedding_doar_utilizator_cu_update_gradient.pth')
        return jsonify({'message': 'Rating saved successfully!!!!!!!!'}), 200
    else:
        return jsonify({'message': 'Missing parameters!'}), 400

def login_user(username, password):
    """
    Verifica daca utilizatorul si parola exista in fisierul 'utilizatori_parole.csv'
    :param username: Numele de utilizator
    :param password: Parola
    :return: True daca utilizatorul si parola sunt corecte, altfel False
    """
    user = utilizatori[utilizatori['Utilizator'] == username]
    if not user.empty and user['Parola'].values[0] == password:
        return True
    return False

USERS_FILE = 'utilizatori_parole.csv'
def save_user(username, password):
    if os.path.exists(USERS_FILE):
        try:
            df = pd.read_csv(USERS_FILE)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame(columns=['userId', 'Utilizator', 'Parola'])
    else:
        df = pd.DataFrame(columns=['userId', 'Utilizator', 'Parola'])
    if username in df['Utilizator'].values:
        return False, "Utilizatorul există deja."
    # IDul unic pentru utilizator
    new_user_id = df['userId'].max() + 1 if not df.empty else 1
    # Adaugam noul utilizator
    new_row = {
        'userId': new_user_id,
        'Utilizator': username,
        'Parola': password
    }
    df = df.append(new_row, ignore_index=True)
    df.to_csv(USERS_FILE, index=False)
    return True, "Utilizator adăugat cu succes."

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    global model, n_users
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if username and password:
            success, message = save_user(username, password)
            if success:
                print("n_users", n_users)
                model, n_users = model.add_new_user(n_users=n_users, n_factors=64)
                num_users = model.user_factors.weight.shape[0]
                print(f"Numărul de utilizatori în modelul nou: {num_users}")
                torch.save(model.state_dict(), f'MODEL_13ian_modificare_embedding_doar_utilizator_cu_update_gradient.pth')
                print("modelor",model.user_factors.weight[num_users-1])
                return redirect(url_for('preferences', username=username))
            else:
                return render_template('error.html', message=message)
        else:
            return render_template('error.html', message="Toate câmpurile sunt obligatorii!")
    return render_template('signup.html', username='', password='')

if __name__ == "__main__":
    app.run(debug=True)
