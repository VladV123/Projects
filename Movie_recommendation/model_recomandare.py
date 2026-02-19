import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import torch.nn as nn
from tqdm.notebook import tqdm
K = 5
ratings = pd.read_csv(r'/content/gdrive/MyDrive/IAOFU/ml-latest-small/ratings.csv')
n_users = ratings["userId"].nunique()
n_items = ratings["movieId"].nunique()
movies = pd.read_csv(r'/content/gdrive/MyDrive/IAOFU/ml-latest-small/movies.csv', low_memory=False)
movieid2info = {}
for index, row in movies.iterrows():
    movieid2info[row['movieId']] = (row['title'], row['genres'].split('|'))
movie_names = movies.set_index('movieId')['title'].to_dict()
print("numarul de users",n_users)
def train_test_split_by_user(ratings, test_size=0.2):
    train_data = []
    test_data = []

    for user_id in ratings['userId'].unique():
        user_ratings = ratings[ratings['userId'] == user_id]
        user_test_size = int(len(user_ratings) * test_size)
        user_test_data = user_ratings.sample(n=user_test_size, random_state=42)
        user_train_data = user_ratings.drop(user_test_data.index)

        train_data.append(user_train_data)
        test_data.append(user_test_data)

    train_data = pd.concat(train_data, ignore_index=True)
    test_data = pd.concat(test_data, ignore_index=True)

    return train_data, test_data
train_data, test_data = train_test_split_by_user(ratings, test_size=0.2)

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

def recommend_movies_for_user(model, user_id, seen_items, train_set, top_n=K):
    """
    Recomandă filme pentru un utilizator dat pe baza modelului de Matrix Factorization.

    :param model: Modelul antrenat de Matrix Factorization
    :param user_id: ID-ul utilizatorului pentru care facem recomandarile
    :param seen_items: Filmele pe care utilizatorul le-a vazut deja
    :param train_set: Setul de date folosit pentru antrenarea modelului
    :param top_n: Numărul de filme recomandate

    :return: Lista cu ID-urile filmelor recomandate
    """
    user_idx = train_set.userid2idx[user_id]
    user_factor = model.user_factors.weight[user_idx].detach().cpu().numpy()

    # Extragem factorii de film
    item_factors = model.item_factors.weight.detach().cpu().numpy()

    # Calculam scorul pentru fiecare film (produsul scalar dintre factorul utilizatorului si factorul filmului)
    scores = np.dot(item_factors, user_factor)  # Scorul pentru fiecare film pentru acest utilizator

    # Obtinem filmele pe care utilizatorul le-a vazut deja
    seen_item_indices = [train_set.movieid2idx[movie_id] for movie_id in seen_items]

    # Seteaza scorurile pentru filmele deja vizionate la -inf pentru a fi excluse din recomandari
    scores[seen_item_indices] = -np.inf

    # Obtinem top_n filmele cu cele mai mari scoruri
    recommended_item_indices = np.argsort(scores)[::-1][:top_n]

    # Convertim indicii recomandarilor în ID-uri de filme
    recommended_movie_ids = [train_set.idx2movieid[idx] for idx in recommended_item_indices]

    return recommended_movie_ids
num_epochs = 256
cuda = torch.cuda.is_available()
model = MatrixFactorizationWithFC(n_users=n_users, n_items=n_items)
print(model)
for name, param in model.named_parameters():
  if cuda: model = model.cuda()
  loss_fn = torch.nn.MSELoss()
if cuda:
  model= model.cuda()
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
train_set = Loader()
train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=4)

print("train loader",train_loader)
i=0
for it in tqdm(range(num_epochs)):
    i+=1
    print(i)
    losses = []
    for x, y in train_loader:
        if cuda: x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        # Split x into user and item tensors
        user = x[:, 0]  # Get user IDs from the first column of x
        item = x[:, 1]  # Get item IDs from the second column of x
        outputs = model(user, item) # Pass user and item to the model
        loss = loss_fn(outputs.squeeze(), y.type(torch.float32))
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    print("iteration #{} loss {:.4f}".format(it, sum(losses) / len(losses)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.save(model.state_dict(), '/content/gdrive/MyDrive/IAOFU/modele/MODEL_13ian_modificare_embedding_doar_utilizator_cu_update_gradient.pth')
loaded_model = MatrixFactorizationWithFC(n_users=n_users, n_items=n_items)
loaded_model.load_state_dict(torch.load('/content/gdrive/MyDrive/IAOFU/modele/MODEL_13ian_modificare_embedding_doar_utilizator_cu_update_gradient.pth', map_location=torch.device('cpu')))
loaded_model.to(device)

# Move inputs to the same device
user_idx = torch.tensor([1], device=device)
item_idx = torch.tensor([4275], device=device)
rating = torch.tensor([4.0], device=device)


# Define loss and optimizer
criterion = nn.MSELoss()
# optimizer = torch.optim.SGD(loaded_model.parameters(), lr= 1e-3)
params_to_optimize = [
    {'params': loaded_model.user_factors.parameters()},
    {'params': loaded_model.item_factors.parameters()}
]
optimizer = torch.optim.SGD(params_to_optimize, lr=0.01)

# Ensure that embeddings are on the same device
# If embeddings are stored in a layer, you can use the following to ensure they are on the correct device
loaded_model.user_factors = loaded_model.user_factors.to(device)
loaded_model.item_factors = loaded_model.item_factors.to(device)

# Call the update function for a single interaction
loss = loaded_model.update_user_embedding(user_idx=user_idx, item_idx=item_idx, rating=rating, optimizer=optimizer, criterion=criterion)
print(f"Loss: {loss}")

mse_values = []
mae_values = []
actual_ratings_list = []
predicted_ratings_list = []

for user_id in test_data['userId'].unique():
  seen_items_with_ratings = ratings[ratings['userId'] == user_id][['movieId', 'rating']]
  relevant_movies = ratings[(ratings['userId'] == user_id) & (ratings['rating'] >= 0)][['movieId', 'rating']].values.tolist()
  for (movie,rating) in relevant_movies:
    loaded_model.eval()
  with torch.no_grad():
      # Ensure user_id and movie are within the valid range for embeddings
      # Adjust user_id to be 0-indexed for the embedding layer.
      # Check if user_id is within the valid range before creating the tensor.
      if 0 <= user_id - 1 < loaded_model.user_factors.num_embeddings:
          user_id_tensor = torch.tensor([user_id - 1], dtype=torch.int64)

      # Check if movie is in train_set.movieid2idx and within the valid range for embeddings.
      if movie in train_set.movieid2idx and 0 <= train_set.movieid2idx[movie] < loaded_model.item_factors.num_embeddings:
          movie_id_tensor = torch.tensor([train_set.movieid2idx[movie]], dtype=torch.int64)

          # Move tensors to CUDA if available
          if cuda:
              user_id_tensor = user_id_tensor.cuda()
              movie_id_tensor = movie_id_tensor.cuda()

          prediction = loaded_model(user_id_tensor, movie_id_tensor)
          mse = (prediction.item() - rating)**2
          mae = abs(prediction.item() - rating)

          mse_values.append(mse)
          mae_values.append(mae)
          actual_ratings_list.append(rating)
          predicted_ratings_list.append(prediction.item())

          print("predict:", prediction.item(), "target:", rating)
      else:
          print(f"Movie ID {movie} not found in training set or out of range for embeddings.")

average_mse = np.mean(mse_values)
print("Average MSE: {:.3f}".format(average_mse))