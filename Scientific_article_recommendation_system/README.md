#  Scientific Article Recommendation System (HybridNeuMF)

##  Project Overview
This project presents the design and implementation of an intelligent recommendation system tailored for academic researchers. To solve the problem of information overload in the academic environment, we developed a **Hybrid Neural Matrix Factorization (HybridNeuMF)** architecture

Unlike traditional methods that rely solely on reading history or keywords, our system combines Collaborative Filtering (user-item interactions) with Content-Based Filtering (abstract texts, social tags, and citation networks). This hybrid approach effectively addresses the "cold start" problem for newly published articles

---

##  Tech Stack & Architecture
* **Core Language & Frameworks:** Python, PyTorch (`torch`), scikit-learn, pandas
* **NLP & Embeddings:** `Sentence-BERT` (specifically the `intfloat/multilingual-e5-large-instruct` model) for generating 1068-dimensional dense semantic embeddings from article titles and abstracts
* **Data Encoding:** Multi-Hot Encoding for tags and citations, stored in sparse matrices for memory optimization
* **Environment:** Google Colab (GPU/CUDA acceleration)

### Neural Network Architecture (HybridNeuMF)
The model processes latent and content features in parallel through two branches
1.**MLP Branch (Side Information):** Concatenates user/item embeddings with text vectors and linear projections of tags/citations, passing them through dense layers (256 -> 128 -> 64) with Batch Normalization, ReLU, and Dropout (0.5)
2. **MF Branch (Matrix Factorization):** Models direct linear interactions via the dot product of latent vectors.
Both branches converge into a final Sigmoid layer to output a relevance probability score

##  Dataset & Training
**Dataset:** We utilized the **CiteULike-t** dataset, processing interactions from 7,947 users and 25,975 articles
**Strategy:** Implemented a Negative Sampling strategy (1:4 ratio) to help the model distinguish between relevant and irrelevant content.
**Training Results:** Over 10 training epochs using the Adam optimizer, the validation Binary Cross Entropy (BCE) Loss dropped significantly to **0.2198**, demonstrating robust convergence without overfitting

---

##  My Research Focus & Contribution: Evaluation Methodologies
As part of the research team, my primary focus was analyzing and defining the correct evaluation strategies based on dataset constraints, specifically comparing **CiteULike** vs. **ERIC**.

* **Supervised Evaluation (CiteULike):** Because CiteULike provides collaborative data (user interaction history), I advocated for historical simulation using **Precision@K** and **NDCG** metrics to measure how well the algorithm predicts actual human preferences[cite: 144, 159, 164, 166]. 
* **Unsupervised Evaluation (ERIC):** For datasets like ERIC (which contain text but no user data), I demonstrated that classical precision metrics fail.Instead, I proposed measuring semantic coherence using **SciBERT semantic similarity** (Cosine distance) and keyword retrieval overlap using the **Jaccard Index**

*Conclusion:* The architecture and success metrics of a recommender system must be strictly dictated by the nature of the available data
