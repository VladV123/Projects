document.addEventListener('DOMContentLoaded', () => {
  // Cautam toate butoanele de trimitere a ratingului
  const submitButtons = document.querySelectorAll('.submit-rating');

  submitButtons.forEach(button => {
    button.addEventListener('click', function() {
      // Preia movieId din atributul data-movie-id al butonului
      const movieId = this.dataset.movieId;

      // Cautam valoarea ratingului selectat pentru filmul respectiv
      const selectedRadio = document.querySelector(`input[name="rating-${movieId}"]:checked`);

      if (selectedRadio) {
        const ratingValue = selectedRadio.value; // Valoarea ratingului selectat

        console.log(`Rating pentru filmul cu ID-ul ${movieId}: ${ratingValue}`);
        const USERNAME = document.body.dataset.username; // Preluam username-ul din atributul data-username
        console.log(`Username-ul utilizatorului: ${USERNAME}`);

        const data = {
          user_name: USERNAME,  // Trimitem username-ul catre server
          movie_id: movieId,
          rating: ratingValue
        };

        // Trimitem datele catre server, route-ul de backend care se ocupa cu salvarea în CSV
        fetch('/rate_movie', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(data)
        })
        .then(response => response.json())
        .then(data => {
          alert(data.message);
        })
        .catch(error => console.error('Error:', error));
      } else {
        alert("Te rugăm să selectezi un rating înainte de a trimite!");
      }
    });
  });
});
