# Bayes Factor Test 

## How to run

1. Clone the repository
   
   git clone https://github.com/Kelly-Yin-99/repo.git


2. Go to the bayes_factor folder:

   cd bayes_factor


3. Build the Docker image:

   docker build -t bayes-test .


4. Run the tests through Docker:

   docker run --rm bayes-test
   

5. Run locally:

   python -m unittest tests/test_bayes_factor.py
