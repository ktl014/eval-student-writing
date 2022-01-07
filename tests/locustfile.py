"""
Stress Test AWS Lambda server through API Gateway
"""
import json

from locust import HttpUser, between, task

# Loading the test JSON data
with open('test_sample.json') as f:
    test_data = json.loads(f.read())


# Creating an API User class inheriting from Locust's HttpUser class
class APIUser(HttpUser):

    # Setting the host name and wait_time
    host = 'https://yvvkq85w94.execute-api.us-west-1.amazonaws.com'
    wait_time = between(1, 5)

    # Defining the post task using the JSON test data
    @task()
    def predict_endpoint(self):
        self.client.post('/Deploy/predict', json=test_data)
