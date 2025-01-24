import random
from locust import HttpUser, between, task

class MyUser(HttpUser):
    """A simple Locust user class that defines the tasks to be performed by the users."""

    wait_time = between(1, 2)

    @task
    def get_root(self) -> None:
        """A task that simulates a user visiting the root URL of the FastAPI app."""
        self.client.get("/")

    @task(3)
    def get_item(self) -> None:
        """A task that simulates a user visiting a random item URL of the FastAPI app."""
        item_id = random.randint(1, 10)
        self.client.get(f"/items/{item_id}")

# Run in terminal: locust -f tests/locustfile.py
# Open webpage: http://localhost:8089
# Fill in the numbers and press start 

"""
With settings: 
- Number of users (peak concurrency):     1 
- Ramp up (users started/sec):            1
- Host:                                   http://127.0.0.1:8000
- Advanced option -> Run time:            1m

What is the average response time of your API?
- 141.6 ms (found under "Average (ms)" -> "Aggregated")

What is the 99th percentile response time of your API?
- 5200 ms (found under "99%ile (ms)" -> "Aggregated")

How many requests per second can your API handle?
- 6.8 (found under "Current RPS" -> "Aggregated")

"""

# Alternatively, run in terminal: locust -f tests/performancetests/locustfile.py \
    # --headless --users 10 --spawn-rate 1 --run-time 1m --host $MYENDPOINT
