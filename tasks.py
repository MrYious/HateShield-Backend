from celery import Celery
# from your_training_module import train_model_from_csv

app = Celery('tasks', broker='redis://localhost:6379/0')

@app.task
def train_daily_task():
    # train_model_from_csv('newdata.csv')
    return False