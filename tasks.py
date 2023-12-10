from celery import Celery
from celery.schedules import crontab

app = Celery('tasks', broker='redis://localhost:6379/0')

@app.task
def train_daily_task():

    print('Hello')

# Schedule the task to run daily at midnight
app.conf.beat_schedule = {
    'daily-training': {
        'task': 'tasks.daily_training_task',
        'schedule': crontab(minute=0, hour=0),
    },
}