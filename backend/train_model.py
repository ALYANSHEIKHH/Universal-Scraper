from app.models.training import train_model
import logging

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_model()
