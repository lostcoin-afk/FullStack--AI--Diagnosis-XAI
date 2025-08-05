from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["diagnosis_db"]

db.users.delete_many({})
db.visualisations.delete_many({})

print("All documents deleted from users and visualisations collections.")
