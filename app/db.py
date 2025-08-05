from app import mongo

def get_users_collection():
    return mongo.db.users

def get_visualisations_collection():
    return mongo.db.visualisations
