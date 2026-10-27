"""
Все модели имеют единый интерфейс:
    model.fit(train_matrix, **extra)          — обучение
    model.score(user_ids)                     — скоры для всех объектов, shape (n_users, n_items)
    model.recommend(user_ids, k, exclude)     — top-k с исключением уже известных объектов
"""
