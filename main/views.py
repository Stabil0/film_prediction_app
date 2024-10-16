# views.py
import joblib
from django.shortcuts import render
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from django.http import HttpResponse

# Загрузка модели
svr_model = joblib.load('main/models/srv_model.pkl')
vectorizer = joblib.load('main/models/vectorizer.pkl')
classifier_model = joblib.load('main/models/classifier_model.pkl')


def predict_review(request):
    if request.method == 'POST':
        user_review = request.POST['review']

        # Преобразуем отзыв пользователя в векторы
        user_vector = vectorizer.transform([user_review])

        # Предсказание оценки
        predicted_rating = svr_model.predict(user_vector)[0]

        # Предсказание категории (позитивный/негативный)
        predicted_label = classifier_model.predict(user_vector)[0]
        review_category = 'Positive' if predicted_label == 1 else 'Negative'

        # Передача предсказаний в шаблон
        return render(request, 'predictions/result.html', {
            'predicted_rating': round(predicted_rating,2),
            'review_category': review_category,
            'user_review': user_review
        })
    else:
        return render(request, 'predictions/form.html')
