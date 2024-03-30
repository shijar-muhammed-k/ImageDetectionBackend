from django.urls import path

from rest_framework.authtoken import views

from .views import PredictionListView

urlpatterns = [
    path('predict/', PredictionListView.as_view(), name='Predict'),
]
