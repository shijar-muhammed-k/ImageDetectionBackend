from django.urls import path
from .views import ProfileListView, ProfileDetailView, LoginView, MessageListView

urlpatterns = [
    path('profile/', ProfileListView.as_view(), name='Profile'),
    path('profile/<id>', ProfileDetailView.as_view(), name='Profile'),
    path('login/', LoginView.as_view(), name='obtain-token'),
    path('message/', MessageListView.as_view(), name='obtain-token'),
    
]
