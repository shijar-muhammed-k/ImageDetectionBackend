from rest_framework import generics, filters
from rest_framework_simplejwt.views import TokenObtainPairView
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .models import Profile, Feedback
from .serializer import ProfileSerializer, UserTokenObtainSerializer, FeedbackSerializer
from .utils import SendMail

# Create your views here.


class ProfileListView(generics.ListCreateAPIView):
    search_fields = ['first_name', 'last_name',]
    filter_backends = (filters.SearchFilter,)

    queryset = Profile.objects.all()
    serializer_class = ProfileSerializer


class ProfileDetailView(generics.RetrieveUpdateDestroyAPIView):

    queryset = Profile.objects.all()
    lookup_field = 'id'
    serializer_class = ProfileSerializer
    
    
class LoginView(TokenObtainPairView):
    serializer_class = UserTokenObtainSerializer
    

class MessageListView(generics.ListCreateAPIView):
    search_fields = ['user__first_name', 'user__last_name',]
    filter_backends = (filters.SearchFilter,)

    queryset = Feedback.objects.all()
    serializer_class = FeedbackSerializer
    
    def post(self, request, *args, **kwargs):
        
        data = self.request.data
        db = Feedback()
        db.user = Profile.objects.get(id=self.request.user.id)
        db.message = data['message']
        db.save()
        return Response({'status': 'Success'})
    
    def patch(self, request, *args, **kwargs):
        instance = Feedback.objects.get(id=request.data['id'])
        instance.replied = True
        instance.save()
        data = request.data
        data['date'] = instance.date
        data['mail'] = instance.user.email
        SendMail(data)
        
        return Response({'status': 'message'})