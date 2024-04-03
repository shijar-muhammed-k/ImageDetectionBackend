import datetime
from rest_framework.response import Response
from django.shortcuts import render
from rest_framework import generics, filters
from .utils import predict, SendMail
from .serializer import PredictionImageSerializer, PredictionImage
from User.models import Profile
from rest_framework.pagination import PageNumberPagination

# Create your views here.
class PredictionListView(generics.ListCreateAPIView):
    
    search_fields = ['user__first_name', 'user__last_name',]
    filter_backends = (filters.SearchFilter,)
    
    pagination_class = PageNumberPagination
    pagination_class.page_size = 10
    
    serializer_class = PredictionImageSerializer
    
    def get_queryset(self):
        user = Profile.objects.get(id = self.request.user.id)
        if(user.role == 1):
            queryset = PredictionImage.objects.filter(reported = True).order_by('-id')
        else :    
            queryset = PredictionImage.objects.filter(user = self.request.user.id).order_by('-id')
        return queryset
    
    def post(self, request, *args, **kwargs):
        data = request.data
        db = PredictionImage()
        db.user = Profile.objects.get(id=self.request.user.id)
        db.image = data['image']
        db.save()
        result = predict('media/'+str(db.image))
        db.prediction = result['prediction']
        db.confidence = result['confidence']
        db.save()
        instance = db
        serializer = self.get_serializer(instance)
        return Response(serializer.data)
    
    
    def patch(self, request, *args, **kwargs):
        data = request.data
        db = PredictionImage.objects.get(id = data['id'])
        db.reported = True
        db.reported_date = datetime.datetime.now()
        db.save()
        SendMail(db)
        return Response({'reported': 'success'})