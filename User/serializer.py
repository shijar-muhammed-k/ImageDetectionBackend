from rest_framework import serializers

from django.contrib.auth.hashers import make_password
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from django.shortcuts import get_object_or_404

from .models import Profile, Feedback


class ProfileSerializer(serializers.ModelSerializer):

    password = serializers.CharField(min_length=8, max_length=52, write_only=True)

    class Meta:
        model = Profile
        fields = ['id', 'username', 'first_name', 'last_name', 'password', 'email', 
                  'date_joined', 'phone', 'profile_image', 'profession', 'dob', 'gender', 'role']
        
    
    def create(self, validated_data):
        password = make_password(validated_data.pop("password"))
        validated_data['password'] = password
        return super().create(validated_data)
    
    
class UserTokenObtainSerializer(TokenObtainPairSerializer):

    def validate(self, attrs):
        user = get_object_or_404(Profile, email = attrs.pop('username'))
        print(user)
        attrs['username'] = user.username
        data = super().validate(attrs)
        profile = Profile.objects.filter(username=self.user).first()
        
        return {
            "access": data['access'],
            "role": profile.role,
            "id": profile.id
        }
        
        

class FeedbackSerializer(serializers.ModelSerializer):
    
    class Meta:
        model = Feedback
        fields = '__all__'
        depth = 1