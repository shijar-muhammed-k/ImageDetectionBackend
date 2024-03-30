from rest_framework import serializers

from .models import PredictionImage


class PredictionImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = PredictionImage
        fields = '__all__'
        depth = 1