from rest_framework import serializers
from .models import UserProfile, Illness, WeightTracker, DailyAdvice, Document
#from django.contrib.auth.models import User
from django.contrib.auth import get_user_model
from django.contrib.auth.password_validation import validate_password


User = get_user_model()
class UserSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, required=True, validators=[validate_password])

    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'password', 'first_name', 'last_name']
        extra_kwargs = {'password': {'write_only': True}}

    def create(self, validated_data):
        # Create the user with a hashed password
        user = User.objects.create_user(
            username=validated_data['username'],
            first_name=validated_data['first_name'],
            last_name=validated_data['last_name'],
            email=validated_data.get('email', ''),
            password=validated_data['password']
        )
        return user

    def update(self, instance, validated_data):
        if 'password' in validated_data:
            instance.set_password(validated_data['password'])
        return super().update(instance, validated_data)

class IllnessSerializer(serializers.ModelSerializer):
    class Meta:
        model = Illness
        fields = ['id', 'name']

class UserProfileSerializer(serializers.ModelSerializer):
    illnesses = IllnessSerializer(many=True, read_only=True)

    class Meta:
        model = UserProfile
        fields = ['id', 'baby_gender', 'due_date', 'estimated_weight', 'mother_weight', 'mother_age', 'illnesses']

class WeightTrackerSerializer(serializers.ModelSerializer):
    class Meta:
        model = WeightTracker
        fields = ['id', 'date', 'weight']

class DailyAdviceSerializer(serializers.ModelSerializer):
    class Meta:
        model = DailyAdvice
        fields = ['id', 'week', 'food_advice', 'exercise_advice']

      
class DocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Document
        fields = ['id', 'title', 'file', 'uploaded_at', 'processed']
        read_only_fields = ['user', 'uploaded_at', 'processed']