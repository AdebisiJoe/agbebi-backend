from django.db import models
from django.contrib.auth.models import User

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    baby_gender = models.CharField(max_length=10, choices=[('male', 'Male'), ('female', 'Female'), ('unknown', 'Unknown')])
    due_date = models.DateField()
    estimated_weight = models.FloatField(null=True, blank=True)
    mother_weight = models.FloatField(null=True, blank=True)
    mother_age = models.IntegerField()

class Illness(models.Model):
    name = models.CharField(max_length=200)
    user_profile = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name='illnesses')

class WeightTracker(models.Model):
    user_profile = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    date = models.DateField()
    weight = models.FloatField()

class DailyAdvice(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    week = models.IntegerField()
    food_advice = models.TextField(null=True, blank=True)
    exercise_advice = models.TextField(null=True, blank=True)

class Document(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='documents')
    title = models.CharField(max_length=255)
    file = models.FileField(upload_to='documents/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed = models.BooleanField(default=False) 

