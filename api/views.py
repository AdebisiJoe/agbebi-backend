import io
import os
import tempfile
from PyPDF2 import PdfReader
from rest_framework.permissions import AllowAny, IsAuthenticated
from django.contrib.auth import authenticate, get_user_model
from rest_framework import viewsets
from rest_framework.decorators import action
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.docstore.document import Document as LangchainDocument
from django.conf import settings
from django.core.files.storage import default_storage
from datetime import date

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.contrib.auth import authenticate
from rest_framework_simplejwt.tokens import RefreshToken

from rest_framework.parsers import MultiPartParser, FormParser

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from django.conf import settings


# from .tasks import process_document
from .models import UserProfile, Illness, WeightTracker, DailyAdvice, Document
from .serializers import UserSerializer, UserProfileSerializer, IllnessSerializer, WeightTrackerSerializer, DailyAdviceSerializer, DocumentSerializer

import logging

logger = logging.getLogger(__name__)

User = get_user_model()
class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    # permission_classes = [IsAuthenticated]

    # def get_permissions(self):
    #     if self.action in ['create', 'register']:
    #         return [AllowAny()]
    #     return super().get_permissions()

    @action(detail=False, methods=['post'])
    def register(self, request):
        serializer = self.get_serializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            user.set_password(request.data['password'])
            user.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=False, methods=['post'])
    def login(self, request):
        username = request.data.get('username')
        password = request.data.get('password')
        
        if not username or not password:
            return Response({'error': 'Please provide both username and password'},
                            status=status.HTTP_400_BAD_REQUEST)
        
        user = authenticate(request, username=username, password=password)
    
        if user is None:
            return Response({'error': 'Invalid credentials'},
                            status=status.HTTP_401_UNAUTHORIZED)
        
        if not user.is_active:
            return Response({'error': 'User account is disabled'},
                            status=status.HTTP_403_FORBIDDEN)
        
        refresh = RefreshToken.for_user(user)
        
        return Response({
            'refresh': str(refresh),
            'access': str(refresh.access_token),
            'user_id': user.id,
            'username': user.username,
            'first_name': user.first_name,
            'last_name': user.last_name,
        })

class UserProfileViewSet(viewsets.ModelViewSet):
    queryset = UserProfile.objects.all()
    serializer_class = UserProfileSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return UserProfile.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        # Set the user to the currently authenticated user
        serializer.save(user=self.request.user)

    @action(detail=True, methods=['post'])
    def add_illness(self, request, pk=None):
        profile = self.get_object()
        illness_name = request.data.get('name')
        illness = Illness.objects.create(name=illness_name, user_profile=profile)
        serializer = IllnessSerializer(illness)
        return Response(serializer.data, status=status.HTTP_201_CREATED)

class WeightTrackerViewSet(viewsets.ModelViewSet):
    queryset = WeightTracker.objects.all()
    serializer_class = WeightTrackerSerializer
    permission_classes = [IsAuthenticated]


    def perform_create(self, serializer):
        try:
            user_profile = UserProfile.objects.get(user=self.request.user)
            serializer.save(user_profile=user_profile)
        except UserProfile.DoesNotExist:
            return Response({'error': 'User profile not found'}, status=status.HTTP_400_BAD_REQUEST)


    def get_queryset(self):
        return WeightTracker.objects.filter(user_profile__user=self.request.user)

class DailyAdviceViewSet(viewsets.ModelViewSet):
    queryset = DailyAdvice.objects.all()
    serializer_class = DailyAdviceSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return DailyAdvice.objects.filter(user=self.request.user)

    def calculate_pregnancy_week(self, due_date):
        today = date.today()
        days_until_due = (due_date - today).days
        weeks_pregnant = 40 - (days_until_due // 7)
        return max(1, min(weeks_pregnant, 40))  # Ensure week is between 1 and 40

    @action(detail=False, methods=['get'])
    def get_daily_advice(self, request):
        user = request.user
        try:
            user_profile = UserProfile.objects.get(user=user)
        except UserProfile.DoesNotExist:
            return Response({"error": "User profile not found"}, status=status.HTTP_404_NOT_FOUND)

        # Calculate the current pregnancy week
        current_week = self.calculate_pregnancy_week(user_profile.due_date)

        # Initialize Neo4jGraph with credentials
        graph = Neo4jGraph(
            url=settings.NEO4J_URI,
            username=settings.NEO4J_USERNAME,
            password=settings.NEO4J_PASSWORD
        )

        llm = ChatOpenAI(temperature=0.7)

        # Include more user-specific information in the prompt
        food_prompt = ChatPromptTemplate.from_template(
            "Based on the pregnancy knowledge graph, provide food advice for a pregnant woman in week {week} of pregnancy. "
            "The baby's gender is {gender}, the mother's age is {age}, and her current weight is {weight} kg. "
            "Include foods to eat and foods to avoid."
        )
        food_chain = LLMChain(llm=llm, prompt=food_prompt)
        food_advice = food_chain.run(
            week=current_week,
            gender=user_profile.baby_gender,
            age=user_profile.mother_age,
            weight=user_profile.mother_weight,
            graph=graph
        )

        exercise_prompt = ChatPromptTemplate.from_template(
            "Based on the pregnancy knowledge graph, what exercises are safe and beneficial for a pregnant woman "
            "in week {week} of pregnancy? Consider that the mother's age is {age} and her current weight is {weight} kg."
        )
        exercise_chain = LLMChain(llm=llm, prompt=exercise_prompt)
        exercise_advice = exercise_chain.run(
            week=current_week,
            age=user_profile.mother_age,
            weight=user_profile.mother_weight,
            graph=graph
        )

        advice, created = DailyAdvice.objects.update_or_create(
            user=user,
            week=current_week,
            defaults={
                'food_advice': food_advice,
                'exercise_advice': exercise_advice
            }
        )

        serializer = self.get_serializer(advice)
        return Response(serializer.data)
    

class LoginView(APIView):
    serializer_class = UserSerializer
    def post(self, request):
        username = request.data.get('username')
        password = request.data.get('password')
        user = authenticate(username=username, password=password)
        if user:
            refresh = RefreshToken.for_user(user)
            return Response({
                'refresh': str(refresh),
                'access': str(refresh.access_token),
            })
        return Response({'error': 'Invalid credentials'}, status=status.HTTP_401_UNAUTHORIZED)    
    

class DocumentViewSet(viewsets.ModelViewSet):
    queryset = Document.objects.all()
    serializer_class = DocumentSerializer
    parser_classes = (MultiPartParser, FormParser)
    permission_classes = [IsAuthenticated]

    def perform_create(self, serializer):
        file = self.request.data.get('file')
        if file:
            filename = os.path.splitext(file.name)[0]
            serializer.save(user=self.request.user, title=filename)
        else:
            serializer.save(user=self.request.user)

    def get_queryset(self):
        return Document.objects.filter(user=self.request.user)

    def create(self, request, *args, **kwargs):
        # If title is not provided, use the filename without extension
        if 'title' not in request.data:
            file = request.data.get('file')
            if file:
                filename = os.path.splitext(file.name)[0]
                request.data['title'] = filename

        serializer = self.get_serializer(data=request.data)
        if serializer.is_valid():
            document = serializer.save(user=request.user)
            
            try:
                text = self.process_document(document)
                return Response({
                    'message': 'Document uploaded, processed, and knowledge graph created successfully',
                    'document_id': document.id,
                    'title': document.title
                }, status=status.HTTP_201_CREATED)

                # return Response({
                #     'message': 'Document uploaded, processed, and knowledge graph created successfully',
                #     'document_id': document.id,
                #     'title': text
                # }, status=status.HTTP_201_CREATED)
            except Exception as e:
                document.delete()
                return Response({
                    'error': f'Document processing failed: {str(e)}'
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def get_queryset(self):
        return Document.objects.filter(user=self.request.user)


    def process_document(self, document):
        try:
            # Read the file content
            file_content = document.file.read()

            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name

            # Use PyPDF2 to limit to 10 pages
            pdf_reader = PdfReader(temp_file_path)
            num_pages = min(len(pdf_reader.pages), 10)  # Limit to 10 pages

            # Extract text from the first 10 pages and create Langchain Document objects
            documents = []
            for i in range(num_pages):
                text = pdf_reader.pages[i].extract_text()
                doc = LangchainDocument(page_content=text, metadata={"page": i + 1})
                documents.append(doc)
            #return documents;
            # Split the documents into chunks
            text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
            splits = text_splitter.split_documents(documents)

            # Create graph documents
            llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125", openai_api_key=settings.OPENAI_API_KEY)
            llm_transformer = LLMGraphTransformer(llm=llm)
            graph_documents = llm_transformer.convert_to_graph_documents(splits)

            # Add to Neo4j graph
            graph = Neo4jGraph(
                url=settings.NEO4J_URI,
                username=settings.NEO4J_USERNAME,
                password=settings.NEO4J_PASSWORD
            )
            graph.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=True)

            # Create vector index
            vector_index = Neo4jVector.from_existing_graph(
                OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY),
                url=settings.NEO4J_URI,
                username=settings.NEO4J_USERNAME,
                password=settings.NEO4J_PASSWORD,
                search_type="hybrid",
                node_label="Document",
                text_node_properties=["text"],
                embedding_node_property="embedding"
            )

            # Update the document status
            document.processed = True
            document.save()

        except Exception as e:
            raise Exception(f"Document processing failed: {str(e)}")

        finally:
            # Clean up: remove the temporary file
            if 'temp_file_path' in locals():
                os.unlink(temp_file_path)