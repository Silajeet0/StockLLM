from django.urls import path
from . import views

urlpatterns = [
    path("",                               views.index,        name="index"),
    path("api/run/",                       views.api_run,      name="api_run"),
    path("api/progress/<str:task_id>/",    views.api_progress, name="api_progress"),
    path("api/results/<str:task_id>/",     views.api_results,  name="api_results"),
    path("api/ohlc/",                      views.api_ohlc,     name="api_ohlc"),
]
