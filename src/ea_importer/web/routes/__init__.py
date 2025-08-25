"""
Web interface routes for EA Importer.

This module exports all route handlers for the web interface:
- Clustering review and approval workflows
- Family management and gold text editing
- Instance and overlay management
- QA testing and results review
- System monitoring and administration
"""

from .clustering import router as clustering_router
from .family import router as family_router

# Import placeholder routers (to be implemented)
from fastapi import APIRouter

# Instance management router (placeholder)
instance_router = APIRouter()

@instance_router.get("/")
async def instances_overview():
    return {"message": "Instance management - Coming Soon"}

# QA testing router (placeholder) 
qa_router = APIRouter()

@qa_router.get("/")
async def qa_overview():
    return {"message": "QA testing dashboard - Coming Soon"}

# System monitoring router (placeholder)
monitoring_router = APIRouter()

@monitoring_router.get("/")
async def monitoring_overview():
    return {"message": "System monitoring - Coming Soon"}

__all__ = [
    'clustering_router',
    'family_router', 
    'instance_router',
    'qa_router',
    'monitoring_router'
]