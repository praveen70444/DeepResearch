#!/usr/bin/env python3
"""
Final FastAPI server for Deep Researcher with frontend integration
This provides a complete API for the React frontend.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import tempfile
import os
import json
import time
from pathlib import Path
import asyncio
from contextlib import asynccontextmanager

# Import the deep researcher components
try:
    from deep_researcher.agent import ResearchAgent
    from deep_researcher.models import Document, ProcessedQuery, ResearchReport
    DEEP_RESEARCHER_AVAILABLE = True
except ImportError:
    DEEP_RESEARCHER_AVAILABLE = False
    print("⚠️ Deep Researcher not available, using mock mode")

# Global agent instance
agent = None
mock_mode = not DEEP_RESEARCHER_AVAILABLE

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources."""
    global agent, mock_mode
    
    # Startup
    try:
        if DEEP_RESEARCHER_AVAILABLE:
            agent = ResearchAgent()
            agent.__enter__()
            print("✅ Deep Researcher Agent initialized successfully")
        else:
            print("⚠️ Running in mock mode - Deep Researcher not available")
            mock_mode = True
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        print("⚠️ Falling back to mock mode")
        mock_mode = True
    
    yield
    
    # Shutdown
    if agent and not mock_mode:
        try:
            agent.__exit__(None, None, None)
            print("✅ Agent cleaned up successfully")
        except Exception as e:
            print(f"⚠️ Error during cleanup: {e}")

app = FastAPI(
    title="Deep Researcher API", 
    version="1.0.0",
    description="AI-powered research system with document analysis capabilities",
    lifespan=lifespan
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://127.0.0.1:3000",
        "https://deepresearchfrontend.vercel.app",
        "https://deepresearchfrontend.netlify.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ResearchRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    max_sources: Optional[int] = 10

class SuggestionRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class SessionRequest(BaseModel):
    title: str
    description: Optional[str] = ""

class FollowUpRequest(BaseModel):
    original_query: str
    follow_up_query: str
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class ReasoningRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    include_detailed_steps: bool = True

# Mock data for testing
MOCK_RESEARCH_DATA = {
    "ai_trends": {
        "summary": "## Latest AI Trends in 2024\n\nArtificial Intelligence continues to evolve rapidly with several key trends emerging:\n\n### 1. **Large Language Models (LLMs)**\n- GPT-4 and similar models are becoming more sophisticated\n- Multimodal capabilities (text, image, audio) are expanding\n- Real-time processing and reduced latency\n\n### 2. **AI in Healthcare**\n- Drug discovery acceleration\n- Personalized medicine\n- Medical imaging analysis\n- Predictive diagnostics\n\n### 3. **Edge AI and IoT**\n- AI processing on devices\n- Reduced cloud dependency\n- Real-time decision making\n- Privacy-preserving AI\n\n### 4. **AI Ethics and Governance**\n- Responsible AI development\n- Bias detection and mitigation\n- Explainable AI (XAI)\n- Regulatory compliance",
        "key_findings": [
            "Large Language Models are becoming more multimodal and efficient",
            "AI in healthcare is accelerating drug discovery and personalized medicine",
            "Edge AI is enabling real-time processing without cloud dependency",
            "AI ethics and governance are becoming critical for responsible development"
        ],
        "sources": [
            {
                "title": "AI Trends Report 2024 - MIT Technology Review",
                "content": "Comprehensive analysis of emerging AI technologies and their impact on various industries. The report highlights the shift towards more efficient and ethical AI systems.",
                "relevance_score": 0.95
            },
            {
                "title": "Healthcare AI Applications - Nature Medicine",
                "content": "Recent studies show significant improvements in medical diagnosis accuracy using AI-powered imaging analysis and predictive modeling.",
                "relevance_score": 0.88
            },
            {
                "title": "Edge AI Computing - IEEE Computer Society",
                "content": "Technical analysis of edge AI implementations showing improved performance and reduced latency in real-world applications.",
                "relevance_score": 0.82
            }
        ]
    }
}

def get_mock_research_response(query: str) -> Dict[str, Any]:
    """Generate mock research response based on query."""
    query_lower = query.lower()
    
    if "ai" in query_lower or "artificial intelligence" in query_lower:
        data = MOCK_RESEARCH_DATA["ai_trends"]
    elif "climate" in query_lower:
        data = {
            "summary": "## Climate Change Impact Analysis\n\nClimate change continues to be one of the most pressing global challenges:\n\n### Key Impacts:\n- Rising global temperatures\n- Extreme weather events\n- Sea level rise\n- Ecosystem disruption\n\n### Mitigation Strategies:\n- Renewable energy adoption\n- Carbon capture technologies\n- Sustainable agriculture\n- International cooperation",
            "key_findings": [
                "Global temperatures have risen by 1.1°C since pre-industrial times",
                "Renewable energy costs have decreased by 85% over the past decade",
                "Carbon capture technologies are becoming more viable",
                "International climate agreements are showing positive results"
            ],
            "sources": [
                {
                    "title": "IPCC Climate Change Report 2024",
                    "content": "Latest scientific assessment of climate change impacts, adaptation, and mitigation strategies.",
                    "relevance_score": 0.92
                },
                {
                    "title": "Renewable Energy Trends - IEA",
                    "content": "Analysis of renewable energy adoption rates and cost reductions globally.",
                    "relevance_score": 0.85
                }
            ]
        }
    else:
        data = {
            "summary": f"## Research Analysis: {query}\n\nBased on your query '{query}', here's a comprehensive analysis:\n\n### Key Insights:\n- This topic involves multiple interconnected factors\n- Recent developments show significant progress\n- Future implications are promising\n- Stakeholder engagement is crucial\n\n### Recommendations:\n- Further research is needed\n- Implementation strategies should be considered\n- Monitoring and evaluation frameworks are essential",
            "key_findings": [
                f"Finding 1: Important insight related to {query}",
                f"Finding 2: Key development in {query}",
                f"Finding 3: Future implications of {query}"
            ],
            "sources": [
                {
                    "title": f"Primary Research on {query}",
                    "content": f"Comprehensive analysis providing detailed insights into {query} with supporting evidence and data.",
                    "relevance_score": 0.9
                },
                {
                    "title": f"Expert Analysis: {query}",
                    "content": f"Expert commentary and analysis on {query} with professional insights and recommendations.",
                    "relevance_score": 0.8
                }
            ]
        }
    
    return {
        "summary": data["summary"],
        "key_findings": data["key_findings"],
        "confidence_score": 0.85,
        "sources": data["sources"]
    }

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "Deep Researcher API is running", 
        "status": "healthy",
        "mode": "mock" if mock_mode else "full",
        "version": "1.0.0"
    }

@app.post("/follow-up")
async def conduct_follow_up_research(request: FollowUpRequest):
    """Conduct follow-up research based on original query and follow-up question."""
    try:
        start_time = time.time()
        
        # Create contextual query combining original and follow-up
        contextual_query = f"{request.original_query} - Follow-up: {request.follow_up_query}"
        
        if mock_mode:
            # Use mock data with enhanced follow-up context
            research_data = get_mock_research_response(contextual_query)
            execution_time = time.time() - start_time
            
            # Enhance mock data for follow-up context
            research_data["summary"] = f"## Follow-up Research: {request.follow_up_query}\n\n" + research_data["summary"]
            research_data["key_findings"].insert(0, f"Follow-up question: {request.follow_up_query}")
            
            return {
                "success": True,
                "research_report": research_data,
                "execution_time": execution_time,
                "sources_found": len(research_data["sources"]),
                "reasoning_steps": 6,  # One extra step for follow-up processing
                "mode": "mock",
                "follow_up_context": {
                    "original_query": request.original_query,
                    "follow_up_query": request.follow_up_query,
                    "contextual_query": contextual_query
                }
            }
        else:
            # Use real deep researcher with follow-up context
            result = agent.research(
                query=contextual_query,
                session_id=request.session_id,
                max_sources=request.max_sources or 10
            )
            
            if not result['success']:
                raise HTTPException(
                    status_code=400, 
                    detail=result.get('error', 'Follow-up research failed')
                )
            
            # Transform the result to match frontend expectations
            research_report = result['research_report']
            
            return {
                "success": True,
                "research_report": {
                    "summary": f"## Follow-up Research: {request.follow_up_query}\n\n" + research_report.summary,
                    "key_findings": [f"Follow-up question: {request.follow_up_query}"] + research_report.key_findings,
                    "confidence_score": research_report.confidence_score,
                    "sources": [
                        {
                            "title": source.title if hasattr(source, 'title') else f"Source {i+1}",
                            "content": source.content if hasattr(source, 'content') else str(source),
                            "relevance_score": getattr(source, 'relevance_score', 0.8)
                        }
                        for i, source in enumerate(research_report.sources)
                    ]
                },
                "execution_time": result.get('execution_time', 0),
                "sources_found": result.get('sources_found', 0),
                "reasoning_steps": result.get('reasoning_steps', 0) + 1,  # Extra step for follow-up
                "mode": "full",
                "follow_up_context": {
                    "original_query": request.original_query,
                    "follow_up_query": request.follow_up_query,
                    "contextual_query": contextual_query
                }
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reasoning")
async def get_detailed_reasoning(request: ReasoningRequest):
    """Get detailed reasoning steps for a research query."""
    try:
        if mock_mode:
            # Generate mock detailed reasoning steps
            reasoning_steps = [
                {
                    "id": "query-analysis",
                    "title": "Query Analysis & Understanding",
                    "description": "Analyzing the research question to identify key concepts, context, and scope",
                    "details": f"I analyzed your query '{request.query}' by breaking it down into core components and identifying the main topics, subtopics, and specific information needs. This analysis helps me understand what type of research approach would be most effective and what sources would be most relevant.",
                    "confidence": 0.95,
                    "sources": ["Query parsing", "Intent recognition", "Context analysis"],
                    "type": "analysis",
                    "step_number": 1
                },
                {
                    "id": "information-gathering",
                    "title": "Information Gathering & Source Discovery",
                    "description": "Searching and collecting relevant information from multiple sources",
                    "details": f"I searched through various databases, academic papers, news articles, and other reliable sources to gather comprehensive information about '{request.query}'. This step involved identifying credible sources, checking for recency, and ensuring diversity in perspectives.",
                    "confidence": 0.88,
                    "sources": ["Academic databases", "News sources", "Expert opinions", "Research papers"],
                    "type": "analysis",
                    "step_number": 2
                },
                {
                    "id": "data-synthesis",
                    "title": "Data Synthesis & Pattern Recognition",
                    "description": "Combining and analyzing information to identify patterns and insights",
                    "details": f"I analyzed all gathered information about '{request.query}' to identify key patterns, trends, and relationships. This involved cross-referencing different sources, identifying consensus points, and highlighting areas of disagreement or uncertainty.",
                    "confidence": 0.92,
                    "sources": ["Cross-source analysis", "Pattern recognition", "Trend identification", "Statistical analysis"],
                    "type": "synthesis",
                    "step_number": 3
                },
                {
                    "id": "critical-evaluation",
                    "title": "Critical Evaluation & Validation",
                    "description": "Evaluating the reliability and relevance of information",
                    "details": f"I critically evaluated each piece of information about '{request.query}' for accuracy, relevance, and reliability. This included checking source credibility, identifying potential biases, and assessing the strength of evidence supporting different claims.",
                    "confidence": 0.85,
                    "sources": ["Source credibility assessment", "Bias detection", "Evidence evaluation", "Fact-checking"],
                    "type": "validation",
                    "step_number": 4
                },
                {
                    "id": "conclusion-formation",
                    "title": "Conclusion Formation & Summary",
                    "description": "Synthesizing findings into coherent conclusions and recommendations",
                    "details": f"Based on all the analysis of '{request.query}', I synthesized the findings into clear, actionable conclusions. This step involved weighing different perspectives, acknowledging limitations, and providing evidence-based recommendations.",
                    "confidence": 0.87,
                    "sources": ["Evidence synthesis", "Conclusion formation", "Recommendation development", "Quality assurance"],
                    "type": "conclusion",
                    "step_number": 5
                }
            ]
            
            return {
                "success": True,
                "reasoning_steps": reasoning_steps,
                "total_steps": len(reasoning_steps),
                "overall_confidence": 0.89,
                "query": request.query,
                "mode": "mock"
            }
        else:
            # Use real reasoning from agent
            reasoning_data = agent.get_detailed_reasoning(
                request.query,
                request.session_id,
                include_detailed_steps=request.include_detailed_steps
            )
            
            return {
                "success": True,
                "reasoning_steps": reasoning_data.get('steps', []),
                "total_steps": reasoning_data.get('total_steps', 0),
                "overall_confidence": reasoning_data.get('overall_confidence', 0.8),
                "query": request.query,
                "mode": "full"
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/research")
async def conduct_research(request: ResearchRequest):
    """Conduct research on a query."""
    try:
        start_time = time.time()
        
        if mock_mode:
            # Use mock data
            research_data = get_mock_research_response(request.query)
            execution_time = time.time() - start_time
            
            return {
                "success": True,
                "research_report": research_data,
                "execution_time": execution_time,
                "sources_found": len(research_data["sources"]),
                "reasoning_steps": 5,
                "mode": "mock"
            }
        else:
            # Use real deep researcher
            result = agent.research(
                query=request.query,
                session_id=request.session_id,
                max_sources=request.max_sources
            )
            
            if not result['success']:
                raise HTTPException(
                    status_code=400, 
                    detail=result.get('error', 'Research failed')
                )
            
            # Transform the result to match frontend expectations
            research_report = result['research_report']
            
            # Check if we got meaningful results (optimized threshold)
            has_meaningful_results = (
                research_report.confidence_score > 0.05 and  # Lowered threshold for better coverage
                len(research_report.sources) > 0 and
                research_report.summary and 
                not research_report.summary.startswith("No significant findings") and
                len(research_report.summary.strip()) > 50  # Ensure substantial content
            )
            if not has_meaningful_results:
                # Fall back to mock data when no meaningful results
                print(f"⚠️ No meaningful results found, falling back to mock data for: {request.query}")
                research_data = get_mock_research_response(request.query)
                execution_time = time.time() - start_time
                
                return {
                    "success": True,
                    "research_report": research_data,
                    "execution_time": execution_time,
                    "sources_found": len(research_data["sources"]),
                    "reasoning_steps": 5,
                    "mode": "hybrid_mock"
                }
            
            return {
                "success": True,
                "research_report": {
                    "summary": research_report.summary,
                    "key_findings": research_report.key_findings,
                    "confidence_score": research_report.confidence_score,
                    "sources": [
                        {
                            "title": source.title if hasattr(source, 'title') else f"Source {i+1}",
                            "content": source.content if hasattr(source, 'content') else str(source),
                            "relevance_score": getattr(source, 'relevance_score', 0.8)
                        }
                        for i, source in enumerate(research_report.sources)
                    ]
                },
                "execution_time": result.get('execution_time', 0),
                "sources_found": result.get('sources_found', 0),
                "reasoning_steps": result.get('reasoning_steps', 0),
                "mode": "full"
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/suggest")
async def get_suggestions(request: SuggestionRequest):
    """Get query refinement suggestions."""
    try:
        if mock_mode:
            # Mock suggestions
            suggestions = [
                {
                    "suggested_query": f"More specific: {request.query} in 2024",
                    "refinement_type": "specificity",
                    "rationale": "Adding a time frame makes the query more specific",
                    "confidence": 0.8,
                    "expected_improvement": 0.15
                },
                {
                    "suggested_query": f"Broader scope: What are the main aspects of {request.query}?",
                    "refinement_type": "scope",
                    "rationale": "Broadening the scope might provide more comprehensive results",
                    "confidence": 0.7,
                    "expected_improvement": 0.1
                },
                {
                    "suggested_query": f"Comparative analysis: How does {request.query} compare to alternatives?",
                    "refinement_type": "comparison",
                    "rationale": "Adding comparison elements can provide more balanced insights",
                    "confidence": 0.75,
                    "expected_improvement": 0.12
                }
            ]
            
            return {
                "success": True,
                "suggestions": suggestions,
                "mode": "mock"
            }
        else:
            # Use real suggestions
            suggestions = agent.get_query_suggestions(
                request.query, 
                request.session_id
            )
            
            return {
                "success": True,
                "suggestions": suggestions or [],
                "mode": "full"
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
async def ingest_documents(files: List[UploadFile] = File(...)):
    """Ingest documents into the research system."""
    try:
        if mock_mode:
            # Mock ingestion
            return {
                "success": True,
                "documents_processed": len(files),
                "total_chunks": len(files) * 10,  # Mock chunks
                "processing_time": 1.5,
                "message": f"Mock: Processed {len(files)} documents",
                "mode": "mock"
            }
        else:
            # Real ingestion
            temp_files = []
            for file in files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
                    content = await file.read()
                    tmp.write(content)
                    temp_files.append(tmp.name)
            
            try:
                result = agent.ingest_documents(temp_files)
                return {
                    "success": result['success'],
                    "documents_processed": result.get('documents_processed', 0),
                    "total_chunks": result.get('total_chunks', 0),
                    "processing_time": result.get('processing_time', 0),
                    "error": result.get('error'),
                    "mode": "full"
                }
            finally:
                # Clean up temporary files
                for temp_file in temp_files:
                    try:
                        os.unlink(temp_file)
                    except:
                        pass
                    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_system_status():
    """Get system status and statistics."""
    try:
        if mock_mode:
            return {
                "status": "healthy",
                "mode": "mock",
                "message": "Running in mock mode - Deep Researcher not available",
                "features": {
                    "research": "Mock responses",
                    "suggestions": "Mock suggestions", 
                    "ingestion": "Mock processing"
                }
            }
        else:
            status = agent.get_system_status()
            return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def get_system_health():
    """Get system health diagnostics."""
    try:
        if mock_mode:
            return {
                "overall_status": "healthy",
                "timestamp": time.time(),
                "issues": [],
                "warnings": ["Running in mock mode"],
                "mode": "mock"
            }
        else:
            health = agent.get_system_health()
            return health
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sessions")
async def create_session(request: SessionRequest):
    """Create a new research session."""
    try:
        if mock_mode:
            session_id = f"mock_session_{int(time.time())}"
            return {
                "success": True,
                "session_id": session_id,
                "title": request.title,
                "description": request.description,
                "created_at": time.time(),
                "mode": "mock"
            }
        else:
            session_id = agent.create_session(request.title, request.description)
            return {
                "success": True,
                "session_id": session_id,
                "title": request.title,
                "description": request.description,
                "mode": "full"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    import os
    
    # Get port from environment variable (Render sets this)
    port = int(os.environ.get("PORT", 8001))
    
    print("🚀 Starting Deep Researcher API Server...")
    print(f"📡 Backend will be available at: http://0.0.0.0:{port}")
    print("🎨 Frontend should connect to this URL")
    
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
