from fastapi import APIRouter, HTTPException, status
from app.models.schemas import InterviewRequest, InterviewResponse
from app.services.session_manager import (
    session_manager,
    SessionAlreadyExistsError,
    SessionNotFoundError,
)
from app.services.interview_engine import interview_engine
from app.utils.logging import logger

from app.data.loader import data_loader

router = APIRouter(prefix="/api", tags=["Interview"])


@router.get(
    "/candidates",
    summary="Get all available candidate profiles",
    description="Returns list of cohort candidates for selection in the frontend UI.",
)
async def get_candidates():
    return {"candidates": data_loader.get_all_candidates()}



@router.post(
    "/interview",
    response_model=InterviewResponse,
    response_model_exclude_none=True,
    status_code=status.HTTP_200_OK,
    summary="Conduct technical interview (start or continuation)",
    description="Single unified endpoint for starting and carrying out conversational technical interviews.",
)
async def interview_endpoint(req: InterviewRequest) -> InterviewResponse:
    session_id = req.sessionId.strip()
    if not session_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Field 'sessionId' cannot be empty.",
        )

    # 1. Start Request Check: candidate payload provided
    if req.candidate is not None:
        try:
            response = interview_engine.start_interview(
                session_id=session_id,
                candidate_data=req.candidate,
            )
            return response
        except SessionAlreadyExistsError as e:
            logger.warning(f"Conflict starting session {session_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(e),
            )
        except ValueError as e:
            logger.warning(f"Validation error starting session {session_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            )
        except Exception as e:
            logger.error(f"Unexpected error starting session {session_id}: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An internal error occurred while initializing the interview.",
            )

    # 2. Continuation Request Check: session already exists
    if session_manager.session_exists(session_id):
        try:
            response = interview_engine.handle_turn(
                session_id=session_id,
                message=req.message,
            )
            return response
        except Exception as e:
            logger.error(f"Unexpected error during turn in session {session_id}: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An internal error occurred while processing the interview turn.",
            )

    # 3. Neither start nor existing session
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Session '{session_id}' does not exist. To start a new interview, provide the 'candidate' object.",
    )
