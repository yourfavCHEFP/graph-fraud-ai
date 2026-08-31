"""
FIX (mentor review item 8): the predictor used to live in a module-level
global (`_predictor`), set by a separate initialize_predictor() function
called from app.py's startup event. That made the service hard to test
(every test shares the same global, can't inject a fake predictor) and
hid the real source of truth (app.state, set by app.py's lifespan --
see mentor review item 7).

This is now a FastAPI dependency function that reads from
`request.app.state.predictor` -- the actual state FastAPI's own lifespan
manages -- and raises a controlled 503 if it's not ready, instead of a
bare RuntimeError. Tests can override this with
`app.dependency_overrides[get_predictor] = lambda: fake_predictor`.
"""

from fastapi import HTTPException, Request


def get_predictor(request: Request):
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not ready. See /ready for status.",
        )
    return predictor
