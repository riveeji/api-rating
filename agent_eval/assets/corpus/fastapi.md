# FastAPI Snapshot
Source: https://fastapi.tiangolo.com/
Reference-Date: 2026-03-31

## response-model
FastAPI can use `response_model` on a path operation to validate, serialize, and filter the data returned by the handler. This is useful when the handler returns a richer internal object but the API contract should expose only the public fields. The filtering happens before the final response is sent, so internal attributes can be excluded from the payload.

## dependencies
FastAPI uses `Depends()` to declare dependencies that can be reused across multiple path operations. Dependencies can fetch auth context, database handles, or request-scoped settings, and they can be nested. A dependency can also receive a `Response` object and set headers or cookies that will be included in the final response.

## background-tasks
`BackgroundTasks` lets a handler schedule work that should run after the response has been returned. The common examples are audit logging, email notifications, and other non-critical actions that should not block the request. It is appropriate for short follow-up work, not for a full job queue or long-running batch pipeline.

## api-router
`APIRouter` helps split a FastAPI application into smaller route modules. Routers can share prefixes, tags, dependencies, and response metadata, which keeps larger services easier to organize. The main app can then include each router rather than defining every endpoint in one file.

## templates-and-static
FastAPI can serve server-rendered HTML with `Jinja2Templates` and static assets with `StaticFiles`. This is a lightweight way to build dashboards, internal tools, and report pages without introducing a separate frontend framework. Templates are rendered from route handlers, and static assets are mounted under a chosen path.

## test-client
FastAPI applications can be tested with `TestClient`, which provides a convenient way to issue requests against the ASGI app in unit tests. This makes it straightforward to verify routes, status codes, JSON responses, and HTML pages without starting a live server.
