# connectaac-server

Minimal flashcard generation API.

## Endpoints
GET /health
GET /metrics
POST /generate_flashcards
Body: { "context": "<string>", "count": 5, "tag": "optional" }
Auth: Authorization: Bearer <JWT>

## Models
Primary: gpt-5-nano
Fallback: gpt-4o-mini

## Logging Tables
flashcard_generations
flashcard_generation_cards
flashcard_feedback (future user edits)

## Env
See .env.example

## Rate Limits
10 generation requests per minute per process (in-memory).

## Metrics (GET /metrics)
totalRequests
generationRequests
generationSuccess
generationError
avgLatencyMs

## Future
- Feedback endpoint
- Embeddings for duplicate avoidance
- Fine-tuning only after dataset large enough