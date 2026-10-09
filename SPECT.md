# Intelligent Conversational Agent
## Overview

The system is a conversational AI agent that conducts short interviews with students of a magic school to gather information about their profile and alignment, and recommends which classes and clubs they should apply to join.

## Core Requirements

1. **Conversational Flow**
    - Implement a bot that can engage in natural conversation with users
    - Open with a one-line consent message explaining what data is collected
    - The bot should collect the following information:
        - Core profile:
            - school_year: integer (1-7)
            - specialization: ALCHEMY | CONJURING | DEFENSE | HEALING
            - interests: free text
        - Alignment:
            - magic_orientation: KNOWLEDGE | POWER | ALTRUISM | ADVENTURE
            - learning_modality: PRACTICAL | THEORETICAL | COLLABORATIVE
        - Constraints:
            - weekly_free_hours: number (>= 0)
    - Extract and validate this information into a structured format
    - Handle basic error cases and invalid inputs gracefully (invalid values, corrections, off-topic answers, refusals)
    - Maintain conversation context throughout the interaction
    - Confirm the collected profile with the student before recommending
2. **Recommendations**
    - Knowledge base: a JSON catalog of ~15 clubs and advanced classes, each with:
        - Descriptive: name, category, description
        - Hard requirements: min_school_year, min_free_hours_required, excluded_specialization
        - Ideal profile: magic_orientation, learning_modality
    - Filter the catalog on hard requirements in Python, then pass the remaining entries to the LLM to rank and explain
    - If nothing passes the filters, say so and explain why
3. **Data Extraction & Storage**
    - Store conversations and extracted data in a structured format (JSON)
    - Implement basic data validation for collected information
    - Generate a summary of the conversation with extracted key points and recommendations
4. **Technical Requirements**
    - Use Python for implementation
    - Implement proper error handling (LLM timeouts, malformed output, STT/TTS failures)
    - Use abstract classes for the LLM, STT, TTS and storage, so that implementations are easy to swap
    - Implementations:
        - LLM: Claude, plus a fake LLM for tests
        - STT: Deepgram Nova-3 (default), or ElevenLabs Scribe v2 selected by configuration
        - TTS: ElevenLabs Flash v2.5
        - Storage: JSON files
    - Voice mode: push-to-talk (record → transcribe → reply → speak). Text mode runs the same engine and is used by the tests
    - Tests with pytest against the fake LLM: validation, corrections, catalog filtering, error cases and a few scripted conversations

## Bonus Features

- Sentiment analysis to detect user frustration, adapting the bot's replies
- Support for multiple languages (the bot replies in the student's language)
- Multi-turn conversation memory (covered by the conversation history; document it in the README)

## Deliverables

1. Source code in a Git repository
2. README with:
    - Setup instructions
    - System architecture overview
    - Explanation of key design decisions
    - Scalability considerations
    - Description of potential improvements (e.g. local Ollama adapter, hybrid RAG with embeddings, database storage)
3. URL to a demo video of sample conversations demonstrating the bot's capabilities
