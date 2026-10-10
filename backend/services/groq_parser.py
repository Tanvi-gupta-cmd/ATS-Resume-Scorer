import os
import json
import logging
from typing import Dict

from groq import Groq

logger = logging.getLogger("ats_resume_scorer")

GROQ_MODEL = "openai/gpt-oss-120b"

_client = None


def _get_client() -> Groq:
    global _client

    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")

        _client = Groq(api_key=api_key)

    return _client


RESUME_SYSTEM_PROMPT = (
    "You are a resume parser. Extract information from the resume "
    "and return ONLY a valid JSON object. No explanation, no markdown."
)


RESUME_USER_PROMPT = """Extract the following from this resume and return as JSON:
{{
  "name": "full name",
  "email": "email address",
  "phone": "phone number",
  "linkedin": "LinkedIn URL if present, otherwise null",
  "github": "GitHub URL if present, otherwise null",
  "professional_summary": "Full professional summary or empty string",
  "skills": ["list of skills"],
  "experience": [
    {{
      "job_title": "",
      "company": "",
      "start_date": "",
      "end_date": "",
      "duration_months": 0,
      "description": ""
    }}
  ],
  "education": [
    {{
      "degree": "",
      "institution": "",
      "year": ""
    }}
  ],
  "certifications": ["list of certifications"],
  "projects": [
    {{
      "title": "project name",
      "description": "what the project does and how it was built",
      "technologies": ["technologies used"]
    }}
  ],
  "action_verbs": ["strong action verbs used in bullet points"],
  "keywords": ["important keywords and phrases for ATS matching"]
}}

Important instructions:
- Calculate duration_months from start_date and end_date when possible.
- Extract all technical and soft skills mentioned anywhere in the resume.
- Extract action verbs used in achievements.
- Extract important technical terms and ATS keywords.
- Return ONLY valid JSON. No markdown code fences or explanations.

Resume Text:
{raw_text}"""


def _call_groq(client: Groq, system_prompt: str, user_prompt: str) -> str:
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=4096,
    )

    return (response.choices[0].message.content or "").strip()


def _try_parse_json(text: str):
    cleaned = text.strip()

    if cleaned.startswith("```"):
        first_newline = cleaned.find("\n")
        if first_newline != -1:
            cleaned = cleaned[first_newline + 1:]

        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return None


def _validate_resume_result(result: dict) -> Dict:
    if not isinstance(result, dict):
        result = {}

    list_fields = [
        "skills",
        "experience",
        "education",
        "certifications",
        "projects",
        "action_verbs",
        "keywords",
    ]

    for field in list_fields:
        if not isinstance(result.get(field), list):
            result[field] = []

    string_fields = [
        "name",
        "email",
        "phone",
        "linkedin",
        "github",
        "professional_summary",
    ]

    for field in string_fields:
        if not isinstance(result.get(field), str):
            result[field] = ""

    for project in result["projects"]:
        if not isinstance(project, dict):
            continue

        for field in ("title", "description"):
            if not isinstance(project.get(field), str):
                project[field] = ""

        if not isinstance(project.get("technologies"), list):
            project["technologies"] = []

    for experience in result["experience"]:
        if not isinstance(experience, dict):
            continue

        if not isinstance(experience.get("duration_months"), (int, float)):
            experience["duration_months"] = 0

    return result


def parse_resume(raw_text: str) -> Dict:
    client = _get_client()
    prompt = RESUME_USER_PROMPT.format(raw_text=raw_text)

    raw_response = _call_groq(
        client,
        RESUME_SYSTEM_PROMPT,
        prompt,
    )

    result = _try_parse_json(raw_response)

    if result is not None:
        return _validate_resume_result(result)

    logger.warning("Groq resume parse returned invalid JSON; retrying.")

    strict_prompt = (
        "Return ONLY a valid JSON object. Do not include markdown, "
        "explanations, or code fences.\n\n"
        + prompt
    )

    raw_response = _call_groq(
        client,
        RESUME_SYSTEM_PROMPT,
        strict_prompt,
    )

    result = _try_parse_json(raw_response)

    if result is not None:
        return _validate_resume_result(result)

    raise ValueError(
        "Groq returned invalid JSON after retry. "
        f"Raw response: {raw_response[:500]}"
    )


JOB_DESCRIPTION_SYSTEM_PROMPT = (
    "You are a job description parser. "
    "Return only a valid JSON object without markdown."
)


JOB_DESCRIPTION_USER_PROMPT = """Extract job requirements from this description.
Return JSON with these keys:
{{
  "keywords": ["important job keywords"],
  "required_skills": ["mandatory skills"],
  "preferred_skills": ["preferred skills"]
}}

Job Description:
{job_description}
"""


def parse_job_description(job_description: str) -> Dict:
    client = _get_client()

    prompt = JOB_DESCRIPTION_USER_PROMPT.format(
        job_description=job_description
    )

    raw_response = _call_groq(
        client,
        JOB_DESCRIPTION_SYSTEM_PROMPT,
        prompt,
    )

    result = _try_parse_json(raw_response)

    if result is None:
        strict_prompt = (
            "Return only a valid JSON object with keys "
            "keywords, required_skills, preferred_skills.\n\n"
            + prompt
        )

        raw_response = _call_groq(
            client,
            JOB_DESCRIPTION_SYSTEM_PROMPT,
            strict_prompt,
        )

        result = _try_parse_json(raw_response)

    if not isinstance(result, dict):
        raise ValueError("Could not parse the job description response.")

    for field in ("keywords", "required_skills", "preferred_skills"):
        if not isinstance(result.get(field), list):
            result[field] = []

    return result