import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

from providers import LLMResponse, build_providers, fanout_generate, pick_provider, synthesize


PROMPTS = {
    "planner": (
        "Create a lecture plan for a university programming course. "
        "Include learning objectives, topics, timing, and required prereqs. "
        "Return markdown with sections: Objectives, Outline, Timing, Prereqs."
    ),
    "content": (
        "Draft explanations and examples for the lecture. "
        "Include 2 short code examples and 1 in-class exercise. "
        "Return markdown with sections: Concepts, Examples, Exercise."
    ),
    "quiz": (
        "Create 5 quiz questions with answers. "
        "Use a mix of MCQ and short answer. Return markdown."
    ),
    "critic": (
        "Review the content for correctness, clarity, and pedagogy. "
        "Return bullet-point feedback and suggested fixes."
    ),
    "slides": (
        "Compose a slide outline with slide titles and bullet points. "
        "Return JSON list of slides: [{title, bullets}]."
    ),
}


def merge_responses(responses: List[LLMResponse]) -> str:
    return "\n\n".join(
        [f"## Provider: {response.provider}\n{response.content}" for response in responses]
    )


def build_prompt(stage: str, topic: str, level: str, weeks: int) -> str:
    base = PROMPTS[stage]
    return (
        f"Topic: {topic}\n"
        f"Audience: {level}\n"
        f"Weeks: {weeks}\n\n"
        f"{base}"
    )


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def extract_json(text: str) -> str | None:
    match = re.search(r"\\{.*\\}|\\[.*\\]", text, re.DOTALL)
    return match.group(0) if match else None


def slides_markdown(slides: List[Dict[str, object]]) -> str:
    sections: List[str] = []
    for idx, slide in enumerate(slides, start=1):
        title = slide.get("title", f"Slide {idx}")
        bullets = slide.get("bullets", [])
        bullet_lines = []
        if isinstance(bullets, list):
            bullet_lines = [f"- {item}" for item in bullets]
        sections.append("\n".join([f"## {title}"] + bullet_lines))
    return "\n\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-LLM lecture slide pipeline")
    parser.add_argument("--topic", required=True)
    parser.add_argument("--level", default="Undergraduate")
    parser.add_argument("--weeks", type=int, default=1)
    parser.add_argument("--output", required=True)
    parser.add_argument("--synth-provider", default=None, help="Provider to synthesize outputs")
    args = parser.parse_args()

    providers = build_providers()
    if not providers:
        raise SystemExit("No API keys found. Set OPENAI_API_KEY, GLM_API_KEY, or GROQ_API_KEY.")

    output_dir = Path(args.output)
    ensure_output_dir(output_dir)

    stage_outputs: Dict[str, Dict[str, str]] = {}
    synth_provider = pick_provider(args.synth_provider, providers)

    for stage in ["planner", "content", "quiz", "critic", "slides"]:
        prompt = build_prompt(stage, args.topic, args.level, args.weeks)
        responses = fanout_generate(prompt, providers)
        merged = merge_responses(responses)
        synthesized = synthesize(prompt, responses, synth_provider)
        stage_outputs[stage] = {
            "merged": merged,
            "synthesized": synthesized.content,
        }
        write_text(output_dir / f"{stage}.raw.md", merged)
        write_text(output_dir / f"{stage}.md", synthesized.content)

    slides_text = stage_outputs["slides"]["synthesized"]
    slides_json_text = extract_json(slides_text)
    slides_data = []
    if slides_json_text:
        try:
            slides_data = json.loads(slides_json_text)
        except json.JSONDecodeError:
            slides_data = []

    if slides_data:
        write_text(output_dir / "slides.md", slides_markdown(slides_data))

    slides_spec = {
        "topic": args.topic,
        "level": args.level,
        "weeks": args.weeks,
        "files": {stage: f"{stage}.md" for stage in stage_outputs},
        "raw_files": {stage: f"{stage}.raw.md" for stage in stage_outputs},
    }
    write_json(output_dir / "slides.json", slides_spec)


if __name__ == "__main__":
    main()
