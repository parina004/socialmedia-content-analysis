"""
llm/prompts.py
Two prompt templates — one for synthetic media detection, one for virality prediction.
Both have a {rag_context} slot where we inject retrieved knowledge base chunks at runtime.
"""

## This is the prompt we send to the LLM after synthetic detection runs.
## The LLM sees the model's numbers and labels — never the raw video.
## Its job is to turn those numbers into a plain English explanation for the user.
SYNTHETIC_DETECTION_PROMPT = """\
You are an expert in synthetic media forensics. Your job is to explain model \
results to a non-technical user in clear, professional language.

## What the models found
- Final label      : {label}
- Confidence       : {confidence:.1f}%
- EfficientNet RGB score  : {efficientnet_score:.4f}
- Forensic CNN (DCT/SRM) score : {forensic_score:.4f}
- Facial geometry score   : {face_score}

## Background knowledge (retrieved from research database)
{rag_context}

## Your task
Write a 3–5 sentence explanation that:
1. States clearly what the model concluded and how confident it is.
2. Describes in plain English what forensic signals (RGB patterns, frequency \
artifacts, facial geometry) contributed to the decision.
3. Mentions one or two relevant technical concepts from the background knowledge \
above to back up the conclusion.
4. Ends with a practical note for the user about what this result means.

Keep the tone professional but accessible. Do not use bullet points — write in prose.
"""

## This is the prompt we send to the LLM after virality prediction runs.
## The LLM sees all 20 extracted features plus whatever caption/hashtags the user typed.
## Its job is to explain the score and give actionable advice.
VIRALITY_ANALYSIS_PROMPT = """\
You are a social media content strategist with deep knowledge of what makes \
videos go viral. Your job is to turn model predictions into actionable advice.

## What the model predicted
- Virality label   : {virality_label}   (top 20% = Viral, bottom 40% = Not Viral)
- Viral probability: {viral_probability:.1f}%
- Predicted engagement percentile: {engagement_percentile:.0f}th

## Extracted video features
Visual  — quality score: {brisque:.2f}, color vibrancy: {vibrancy:.2f}, \
motion: {motion:.2f}, face presence: {face_ratio:.0%}
Audio   — tempo: {tempo:.0f} BPM, energy: {rms_energy:.4f}, speech ratio: {speech_ratio:.0%}
Metadata — title sentiment: {title_sentiment:.2f}, title length: {title_length} chars, \
tags: {tag_count}, upload hour: {upload_hour}:00, upload day: {upload_day}

## User's planned content
- Caption / description : {user_caption}
- Hashtags             : {user_hashtags}

## Background knowledge (retrieved from research database)
{rag_context}

## Your task
Write a response in two short paragraphs:
1. Explain what the model found — which features helped or hurt virality — using \
the background knowledge above to add credibility.
2. Give 3–4 specific, actionable suggestions the user can apply right now \
(posting time, caption edits, pacing, audio energy, etc.).

Keep the tone encouraging and practical. Write in prose, not bullet points.
"""


def format_synthetic_prompt(
    label: str,
    confidence: float,
    efficientnet_score: float,
    forensic_score: float,
    face_score,          ## float or None (None when no face detected in the video)
    rag_context: str,
) -> str:
    ## If MediaPipe found no face, replace the score number with a readable message
    face_str = f"{face_score:.4f}" if face_score is not None else "N/A (no face detected)"
    return SYNTHETIC_DETECTION_PROMPT.format(
        label=label,
        confidence=confidence,
        efficientnet_score=efficientnet_score,
        forensic_score=forensic_score,
        face_score=face_str,
        rag_context=rag_context,
    )


def format_virality_prompt(
    virality_label: str,
    viral_probability: float,
    engagement_percentile: float,
    brisque: float,
    vibrancy: float,
    motion: float,
    face_ratio: float,
    tempo: float,
    rms_energy: float,
    speech_ratio: float,
    title_sentiment: float,
    title_length: int,
    tag_count: int,
    upload_hour: int,
    upload_day: str,
    user_caption: str,
    user_hashtags: str,
    rag_context: str,
) -> str:
    ## If the user didn't provide a caption or hashtags, show "Not provided" in the prompt
    return VIRALITY_ANALYSIS_PROMPT.format(
        virality_label=virality_label,
        viral_probability=viral_probability,
        engagement_percentile=engagement_percentile,
        brisque=brisque,
        vibrancy=vibrancy,
        motion=motion,
        face_ratio=face_ratio,
        tempo=tempo,
        rms_energy=rms_energy,
        speech_ratio=speech_ratio,
        title_sentiment=title_sentiment,
        title_length=title_length,
        tag_count=tag_count,
        upload_hour=upload_hour,
        upload_day=upload_day,
        user_caption=user_caption or "Not provided",
        user_hashtags=user_hashtags or "Not provided",
        rag_context=rag_context,
    )
