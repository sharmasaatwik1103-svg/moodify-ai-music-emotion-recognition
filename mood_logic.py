def mood_from_scores(valence, arousal):
    # Thresholds are a simple product convention, not labels supplied by DEAM.
    v_high = valence >= 5.0
    a_high = arousal >= 5.0
    if v_high and a_high:
        return "Happy / Energetic"
    if not v_high and a_high:
        return "Tense / Intense"
    if v_high and not a_high:
        return "Calm / Pleasant"
    return "Sad / Low-energy"

def normalize_score(value):
    return max(0.0, min(1.0, (float(value) - 1.0) / 8.0))
