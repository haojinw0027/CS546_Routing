"""
Adaptive reasoning metric for prompt optimization.

This metric evaluates both correctness and reasoning efficiency,
allowing the model to self-reflect on whether it's overthinking or underthinking.
"""

import dspy


def estimate_reasoning_length(prediction):
    """
    Estimate the length of model's reasoning.

    Args:
        prediction: DSPy prediction result

    Returns:
        int: Word count of reasoning
    """
    # DSPy ChainOfThought includes rationale in prediction
    reasoning_text = getattr(prediction, 'rationale', '')
    if not reasoning_text:
        reasoning_text = str(prediction)

    word_count = len(reasoning_text.split())
    return word_count


def check_correctness(example, prediction):
    """
    Check if the answer is correct.

    Args:
        example: Ground truth example
        prediction: Model prediction

    Returns:
        tuple: (is_correct, predicted_answer, correct_answer)
    """
    task_type = example.get('task_type', 'math')
    correct_answer = example['answer']

    try:
        if task_type == 'arc':
            # ARC: single letter answer
            predicted_answer = str(prediction.answer).strip().upper()
            correct_answer = str(correct_answer).strip().upper()
            is_correct = (predicted_answer == correct_answer)
        else:
            # MATH: integer answer
            predicted_answer = int(prediction.answer)
            correct_answer = int(correct_answer) if isinstance(correct_answer, int) else correct_answer
            is_correct = (predicted_answer == correct_answer)
    except (ValueError, AttributeError):
        is_correct = False
        predicted_answer = str(getattr(prediction, 'answer', 'NO_ANSWER'))

    return is_correct, predicted_answer, correct_answer


def metric_with_adaptive_reasoning_feedback(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """
    Evaluate correctness and reasoning efficiency with feedback for reflection model.

    This metric provides simple feedback asking the model to self-reflect on:
    - Is the answer correct?
    - Might the model be overthinking (too verbose)?
    - Should the model think twice (too hasty)?

    Returns:
        dspy.Prediction with:
            - score: float between 0 and 1
            - feedback: feedback text for reflection
    """
    # Get task type
    task_type = example.get('task_type', 'unknown')

    # Check correctness
    is_correct, predicted_answer, correct_answer = check_correctness(example, prediction)

    # Get reasoning length
    reasoning_length = estimate_reasoning_length(prediction)

    # Build feedback
    feedback_parts = []

    feedback_parts.append("EVALUATION RESULT:")
    feedback_parts.append(f"Predicted Answer: {predicted_answer}")
    feedback_parts.append(f"Correct Answer: {correct_answer}")
    feedback_parts.append(f"Result: {'CORRECT' if is_correct else 'INCORRECT'}")
    feedback_parts.append(f"Reasoning Length: {reasoning_length} words")
    feedback_parts.append("")

    # Provide reflection-oriented feedback
    if not is_correct:
        feedback_parts.append("FEEDBACK:")
        feedback_parts.append("The answer is incorrect.")

        if reasoning_length < 20:
            feedback_parts.append("The reasoning seems very brief. Should the model think twice?")
            feedback_parts.append("Consider: Does this problem require more careful analysis?")
        elif reasoning_length > 200:
            feedback_parts.append("The reasoning is quite lengthy, yet still wrong.")
            feedback_parts.append("Consider: Is the model overthinking in the wrong direction?")
        else:
            feedback_parts.append("Consider: What reasoning steps were missing or incorrect?")

    else:
        feedback_parts.append("FEEDBACK:")
        feedback_parts.append("The answer is correct.")

        if reasoning_length > 300:
            feedback_parts.append("However, the reasoning is very lengthy.")
            feedback_parts.append("Consider: Might the model be overthinking this problem?")
            feedback_parts.append("Could this be solved more efficiently?")
        elif reasoning_length < 10:
            feedback_parts.append("The reasoning is very brief.")
            feedback_parts.append("Consider: Is the model getting lucky, or is this genuinely simple?")
        else:
            feedback_parts.append("The reasoning length seems reasonable.")

    feedback_parts.append("")
    feedback_parts.append("REFLECTION QUESTION:")
    feedback_parts.append("How should the prompt guide the model to:")
    feedback_parts.append("1. Self-assess problem complexity?")
    feedback_parts.append("2. Decide when to think deeply vs. when to be concise?")
    feedback_parts.append("3. Avoid both overthinking and underthinking?")

    feedback_text = "\n".join(feedback_parts)

    # Calculate score
    correctness_score = 1.0 if is_correct else 0.0

    # Gentle efficiency bonus/penalty (±20%)
    if is_correct:
        if reasoning_length > 300:
            efficiency_factor = 0.8  # Mild penalty for overthinking
        elif reasoning_length < 10:
            efficiency_factor = 0.9  # Slight penalty for too brief
        else:
            efficiency_factor = 1.0  # No penalty
    else:
        efficiency_factor = 1.0  # Don't penalize efficiency if wrong

    final_score = correctness_score * efficiency_factor

    return dspy.Prediction(score=final_score, feedback=feedback_text)


def metric_simple_adaptive(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """
    Simplified metric for evaluation (without detailed feedback).
    """
    is_correct, predicted_answer, correct_answer = check_correctness(example, prediction)
    reasoning_length = estimate_reasoning_length(prediction)
    task_type = example.get('task_type', 'unknown')

    # Simple feedback
    feedback += f"Result: {'CORRECT' if is_correct else 'WRONG'}\n"
    feedback += f"Expected: {correct_answer}, Got: {predicted_answer}\n"
    feedback += f"Reasoning: {reasoning_length} words\n"

    if is_correct and reasoning_length > 300:
        feedback += "Note: Might be overthinking.\n"
    elif not is_correct and reasoning_length < 20:
        feedback += "Note: Should think twice.\n"

    # Score
    correctness_score = 1.0 if is_correct else 0.0

    if is_correct and reasoning_length > 300:
        efficiency_penalty = 0.2
    else:
        efficiency_penalty = 0.0

    final_score = max(0, correctness_score - efficiency_penalty)

    return dspy.Prediction(score=final_score, feedback=feedback)
