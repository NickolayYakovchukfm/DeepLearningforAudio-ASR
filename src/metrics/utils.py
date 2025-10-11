# Based on seminar materials
import editdistance

# Don't forget to support cases when target_text == ''


def calc_cer(target_text, predicted_text) -> float:
    """
    Input:
    target_text (str)
    predicted_text (str)

    Output:
    cer (1, )
    """
    assert (
        len(target_text) != 0
    ), f"!!!ALERT: watch this on calc_cer: target_text_len = {len(target_text)}"
    return editdistance.eval(target_text, predicted_text) / len(target_text)


def calc_wer(target_text, predicted_text) -> float:
    """
    Input:
    target_text (str)
    predicted_text (str)

    Output:
    wer (1, )
    """
    assert (
        len(target_text) != 0
    ), f"!!!ALERT: watch this on calc_wer : target_text_len = {len(target_text)}"
    return editdistance.eval(target_text.split(), predicted_text.split()) / len(
        target_text.split()
    )
