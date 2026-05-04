PRESET_EXPECTED_ITEMS: dict[str, list[str]] = {
    "School Commute": ["bag", "laptop", "book", "bottle", "umbrella"],
    "Work Session": ["laptop", "mouse", "keyboard", "cup"],
}

PRESET_DESCRIPTIONS: dict[str, str] = {
    "School Commute": (
        "Checks for everyday commute essentials in your bag setup "
        "(for example: laptop, book, bottle, umbrella)."
    ),
    "Work Session": (
        "Checks for a focused desk/work setup and flags common distractions "
        "(for example: cell phone, bag, book)."
    ),
}

PRESET_PROMPTS: dict[str, str] = {
    "School Commute": (
        "You are a helpful assistant for a student preparing for their commute.\n"
        "The user's bag has been scanned and the following items were detected: {labels}.\n"
        "The expected items for a school commute are: bag, laptop, book, bottle, umbrella.\n"
        "Identify what is missing and provide brief, practical contextual tips.\n"
        "For example, if an umbrella is missing, suggest checking the weather.\n"
        "Keep the response friendly and concise."
    ),
    "Work Session": (
        "You are a focus assistant helping a user prepare for a productive work session.\n"
        "The following items were detected in the user's workspace: {labels}.\n"
        "The expected items for a work session are: laptop, mouse, keyboard, cup.\n"
        "Do two things:\n"
        "1. Flag any expected items that are missing.\n"
        "2. Flag any detected distraction items (cell phone, book, bag) and advise the user to remove them.\n"
        "Keep the tone motivating but direct."
    ),
}
