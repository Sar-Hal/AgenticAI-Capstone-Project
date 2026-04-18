from datetime import datetime
from langchain_core.tools import tool

@tool
def calculate_project_deadline(query: str = "") -> str:
    """
    Calculates the time remaining until the Capstone Project deadline.
    The deadline is strictly April 21, 2026 at 11:59 PM.
    Always returns a string — never raises exceptions.
    """
    try:
        deadline = datetime(2026, 4, 21, 23, 59, 59)
        now = datetime.now()

        if now > deadline:
            return "The Capstone Project deadline (April 21, 2026, 11:59 PM) has already passed!"

        time_remaining = deadline - now
        days = time_remaining.days
        hours, remainder = divmod(time_remaining.seconds, 3600)
        minutes, _ = divmod(remainder, 60)

        return (
            f"The Capstone Project deadline is April 21, 2026 at 11:59 PM. "
            f"You have {days} day(s), {hours} hour(s), and {minutes} minute(s) remaining. "
            f"No extensions will be granted under any circumstances."
        )
    except Exception as e:
        return f"Error calculating deadline: {str(e)}"
